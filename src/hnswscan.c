#include "postgres.h"

#include "access/relscan.h"
#include "hnsw.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"

/*
 * Algorithm 5 from paper
 */
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL);
		ep = w;
	}

	return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, false, NULL, &so->v, hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples);
}

static List *
GetScanItemsColumn(IndexScanDesc scan, Datum value, int col)
{
	HnswScanOpaqueMulti soMulti = (HnswScanOpaqueMulti) scan->opaque;
	Relation index = scan->indexRelation;

	if (col < 0 || col >= soMulti->nkeys)
		elog(ERROR, "hnsw scan col out of range: col=%d nkeys=%d", col, soMulti->nkeys);

	HnswScanOpaque so = &soMulti->cols[col];          /* 当前列自己的 opaque */
	HnswSupport *support = &so->support;

	List *ep;
	List *w;
	int m;
	HnswElement entryPoint;
	char *base = NULL;
	HnswQuery *q = &so->q;

	/*
	 * ✅ 关键差异：按列读取 m / entryPoint
	 * - 旧布局时 HnswGetMetaPageInfoMulti(col=0) 会兼容
	 * - 新布局时会取 graphs[col]
	 */
	HnswGetMetaPageInfoMulti(index, col, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc,
							index, support, m,
							false,
							NULL, NULL, NULL,
							true,
							NULL);
		ep = w;
	}

	return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0,
						   index, support, m,
						   false,
						   NULL,
						   &so->v,
						   (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF) ? &so->discarded : NULL,
						   true,
						   &so->tuples);
}


/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;
	int			batch_size = hnsw_ef_search;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

		ep = lappend(ep, sc);
	}

	return HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples);
}


static List *
ResumeScanItemsColumn(IndexScanDesc scan, int col)
{
	HnswScanOpaqueMulti soMulti = (HnswScanOpaqueMulti) scan->opaque;
	Relation index = scan->indexRelation;

	if (col < 0 || col >= soMulti->nkeys)
		elog(ERROR, "hnsw scan col out of range: col=%d nkeys=%d", col, soMulti->nkeys);

	HnswScanOpaque so = &soMulti->cols[col];          /* 当前列自己的 opaque */

	List *ep = NIL;
	char *base = NULL;
	int batch_size = hnsw_ef_search;

	if (so->discarded == NULL || pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));
		ep = lappend(ep, sc);
	}

	return HnswSearchLayer(base, &so->q, ep, batch_size, 0,
						   index, &so->support, so->m,
						   false,
						   NULL,
						   &so->v,
						   &so->discarded,
						   false,
						   &so->tuples);
}


/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

#if defined(HNSW_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	scan->opaque = so;

	return scan;
}

/*
 * Prepare for an index scan (multi-column)
 */
/*
 * Prepare for an index scan (multi-column)
 * - 每列一份 HnswScanOpaqueData，完全独立（允许冗余但绝不串列）
 */
IndexScanDesc
hnswbeginscanmulti(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaqueMulti soMulti;
	double maxMemory;
	int nkeys_index;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	/* key 列数（不含 INCLUDE） */
	nkeys_index = IndexRelationGetNumberOfKeyAttributes(index);

	soMulti = (HnswScanOpaqueMulti) palloc0(sizeof(HnswScanOpaqueDataMulti));
	soMulti->nkeys = nkeys_index;

	/*
	 * beginscan 阶段还无法确定 ORDER BY 使用哪一列：
	 * 先默认 0，后续你在 rescan 链路里再设置 soMulti->col
	 */
	soMulti->col = 0;

	/* 分配每列一份原版 opaque */
	soMulti->cols = (HnswScanOpaqueData *) palloc0(sizeof(HnswScanOpaqueData) * nkeys_index);

	for (int col = 0; col < nkeys_index; col++)
	{
		HnswScanOpaque so = &soMulti->cols[col];

		/* typeInfo 必须按列取，支持不同向量类型 */
		so->typeInfo = HnswGetTypeInfoColumn(index, col);

		/* support 必须按列初始化，支持不同 opclass/procs */
		HnswInitSupportColumn(&so->support, index, col);

		/*
		 * tmpCtx / maxMemory：按列各自一份（不共享，冗余但正确）
		 */
		so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
										   "Hnsw scan temporary context",
										   0, 8 * 1024, 256 * 1024);

		maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
		so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

		/* 其他字段保持 0/NULL（palloc0 已清零），后续 rescan/gettuple 再使用 */
	}

	/* opaque 放 multi 指针 */
	scan->opaque = soMulti;

	return scan;
}



IndexScanDesc
hnswbeginscan_dispatch(Relation index, int nkeys, int norderbys)
{
	int nkeys_index = IndexRelationGetNumberOfKeyAttributes(index);

	if (nkeys_index <= 1) {
		elog(LOG, "hnswbeginscan");
		return hnswbeginscan(index, nkeys, norderbys);
	}
	else {
		elog(LOG, "hnswbeginscanmulti");
		return hnswbeginscanmulti(index, nkeys, norderbys);
	}

}



/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->tuples = 0;
	so->previousDistance = -get_float8_infinity();
	MemoryContextReset(so->tmpCtx);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

void
hnswrescanmulti(IndexScanDesc scan, ScanKey keys, int nkeys,
				ScanKey orderbys, int norderbys)
{
	HnswScanOpaqueMulti soMulti = (HnswScanOpaqueMulti) scan->opaque;

	/* 先根据 orderbys 决定本次 scan 用哪一列（0-based） */
	int col = 0;
	if (orderbys != NULL && norderbys > 0)
	{
		int attno = orderbys[0].sk_attno;  /* 1-based index column number */
		if (attno > 0)
			col = attno - 1;
	}
	if (col < 0 || col >= soMulti->nkeys)
		col = 0;

	soMulti->col = col;

	/* 重置每一列的运行态状态（最安全，避免上一次 scan 的状态残留） */
	for (int i = 0; i < soMulti->nkeys; i++)
	{
		HnswScanOpaque so = &soMulti->cols[i];

		so->first = true;
		/* v and discarded are allocated in tmpCtx */
		so->v.tids = NULL;
		so->discarded = NULL;
		so->tuples = 0;
		so->previousDistance = -get_float8_infinity();
		MemoryContextReset(so->tmpCtx);
	}

	/* 拷贝 keys / orderbys（保持与原版一致） */
	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}


void
hnswrescan_dispatch(IndexScanDesc scan, ScanKey keys, int nkeys,
					ScanKey orderbys, int norderbys)
{
	int nkeys_index = IndexRelationGetNumberOfKeyAttributes(scan->indexRelation);

	if (nkeys_index <= 1)
		hnswrescan(scan, keys, nkeys, orderbys, norderbys);
	else
		hnswrescanmulti(scan, keys, nkeys, orderbys, norderbys);
}


/*
 * Fetch the next tuple in the given scan
 */
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->w = GetScanItems(scan, value);

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer heaptid;

		if (list_length(so->w) == 0)
		{
			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
				break;

			/* Empty index */
			if (so->discarded == NULL)
				break;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					break;

				/* Return remaining tuples */
				so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
			}
			else
			{
				/*
				 * Locking ensures when neighbors are read, the elements they
				 * reference will not be deleted (and replaced) during the
				 * iteration.
				 *
				 * Elements loaded into memory on previous iterations may have
				 * been deleted (and replaced), so when reading neighbors, the
				 * element version must be checked.
				 */
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				so->w = ResumeScanItems(scan);

				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				break;
		}

		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
				continue;

			so->previousDistance = sc->distance;
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

static int
HnswGetOrderByCol(IndexScanDesc scan)
{
	int nkeys;
	int attno;

	/* hnsw 必须有 order by（原版也会这样检查） */
	if (scan->orderByData == NULL || scan->numberOfOrderBys <= 0)
		elog(ERROR, "cannot scan hnsw index without order");

	/*
	 * sk_attno: 1-based index attribute number (key attrs)
	 * 对应 planner 选择的“哪个 index key 列”来做 orderby
	 */
	attno = scan->orderByData[0].sk_attno;

	/* key 列数量（不含 INCLUDE） */
	nkeys = IndexRelationGetNumberOfKeyAttributes(scan->indexRelation);

	if (attno <= 0 || attno > nkeys)
		elog(ERROR, "invalid hnsw orderby attno=%d (nkeys=%d)", attno, nkeys);

	/* 防御：如果有多个 orderby，要求它们都指向同一列 */
	for (int i = 1; i < scan->numberOfOrderBys; i++)
	{
		if (scan->orderByData[i].sk_attno != attno)
			elog(ERROR,
				 "multi-orderby not supported for hnsw multi-column scan: "
				 "orderby attno mismatch (%d vs %d)",
				 attno, scan->orderByData[i].sk_attno);
	}

	return attno - 1; /* 0-based col */
}

bool
hnswgettuplenew(IndexScanDesc scan, ScanDirection dir)
{
    HnswScanOpaqueMulti soMulti = (HnswScanOpaqueMulti) scan->opaque;
    int col;

    /* Index can be used to scan backward, but Postgres doesn't support backward scan on operators */
    Assert(ScanDirectionIsForward(dir));

    /* Safety check */
    if (scan->orderByData == NULL)
        elog(ERROR, "cannot scan hnsw index without order");

    /* Requires MVCC-compliant snapshot as not able to maintain a pin */
    if (!IsMVCCSnapshot(scan->xs_snapshot))
        elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

    /* 关键：从 orderByData 决定当前列（0-based） */
    col = HnswGetOrderByCol(scan);

    if (col < 0 || col >= soMulti->nkeys)
        elog(ERROR, "hnsw scan col out of range: col=%d nkeys=%d", col, soMulti->nkeys);

    /* 当前列的单列 opaque（每列一份状态） */
    HnswScanOpaque so = &soMulti->cols[col];

    /* 从这一列的 tmpCtx 开始工作 */
    MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

    if (so->first)
    {
        Datum value;

        /* Count index scan for stats */
        pgstat_count_index_scan(scan->indexRelation);

        /* Get scan value */
        value = GetScanValue(scan);

        /*
         * Get a shared lock. This allows vacuum to ensure no in-flight scans
         * before marking tuples as deleted.
         */
        LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

        /* 关键改动 1：按列取 scan items（scan->opaque 仍是 Multi） */
        so->w = GetScanItemsColumn(scan, value, col);

        UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

        so->first = false;

#if defined(HNSW_MEMORY)
        ShowMemoryUsage(so);
#endif
    }

    for (;;)
    {
        char *base = NULL;
        HnswSearchCandidate *sc;
        HnswElement element;
        ItemPointer heaptid;

        if (list_length(so->w) == 0)
        {
            if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
                break;

            /* Empty index */
            if (so->discarded == NULL)
                break;

            /* Reached max number of tuples or memory limit */
            if (so->tuples >= hnsw_max_scan_tuples ||
                MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
            {
                if (pairingheap_is_empty(so->discarded))
                    break;

                /* Return remaining tuples */
                so->w = lappend(so->w,
                                HnswGetSearchCandidate(w_node,
                                                      pairingheap_remove_first(so->discarded)));
            }
            else
            {
                LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

                /* 关键改动 2：按列 resume（scan->opaque 仍是 Multi） */
                so->w = ResumeScanItemsColumn(scan, col);

                UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
                ShowMemoryUsage(so);
#endif
            }

            if (list_length(so->w) == 0)
                break;
        }

        sc = llast(so->w);
        element = HnswPtrAccess(base, sc->element);

        /* Move to next element if no valid heap TIDs */
        if (element->heaptidsLength == 0)
        {
            so->w = list_delete_last(so->w);

            if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
            {
                pfree(element);
                pfree(sc);
            }

            continue;
        }

        heaptid = &element->heaptids[--element->heaptidsLength];

        if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
        {
            if (sc->distance < so->previousDistance)
                continue;

            so->previousDistance = sc->distance;
        }

        MemoryContextSwitchTo(oldCtx);

        scan->xs_heaptid = *heaptid;
        scan->xs_recheck = false;
        scan->xs_recheckorderby = false;
        return true;
    }

    MemoryContextSwitchTo(oldCtx);
    return false;
}



bool
hnswgettuplemulti(IndexScanDesc scan, ScanDirection dir)
{
	return hnswgettuplenew(scan, dir);
}



bool
hnswgettuple_dispatch(IndexScanDesc scan, ScanDirection dir)
{
	int nkeys_index = IndexRelationGetNumberOfKeyAttributes(scan->indexRelation);

	if (nkeys_index <= 1)
		return hnswgettuple(scan, dir);
	else
		return hnswgettuplemulti(scan, dir);
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}

/*
 * End a scan and release resources (multi-column)
 */
void
hnswendscanmulti(IndexScanDesc scan)
{
	HnswScanOpaqueMulti soMulti = (HnswScanOpaqueMulti) scan->opaque;

	if (soMulti == NULL)
		return;

	if (soMulti->cols != NULL)
	{
		for (int i = 0; i < soMulti->nkeys; i++)
		{
			HnswScanOpaque so = &soMulti->cols[i];

			if (so->tmpCtx != NULL)
			{
				MemoryContextDelete(so->tmpCtx);
				so->tmpCtx = NULL;
			}
		}

		pfree(soMulti->cols);
		soMulti->cols = NULL;
	}

	pfree(soMulti);
	scan->opaque = NULL;
}


void
hnswendscan_dispatch(IndexScanDesc scan)
{
	int nkeys_index = IndexRelationGetNumberOfKeyAttributes(scan->indexRelation);

	if (nkeys_index <= 1)
		hnswendscan(scan);
	else
		hnswendscanmulti(scan);
}

