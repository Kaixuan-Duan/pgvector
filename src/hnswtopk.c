//
// Created by kaixu on 2026/1/20.
//
#include "postgres.h"

#include "access/genam.h"
#include "access/relscan.h"
#include "access/skey.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"
#include "storage/itemptr.h"
#include "utils/rel.h"

#include "utils/snapmgr.h"
#include <math.h>
#include "hnswtopk.h"

#include "utils/lsyscache.h"   /* get_op_rettype */
#include "utils/errcodes.h"

/*
 * 可选：如果你 multi-scan 依赖 scan->opaque->col（而不是仅靠 sk_attno 推导），
 * 可以在 hnswscan.c 里提供一个非 static 的 setter，然后这里调用。
 *
 * 例如在 hnswscan.c 中实现：
 *   void HnswScanSetColumn(IndexScanDesc scan, int col);
 *
 * 如果你的实现已经是从 orderByData[0].sk_attno 推导列（你历史里建议过这种做法），
 * 那下面这个 extern 和调用都可以删掉。
 */
#ifdef HNSW_HAVE_SCAN_SET_COLUMN
extern void HnswScanSetColumn(IndexScanDesc scan, int col);
#endif

struct HnswTopKStream
{
    IndexScanDesc scan;
    ScanKeyData orderbykey;
    bool done;
};

HnswTopKStream *
HnswTopKStreamBegin(Relation heapRel,
                    Relation indexRel,
                    int col,
                    Oid orderby_op,
                    Datum query)
{
    HnswTopKStream *stream;
    Oid orderby_proc;
    Snapshot snapshot;

    if (col < 0 || col >= IndexRelationGetNumberOfKeyAttributes(indexRel))
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("invalid col=%d for index with %d key attributes",
                        col, IndexRelationGetNumberOfKeyAttributes(indexRel))));

    if (!OidIsValid(orderby_op))
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("orderby operator OID is invalid"),
                 errhint("Pass the operator OID from the ORDER BY distance operator (e.g. <->, <#>, <=>).")));

    /*
     * 关键修复：ScanKeyEntryInitialize 的 sk_func 必须是 pg_proc 里的函数 OID
     * 不能直接用 operator OID，否则内核 fmgr 会报 cache lookup failed for function <op_oid>
     */
    orderby_proc = get_opcode(orderby_op); /* operator -> underlying function OID */
    if (!OidIsValid(orderby_proc))
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("orderby operator OID %u has no underlying procedure", orderby_op)));

    /* executor 期间一般都有 active snapshot */
    snapshot = GetActiveSnapshot();
    if (snapshot == NULL)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("no active snapshot")));

    /*
     * nkeys=0 (不走 WHERE 过滤), norderbys=1 (KNN orderby)
     * 你的 HNSW AM 会在 rescan/getnext_tid 时跑近邻搜索并按距离顺序返回 heaptid
     */
    stream = palloc0(sizeof(*stream));
    stream->scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

    /*
     * 设置 order-by key：
     * - sk_attno：1-based 的 index key attribute number
     * - sk_func：必须是函数 OID（orderby_proc）
     * - sk_argument：query 向量 datum
     */
    ScanKeyEntryInitialize(&stream->orderbykey,
                           0,                 /* flags */
                           (AttrNumber) (col + 1), /* sk_attno: 1-based key attno */
                           InvalidStrategy,   /* 对 orderby key 通常不需要 strategy */
                           InvalidOid,        /* subtype */
                           InvalidOid,        /* collation: 向量一般无 collation */
                           orderby_proc,        /* **重要**：距离算子 Oid */
                           query);            /* sk_argument: query vector Datum */

    /* 触发 AM rescan（HNSW AM 通常在这里或首次 getnext_tid 初始化候选集） */
    index_rescan(stream->scan, NULL, 0, &stream->orderbykey, 1);

#ifdef HNSW_HAVE_SCAN_SET_COLUMN
    /* 如果你的 multi 实现必须依赖 opaque->col，就强制设一下 */
    HnswScanSetColumn(stream->scan, col);
#endif

    return stream;
}

int
HnswTopKStreamNextBatch(HnswTopKStream *stream,
                        int max_batch,
                        HnswTopKItem *out,
                        int *out_nfound,
                        bool *out_done)
{
    int n = 0;

    if (out_nfound)
        *out_nfound = 0;
    if (out_done)
        *out_done = stream == NULL || stream->done;

    if (stream == NULL || stream->done || max_batch <= 0)
        return 0;

    while (n < max_batch)
    {
        bool found = index_getnext_tid(stream->scan, ForwardScanDirection);
        if (!found)
        {
            stream->done = true;
            break;
        }

        if (!ItemPointerIsValid(&stream->scan->xs_heaptid))
            ereport(ERROR,
                    (errmsg("HnswTopKStreamNextBatch: index_getnext_tid returned invalid TID")));

        out[n].tid = stream->scan->xs_heaptid;

        /*
         * 不读取 scan->xs_orderbyvals / xs_orderbynulls：
         * - 你当前 RRF 评分不需要 distance
         * - 避免 AM 未正确维护 xs_orderby* 导致 SIGSEGV
         */
        out[n].distance = 0.0;

        n++;
    }

    if (out_nfound)
        *out_nfound = n;
    if (out_done)
        *out_done = stream->done;

    return n;
}

void
HnswTopKStreamEnd(HnswTopKStream *stream)
{
    if (stream == NULL)
        return;

    if (stream->scan != NULL)
        index_endscan(stream->scan);

    pfree(stream);
}

int
HnswTopKForColumn(Relation heapRel,
                  Relation indexRel,
                  int col,
                  Oid orderby_op,     /* operator OID: <-> / <#> / <=> */
                  Datum query,
                  int topk,
                  HnswTopKItem *out,
                  int *out_nfound)
{
    HnswTopKStream *stream;
    int n = 0;
    bool done = false;

    if (out_nfound)
        *out_nfound = 0;

    if (topk <= 0)
        return 0;

    stream = HnswTopKStreamBegin(heapRel, indexRel, col, orderby_op, query);

    while (n < topk && !done)
    {
        int got = 0;

        HnswTopKStreamNextBatch(stream, topk - n, out + n, &got, &done);
        n += got;
    }

    HnswTopKStreamEnd(stream);

    if (out_nfound)
        *out_nfound = n;

    return n;
}
