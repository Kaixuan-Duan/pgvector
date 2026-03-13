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
#include "access/tableam.h"        /* 提供 table_index_fetch_tuple 等函数 */
#include "executor/tuptable.h"     /* 提供 TupleTableSlot 结构体和 slot_getattr */
#include "executor/executor.h"     /* 提供 ExecClearTuple 等清理函数 */

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
    IndexScanDesc scan;
    ScanKeyData   orderbykey;
    Snapshot      snapshot;
    int           n = 0;

    if (out_nfound)
        *out_nfound = 0;

    if (topk <= 0)
        return 0;

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
    Oid orderby_proc = get_opcode(orderby_op); /* operator -> underlying function OID */
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
    scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

    elog(WARNING, "HnswTopKForColumn: op=%u proc=%u col=%d", orderby_op, orderby_proc, col);

    /*
     * 🚀 终极修复：PostgreSQL 的 index_beginscan 绝对不会自动分配 xs_orderby 数组。
     * 因为我们是 CustomScan 直调底层，必须手动 palloc，否则读写必报 SIGSEGV！
     */
    if (scan->numberOfOrderBys > 0)
    {
        scan->xs_orderbyvals = (Datum *) palloc0(sizeof(Datum) * scan->numberOfOrderBys);
        scan->xs_orderbynulls = (bool *) palloc0(sizeof(bool) * scan->numberOfOrderBys);
        /* 默认先把 null 标志位设为 true，防止底层 AM 偷懒没写时我们读到脏数据 */
        scan->xs_orderbynulls[0] = true;
    }

    /*
     * 设置 order-by key：
     * - sk_attno：1-based 的 index key attribute number
     * - sk_func：必须是函数 OID（orderby_proc）
     * - sk_argument：query 向量 datum
     */
    ScanKeyEntryInitialize(&orderbykey,
                           0,                 /* flags */
                           (AttrNumber) (col + 1), /* sk_attno: 1-based key attno */
                           InvalidStrategy,   /* 对 orderby key 通常不需要 strategy */
                           InvalidOid,        /* subtype */
                           InvalidOid,        /* collation: 向量一般无 collation */
                           orderby_proc,        /* **重要**：距离算子 Oid */
                           query);            /* sk_argument: query vector Datum */

    /* 触发 AM rescan（HNSW AM 通常在这里或首次 getnext_tid 初始化候选集） */
    index_rescan(scan, NULL, 0, &orderbykey, 1);

#ifdef HNSW_HAVE_SCAN_SET_COLUMN
    /* 如果你的 multi 实现必须依赖 opaque->col，就强制设一下 */
    HnswScanSetColumn(scan, col);
#endif


    /* * =====================================================================
     * 🚀 终极绝招：既然索引不返回距离，我们就手动 Fetch 数据并当场计算！
     * =====================================================================
     */
    IndexFetchTableData *fetch = table_index_fetch_begin(heapRel);
    TupleTableSlot *slot = table_slot_create(heapRel, NULL);
    elog(WARNING,"SLOT创建完成");

    while (n < topk)
    {
        elog(WARNING, "DEBUG D: Loop n=%d - 准备 index_getnext_tid", n);
        bool found = index_getnext_tid(scan, ForwardScanDirection);
        if (!found) break;

        if (!ItemPointerIsValid(&scan->xs_heaptid)) break;

        out[n].tid = scan->xs_heaptid;

        elog(WARNING, "DEBUG E: Loop n=%d - 准备 fetch_tuple, TID=(%u, %u)",
             n, ItemPointerGetBlockNumber(&out[n].tid), ItemPointerGetOffsetNumber(&out[n].tid));

        /* 通过 TID 从原表中快速拉取这行数据 */
        bool call_again = false;
        bool all_dead = false;
        if (table_index_fetch_tuple(fetch, &out[n].tid, snapshot, slot, &call_again, &all_dead))
        {
            elog(WARNING, "DEBUG F: Loop n=%d - fetch_tuple 成功返回数据", n);
            bool isnull;

            /* * 🚀 绝杀修复：获取索引列在原表中的真实物理列号！
             * indexRel->rd_index->indkey 数组保存了索引列到原表列的映射关系。
             * 比如 col=0 时，它会准确地告诉你这是原表的第 3 列还是第 4 列。
             */
            AttrNumber heap_attnum = indexRel->rd_index->indkey.values[col];

            elog(WARNING, "DEBUG G: Loop n=%d - 准备 slot_getattr (真实表列号 heap_attnum=%d)", n, heap_attnum);

            /* 使用真实的列号提取数据 */
            Datum val = slot_getattr(slot, heap_attnum, &isnull);

            if (!isnull)
            {
                /* * 最核心的一步：调用底层的向量距离函数！
                 * 相当于在 C 语言里执行了一次 `val <#> query`
                 */
                elog(WARNING, "DEBUG H: Loop n=%d - 准备调用 OidFunctionCall2Coll (底层距离计算)", n);
                out[n].distance = DatumGetFloat8(OidFunctionCall2Coll(orderby_proc, InvalidOid, val, query));
                elog(WARNING, "DEBUG I: Loop n=%d - 计算成功距离: %f", n, out[n].distance);
            }
            else
            {
                elog(WARNING, "DEBUG J: Loop n=%d - 这一列是 NULL", n);
                out[n].distance = 0.0; /* 如果这行数据该列是 NULL，给个兜底值 */
            }
            elog(WARNING, "DEBUG K: Loop n=%d - 准备 ExecClearTuple", n);
            ExecClearTuple(slot); /* 清理槽位，给下一次循环复用 */
        }
        else
        {
            elog(WARNING, "DEBUG L: Loop n=%d - fetch_tuple 未找到可见数据", n);
            out[n].distance = 0.0;
        }

        n++;
    }

    /* 释放手动计算用的内存 */
    elog(WARNING, "DEBUG M: 准备清理内存");
    ExecDropSingleTupleTableSlot(slot);
    table_index_fetch_end(fetch);

    index_endscan(scan);

    if (out_nfound) *out_nfound = n;
    return n;
}