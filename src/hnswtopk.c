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
                  Oid orderby_op,
                  Datum query,
                  int topk,
                  HnswTopKItem *out,
                  int *out_nfound)
{
    IndexScanDesc scan;
    ScanKeyData orderbykey;
    Snapshot snapshot;
    int n = 0;

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
                 errhint("Pass the operator OID from the distance OpExpr (e.g. <->, <#>, <=>) via planner custom_private.")));

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

    /*
     * 关键：告诉 AM 这是对第 col 列的 order-by
     * 你历史里多列分流/选列逻辑经常用：
     *   scan->orderByData[0].sk_attno  (1-based) 来推 col
     */
    ScanKeyEntryInitialize(&orderbykey,
                           0,                 /* flags */
                           (AttrNumber) (col + 1), /* sk_attno: 1-based key attno */
                           InvalidStrategy,   /* 对 orderby key 通常不需要 strategy */
                           InvalidOid,        /* subtype */
                           InvalidOid,        /* collation: 向量一般无 collation */
                           orderby_op,        /* **重要**：距离算子 Oid */
                           query);            /* sk_argument: query vector Datum */

    /* 把 orderby operator/collation 也填进 scan 描述符（很多 AM 会看这里） */
    // scan->orderByOperators[0] = orderby_op;
    // scan->orderByCollations[0] = InvalidOid;
    // scan->orderByNullsFirst[0] = false;

    /* 触发 AM rescan（你的 HNSW 会在这里或首次 getnext_tid 初始化候选集） */
    index_rescan(scan, NULL, 0, &orderbykey, 1);

#ifdef HNSW_HAVE_SCAN_SET_COLUMN
    /* 如果你的 multi 实现必须依赖 opaque->col，就强制设一下 */
    HnswScanSetColumn(scan, col);
#endif

    while (n < topk)
    {
        bool found = index_getnext_tid(scan, ForwardScanDirection);
        if (!found)
            break;

        out[n].tid = scan->xs_heaptid;

        /* distance 是可选调试字段：一定要防御，避免 xs_orderbyvals 为空导致崩溃 */
        out[n].distance = 0.0;

        /*
         * 你现在普通 IndexScan+ORDER BY 能工作，说明 AM 很可能设置了 xs_orderbyvals[0]
         * 如果这里读出来不对/为空，你可以在 AM 里确保对每个返回的 tid 都写 xs_orderbyvals。
         */
        /*
        if (scan->xs_orderbynulls && scan->xs_orderbynulls[0])
        {
            out[n].distance = HUGE_VAL;
        }
        else
        {
            out[n].distance = DatumGetFloat8(scan->xs_orderbyvals[0]);
        }
        */
        if (scan->xs_orderbyvals != NULL &&
            scan->xs_orderbynulls != NULL &&
            !scan->xs_orderbynulls[0])
        {
            Oid rettype = get_op_rettype(orderby_op);

            if (rettype == FLOAT8OID)
                out[n].distance = DatumGetFloat8(scan->xs_orderbyvals[0]);
            else if (rettype == FLOAT4OID)
                out[n].distance = (float8) DatumGetFloat4(scan->xs_orderbyvals[0]);
            else
                out[n].distance = 0.0; /* 不认识的返回类型就别读了 */
        }
        else
        {
            /* 没有返回 distance，也不要崩；你也可以用 HUGE_VAL 表示 unknown */
            out[n].distance = HUGE_VAL;
        }

        n++;
    }

    index_endscan(scan);

    if (out_nfound)
        *out_nfound = n;

    return n;
}
