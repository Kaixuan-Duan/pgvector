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

int
HnswTopKForColumn(Relation heapRel,
                  Relation indexRel,
                  int col,
                  Oid orderby_op,     /* operator OID, e.g. <-> / <#> / <=> */
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
                 errhint("Pass the operator OID from the distance OpExpr (e.g. <->, <#>, <=>).")));

    /*
     * 关键修复：
     * ScanKeyEntryInitialize 的 “procedure” 参数必须是 *function OID* (pg_proc)。
     * 你之前传的是 operator OID，内核会把它当函数去查 pg_proc，导致
     * cache lookup failed for function <operator_oid>
     */
    Oid orderby_proc = get_opcode(orderby_op);  /* operator -> underlying function OID */
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
                           orderby_proc,        /* **重要**：距离算子 Oid */
                           query);            /* sk_argument: query vector Datum */

    /*
     * 同时把 operator OID 填进 scan 描述符（很多 AM / 计划展示会看这个）
     * 注意：这些数组在 index_beginscan 后才可用
     */
    scan->orderByOperators[0]  = orderby_op;
    scan->orderByCollations[0] = InvalidOid;
    scan->orderByNullsFirst[0] = false;

    /* 触发 AM rescan（HNSW AM 会在这里或首次 getnext_tid 初始化候选集） */
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

        /* distance 是可选字段：要防御 xs_orderbyvals 为空 */
        out[n].distance = HUGE_VAL;

        if (scan->xs_orderbyvals != NULL &&
            scan->xs_orderbynulls != NULL &&
            !scan->xs_orderbynulls[0])
        {
            /*
             * 更稳：用 underlying procedure 的返回类型判断
             * （也可以用 get_op_rettype(orderby_op)，但这里已经有 orderby_proc 了）
             */
            Oid rettype = get_func_rettype(orderby_proc);

            if (rettype == FLOAT8OID)
                out[n].distance = DatumGetFloat8(scan->xs_orderbyvals[0]);
            else if (rettype == FLOAT4OID)
                out[n].distance = (float8) DatumGetFloat4(scan->xs_orderbyvals[0]);
            else
                out[n].distance = 0.0; /* 不认识的返回类型就别强读 */
        }

        n++;
    }

    index_endscan(scan);

    if (out_nfound)
        *out_nfound = n;

    return n;
}