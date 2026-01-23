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

#include "hnsw.h"


void
HnswScanSetColumn(IndexScanDesc scan, int col)
{
    HnswScanOpaqueMulti soMulti = (HnswScanOpaqueMulti) scan->opaque;

    if (soMulti == NULL)
        elog(ERROR, "HnswScanSetColumn: scan->opaque is NULL");

    if (col < 0 || col >= soMulti->nkeys)
        elog(ERROR, "HnswScanSetColumn: col out of range: %d (nkeys=%d)",
             col, soMulti->nkeys);

    soMulti->col = col;
}

void
HnswScanSetOrderByOp(IndexScanDesc scan, Oid orderby_op)
{
    HnswScanOpaqueMulti soMulti = (HnswScanOpaqueMulti) scan->opaque;
    int col;

    if (soMulti == NULL)
        elog(ERROR, "HnswScanSetOrderByOp: scan->opaque is NULL");

    col = soMulti->col;

    if (col < 0 || col >= soMulti->nkeys)
        elog(ERROR, "HnswScanSetOrderByOp: col out of range: %d (nkeys=%d)",
             col, soMulti->nkeys);

    soMulti->cols[col].orderby_op = orderby_op;
    soMulti->cols[col].orderby_proc = get_opcode(orderby_op);

    /* 可选：这里不 ERROR，只做日志；真正用时再 ERROR 也行 */
    if (!OidIsValid(soMulti->cols[col].orderby_proc))
        elog(WARNING, "HnswScanSetOrderByOp: op=%u has invalid proc (col=%d)",
             orderby_op, col);
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
                 errmsg("orderby operator OID is invalid")));

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
     * ORDER BY key：
     * - flags: SK_ORDER_BY
     * - sk_attno: 1-based key attno
     * - sk_func: underlying proc oid
     * - sk_argument: query datum
     */
    ScanKeyEntryInitialize(&orderbykey,
                           SK_ORDER_BY,
                           (AttrNumber) (col + 1),
                           InvalidStrategy,
                           InvalidOid,
                           InvalidOid,
                           orderby_proc,
                           query);

    /* 触发 AM rescan（HNSW AM 通常在这里或首次 getnext_tid 初始化候选集） */
    index_rescan(scan, NULL, 0, &orderbykey, 1);

    /* rescan 之后再 set，避免被 rescan 清掉 */
    HnswScanSetColumn(scan, col);              /* 可选但建议 */
    HnswScanSetOrderByOp(scan, orderby_op);    /* 关键：必须在 rescan 后 */

    while (n < topk)
    {
        bool found = index_getnext_tid(scan, ForwardScanDirection);
        if (!found)
            break;

        if (!ItemPointerIsValid(&scan->xs_heaptid))
            ereport(ERROR,
                    (errmsg("HnswTopKForColumn: invalid xs_heaptid (col=%d op=%u)",
                            col, orderby_op)));

        out[n].tid = scan->xs_heaptid;

        /*
         * 不读取 scan->xs_orderbyvals / xs_orderbynulls：
         * - 你当前 RRF 评分不需要 distance
         * - 避免 AM 未正确维护 xs_orderby* 导致 SIGSEGV
         */
        out[n].distance = 0.0;

        n++;
    }

    index_endscan(scan);

    if (out_nfound)
        *out_nfound = n;

    return n;
}
