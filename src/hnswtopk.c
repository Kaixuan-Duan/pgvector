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


    Oid orderby_proc = get_opcode(orderby_op); /* operator -> underlying function OID */
    if (!OidIsValid(orderby_proc))
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("orderby operator OID %u has no underlying procedure", orderby_op)));

    snapshot = GetActiveSnapshot();
    if (snapshot == NULL)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("no active snapshot")));


    scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

    elog(WARNING, "HnswTopKForColumn: op=%u proc=%u col=%d", orderby_op, orderby_proc, col);


    ScanKeyEntryInitialize(&orderbykey,
                           0,
                           (AttrNumber) (col + 1),
                           InvalidStrategy,
                           InvalidOid,
                           InvalidOid,
                           orderby_proc,
                           query);

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

        if (!ItemPointerIsValid(&scan->xs_heaptid))
            ereport(ERROR,
                    (errmsg("HnswTopKForColumn: index_getnext_tid returned invalid TID (col=%d, op=%u)",
                            col, orderby_op)));

        out[n].tid = scan->xs_heaptid;


        out[n].distance = 0.0;

        n++;
    }

    index_endscan(scan);

    if (out_nfound)
        *out_nfound = n;

    return n;
}