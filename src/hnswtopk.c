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
#include "hnsw.h"
#include "hnswtopk.h"

#include "utils/lsyscache.h"
#include "utils/errcodes.h"
#include "access/tableam.h"
#include "executor/tuptable.h"
#include "executor/executor.h"

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

    Oid orderby_proc = get_opcode(orderby_op);
    Oid distance_proc;
    Oid norm_proc;
    Oid commutator_op;
    bool can_reuse_distance;

    if (!OidIsValid(orderby_proc))
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("orderby operator OID %u has no underlying procedure", orderby_op)));

    distance_proc = index_getprocid(indexRel,
                                    (AttrNumber) (col + 1),
                                    HNSW_DISTANCE_PROC);
    norm_proc = index_getprocid(indexRel,
                                (AttrNumber) (col + 1),
                                HNSW_NORM_PROC);
    commutator_op = get_commutator(orderby_op);
    /*
     * The graph-search distance has the SQL operator's numeric semantics only
     * when both paths call the same commutative function on unnormalized data.
     */
    can_reuse_distance = OidIsValid(distance_proc) &&
                         distance_proc == orderby_proc &&
                         !OidIsValid(norm_proc) &&
                         commutator_op == orderby_op;

    snapshot = GetActiveSnapshot();
    if (snapshot == NULL)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("no active snapshot")));

    scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

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
    HnswScanSetColumn(scan, col);
#endif
    IndexFetchTableData *fetch = table_index_fetch_begin(heapRel);
    TupleTableSlot *slot = table_slot_create(heapRel, NULL);

    while (n < topk)
    {
        bool found = index_getnext_tid(scan, ForwardScanDirection);
        if (!found)
            break;

        if (!ItemPointerIsValid(&scan->xs_heaptid))
            break;

        out[n].tid = scan->xs_heaptid;

        bool call_again = false;
        bool all_dead = false;
        if (table_index_fetch_tuple(fetch, &out[n].tid, snapshot, slot, &call_again, &all_dead))
        {
            double index_distance;

            if (can_reuse_distance &&
                HnswGetLastDistance(scan, col, &index_distance))
            {
                out[n].distance = index_distance;
            }
            else
            {
                bool isnull;
                AttrNumber heap_attnum = indexRel->rd_index->indkey.values[col];
                Datum val = slot_getattr(slot, heap_attnum, &isnull);

                if (!isnull)
                {
                    out[n].distance = DatumGetFloat8(OidFunctionCall2Coll(orderby_proc, InvalidOid, val, query));
                }
                else
                    out[n].distance = 0.0;
            }

            ExecClearTuple(slot);
        }
        else
            out[n].distance = 0.0;

        n++;
    }

    ExecDropSingleTupleTableSlot(slot);
    table_index_fetch_end(fetch);

    index_endscan(scan);

    if (out_nfound)
        *out_nfound = n;
    return n;
}
