#include "postgres.h"
#include "fmgr.h"
#include "access/parallel.h"
#include "access/genam.h"
#include "storage/shm_toc.h"
#include "access/table.h"
#include "utils/rel.h"
#include "hnswtopk.h"
#include "linear_parallel.h"

void
LinearParallelWorkerMain(dsm_segment *seg, shm_toc *toc)
{
    (void) seg;

    LinearSharedState *shared_state;
    Pointer         q2_ptr;
    HnswTopKItem   *list2_shared;
    Datum           q2_datum;
    Relation        heapRel;
    Relation        indexRel;

    shared_state = shm_toc_lookup(toc, PARALLEL_KEY_LINEAR_SHARED, false);
    q2_ptr       = shm_toc_lookup(toc, PARALLEL_KEY_LINEAR_Q2, false);
    list2_shared = shm_toc_lookup(toc, PARALLEL_KEY_LINEAR_LIST2, false);

    q2_datum = PointerGetDatum(q2_ptr);

    heapRel  = table_open(shared_state->heap_oid, AccessShareLock);
    indexRel = index_open(shared_state->index_oid, AccessShareLock);

    HnswTopKForColumn(heapRel,
                      indexRel,
                      shared_state->col2,
                      shared_state->op2,
                      q2_datum,
                      shared_state->cand2,
                      list2_shared,
                      &shared_state->n2_result);

    index_close(indexRel, AccessShareLock);
    table_close(heapRel, AccessShareLock);
}
