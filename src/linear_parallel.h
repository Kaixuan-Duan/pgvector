#ifndef PGVECTOR_LINEAR_PARALLEL_H
#define PGVECTOR_LINEAR_PARALLEL_H

#include "postgres.h"
#include "storage/dsm.h"
#include "storage/shm_toc.h"
#include "utils/rel.h"
#include "access/table.h"

#define PARALLEL_KEY_LINEAR_SHARED 1
#define PARALLEL_KEY_LINEAR_Q2     2
#define PARALLEL_KEY_LINEAR_LIST2  3

typedef struct LinearSharedState
{
    Oid     heap_oid;
    Oid     index_oid;
    int     col2;
    Oid     op2;
    int     cand2;
    int     n2_result;
} LinearSharedState;

PGDLLEXPORT void LinearParallelWorkerMain(dsm_segment *seg, shm_toc *toc);

#endif
