//
// Created by lzc on 2026/3/6.
//

#ifndef PGVECTOR_RRF_PARALLEL_H
#define PGVECTOR_RRF_PARALLEL_H

#include "postgres.h"
#include "storage/dsm.h"
#include "storage/shm_toc.h"
#include "utils/rel.h"      /* index_open 的声明在这里 */
#include "access/table.h"   /* table_open 的声明在这里 */

/* 共享内存结构的 Key */
#define PARALLEL_KEY_RRF_SHARED 1
#define PARALLEL_KEY_RRF_Q2     2
#define PARALLEL_KEY_RRF_LIST2  3

/* 共享状态结构体：保证 Leader 和 Worker 看到的内存布局绝对一致 */
typedef struct RrfSharedState
{
    Oid     heap_oid;
    Oid     index_oid;
    int     col2;
    Oid     op2;
    int     cand2;
    int     n2_result;
} RrfSharedState;

/* 导出函数声明 */
PGDLLEXPORT void RrfParallelWorkerMain(dsm_segment *seg, shm_toc *toc);


#endif //PGVECTOR_RRF_PARALLEL_H