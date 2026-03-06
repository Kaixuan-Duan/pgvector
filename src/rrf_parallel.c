//
// Created by lzc on 2026/3/6.
//

#include "postgres.h"
#include "fmgr.h"
#include "access/parallel.h"
#include "storage/shm_toc.h"
#include "utils/snapmgr.h"
#include "access/table.h"
#include "utils/rel.h"
#include "hnswtopk.h"
#include "rrf_parallel.h"
#include "access/genam.h"


/*
 * Worker 进程的真正入口
 */
void
RrfParallelWorkerMain(dsm_segment *seg, shm_toc *toc)
{
    RrfSharedState *shared_state;
    Pointer         q2_ptr;
    HnswTopKItem   *list2_shared;
    Datum           q2_datum;
    Relation        heapRel;
    Relation        indexRel;

    /* 1. 从共享内存中取出参数 */
    shared_state = shm_toc_lookup(toc, PARALLEL_KEY_RRF_SHARED, false);
    q2_ptr       = shm_toc_lookup(toc, PARALLEL_KEY_RRF_Q2, false);
    list2_shared = shm_toc_lookup(toc, PARALLEL_KEY_RRF_LIST2, false);
    elog(WARNING, "worker取参数成功");

    /* 构造 Datum：直接指向 DSM 中的变长数组 (varlena) */
    q2_datum = PointerGetDatum(q2_ptr);

    /* 2. 打开表和索引（获取与主进程相同的锁） */
    heapRel  = table_open(shared_state->heap_oid, AccessShareLock);
    indexRel = index_open(shared_state->index_oid, AccessShareLock);
    elog(WARNING, "worker打开表和索引成功");

    /* * 3. 执行核心搜索逻辑
     * 注意：Parallel Worker 会自动继承并设置主进程的 ActiveSnapshot，
     * 所以 HnswTopKForColumn 内部调用的 GetActiveSnapshot() 会正常工作。
     */
    elog(WARNING, "work执行Topk逻辑");
    HnswTopKForColumn(heapRel,
                      indexRel,
                      shared_state->col2,
                      shared_state->op2,
                      q2_datum,
                      shared_state->cand2,
                      list2_shared,        /* 直接将结果写回 DSM */
                      &shared_state->n2_result);

    /* 4. 清理资源并退出 */
    index_close(indexRel, AccessShareLock);
    table_close(heapRel, AccessShareLock);
}