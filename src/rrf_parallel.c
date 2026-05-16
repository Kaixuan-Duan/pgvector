//
// Created by lzc on 2026/3/6.
//

#include "postgres.h"
#include "fmgr.h"
#include "access/parallel.h"
#include "storage/shm_toc.h"
#include "utils/wait_event.h"
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
    HnswTopKItem   *batch2_shared;
    HnswTopKStream *stream = NULL;
    Datum           q2_datum;
    Relation        heapRel;
    Relation        indexRel;
    uint32          processed_request = 0;
    int             depth = 0;

    /* 1. 从共享内存中取出参数 */
    shared_state = shm_toc_lookup(toc, PARALLEL_KEY_RRF_SHARED, false);
    q2_ptr       = shm_toc_lookup(toc, PARALLEL_KEY_RRF_Q2, false);
    batch2_shared = shm_toc_lookup(toc, PARALLEL_KEY_RRF_BATCH2, false);
    //elog(WARNING, "worker取参数成功");

    /* 构造 Datum：直接指向 DSM 中的变长数组 (varlena) */
    q2_datum = PointerGetDatum(q2_ptr);

    /* 2. 打开表和索引（获取与主进程相同的锁） */
    heapRel  = table_open(shared_state->heap_oid, AccessShareLock);
    indexRel = index_open(shared_state->index_oid, AccessShareLock);
    //elog(WARNING, "worker打开表和索引成功");

    /*
     * 3. 持有 col2 的流式 scan，每次等 leader 请求后写一批到 DSM。
     * Parallel Worker 会继承主进程 ActiveSnapshot，所以 stream 内部
     * GetActiveSnapshot() 与现有一次性 topK 路径一致。
     */
    stream = HnswTopKStreamBegin(heapRel,
                                 indexRel,
                                 shared_state->col2,
                                 shared_state->op2,
                                 q2_datum);

    for (;;)
    {
        uint32 request_no;
        bool stop;
        bool stream_done = false;
        int remaining;
        int max_batch;
        int nitems = 0;

        for (;;)
        {
            SpinLockAcquire(&shared_state->mutex);
            request_no = shared_state->request_no;
            stop = shared_state->stop;
            if (stop || request_no > processed_request)
            {
                SpinLockRelease(&shared_state->mutex);
                break;
            }
            SpinLockRelease(&shared_state->mutex);

            ConditionVariableSleep(&shared_state->cv,
                                   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
        }
        ConditionVariableCancelSleep();

        if (stop)
            break;

        remaining = shared_state->cand2 - depth;
        max_batch = Min(shared_state->batch_size, remaining);

        if (max_batch > 0)
            HnswTopKStreamNextBatch(stream,
                                    max_batch,
                                    batch2_shared,
                                    &nitems,
                                    &stream_done);
        else
            stream_done = true;

        depth += nitems;

        SpinLockAcquire(&shared_state->mutex);
        shared_state->nitems = nitems;
        shared_state->depth2 = depth;
        shared_state->done = stream_done || depth >= shared_state->cand2;
        shared_state->ready_no = request_no;
        SpinLockRelease(&shared_state->mutex);

        ConditionVariableBroadcast(&shared_state->cv);

        processed_request = request_no;
        if (stream_done || depth >= shared_state->cand2)
            break;
    }

    /* 4. 清理资源并退出 */
    HnswTopKStreamEnd(stream);
    index_close(indexRel, AccessShareLock);
    table_close(heapRel, AccessShareLock);
}
