//
// Created by kaixu on 2026/1/20.
//
/* src/rrf.c */
#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"

PG_FUNCTION_INFO_V1(rrf);

/*
 * rrf(...) stub
 * Real score is produced by VectorRRF CustomScan executor.
 */
/* 引入外部变量声明 */
extern double *current_rrf_score;

Datum
rrf(PG_FUNCTION_ARGS)
{
    /* * 1. 检查是否处于 VectorRRF CustomScan 上下文中
     * 如果全局指针不为空，说明是 CustomScan 正在调用我们。
     * 我们直接返回还没凉热乎的分数，不做任何计算。
     */
    if (current_rrf_score != NULL)
    {
        PG_RETURN_FLOAT8(*current_rrf_score);
    }

    /* * 2. 如果指针为空，说明用户在普通 SELECT 中调用了 rrf()，
     * 或者没有走 CustomScan 索引扫描。
     * 此时抛出原来的错误，强制用户加 ORDER BY。
     */
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("rrf() must be used with ORDER BY to enable VectorRRF CustomScan"),
             errhint("Use: SELECT ..., rrf(emb1, '<#>'::regoperator, q1, emb2, '<#>'::regoperator, q2, ...) AS s1 "
                     "FROM ... ORDER BY s1 DESC LIMIT ...")));
    // PG_RETURN_FLOAT8(0);
    /* 永远不会走到这里 */
    PG_RETURN_NULL();
}
