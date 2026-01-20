//
// Created by kaixu on 2026/1/20.
//


#pragma once

#include "postgres.h"
#include "access/htup_details.h"
#include "utils/rel.h"

typedef struct HnswTopKItem
{
    ItemPointerData tid;
    float8 distance;   /* d1/d2 */
} HnswTopKItem;

/*
 * col: 0-based，第几列/第几张图
 * orderby_op: 该列对应的距离算子 Oid（例如 <#> / <-> / <=>），强烈建议由 planner 从 OpExpr 提取后传入
 * query: 该列的 query 向量 Datum
 */
int HnswTopKForColumn(Relation heapRel,
                      Relation indexRel,
                      int col,
                      Oid orderby_op,
                      Datum query,
                      int topk,
                      HnswTopKItem *out,    /* caller alloc: topk 个 */
                      int *out_nfound);
