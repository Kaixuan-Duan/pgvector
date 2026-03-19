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
    float8 distance;
} HnswTopKItem;


int HnswTopKForColumn(Relation heapRel,
                      Relation indexRel,
                      int col,
                      Oid orderby_op,
                      Datum query,
                      int topk,
                      HnswTopKItem *out,
                      int *out_nfound);
