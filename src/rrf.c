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
Datum
rrf(PG_FUNCTION_ARGS)
{
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("rrf() must be used with ORDER BY to enable vector RRF CustomScan"),
             errhint("Use: SELECT ..., rrf(...) AS s1 FROM ... ORDER BY s1 DESC LIMIT ...")));
    PG_RETURN_FLOAT8(0); /* keep compiler quiet */
}
