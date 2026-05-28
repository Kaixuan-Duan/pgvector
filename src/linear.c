#include "postgres.h"
#include "fmgr.h"

PG_FUNCTION_INFO_V1(linear);

extern double *current_linear_score;

Datum
linear(PG_FUNCTION_ARGS)
{
    (void) fcinfo;

    if (current_linear_score != NULL)
        PG_RETURN_FLOAT8(*current_linear_score);

    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("linear() must be used with ORDER BY to enable VectorLinear CustomScan"),
             errhint("Use: SELECT ..., linear(emb1, '<#>'::regoperator, q1, emb2, '<#>'::regoperator, q2, ...) AS s1 "
                     "FROM ... ORDER BY s1 DESC LIMIT ...")));
    PG_RETURN_NULL();
}
