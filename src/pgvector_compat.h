#ifndef PGVECTOR_COMPAT_H
#define PGVECTOR_COMPAT_H

#include "postgres.h"

/* pgvector 0.8.x varlena layouts used by the hydex on-disk format. */
#define VECTOR_MAX_DIM 16000
#define SPARSEVEC_MAX_DIM 1000000000

typedef struct Vector
{
	int32		vl_len_;
	int16		dim;
	int16		unused;
	float		x[FLEXIBLE_ARRAY_MEMBER];
}			Vector;

typedef struct SparseVector
{
	int32		vl_len_;
	int32		dim;
	int32		nnz;
	int32		unused;
	int32		indices[FLEXIBLE_ARRAY_MEMBER];
}			SparseVector;

StaticAssertDecl(offsetof(Vector, x) == 8,
				 "unexpected pgvector 0.8.x Vector layout");
StaticAssertDecl(offsetof(SparseVector, indices) == 16,
				 "unexpected pgvector 0.8.x SparseVector layout");

#if PG_VERSION_NUM >= 160000
#define FUNCTION_PREFIX
#else
#define FUNCTION_PREFIX PGDLLEXPORT
#endif

#endif
