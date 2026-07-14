#include "postgres.h"

#include "fmgr.h"
#include "utils/errcodes.h"

#include "hnsw.h"

static void
HydexSparsevecCheckValue(Pointer value)
{
	SparseVector *vec = (SparseVector *) value;

	if (vec->nnz > HNSW_MAX_NNZ)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than %d non-zero elements for hydex index",
						HNSW_MAX_NNZ)));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hydex_halfvec_support);
Datum
hydex_halfvec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hydex_bit_support);
Datum
hydex_bit_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 32,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hydex_sparsevec_support);
Datum
hydex_sparsevec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SPARSEVEC_MAX_DIM,
		.checkValue = HydexSparsevecCheckValue
	};

	PG_RETURN_POINTER(&typeInfo);
}
