-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION hybrid_vector" to load this file. \quit

DO $$
DECLARE
	vector_version text;
BEGIN
	SELECT extversion INTO vector_version
	FROM pg_catalog.pg_extension
	WHERE extname = 'vector';

	IF vector_version IS DISTINCT FROM '0.8.0' THEN
		RAISE EXCEPTION 'hybrid_vector requires vector extension version 0.8.0, found %',
			COALESCE(vector_version, 'not installed');
	END IF;
END
$$;

-- hydex access method

CREATE FUNCTION hydexhandler(internal) RETURNS index_am_handler
	AS 'MODULE_PATHNAME', 'hnswhandler' LANGUAGE C;

CREATE ACCESS METHOD hydex TYPE INDEX HANDLER hydexhandler;

COMMENT ON ACCESS METHOD hydex IS 'hydex index access method';

-- hydex support functions

CREATE FUNCTION hydex_halfvec_support(internal) RETURNS internal
	AS 'MODULE_PATHNAME', 'hnsw_halfvec_support' LANGUAGE C;

CREATE FUNCTION hydex_bit_support(internal) RETURNS internal
	AS 'MODULE_PATHNAME', 'hnsw_bit_support' LANGUAGE C;

CREATE FUNCTION hydex_sparsevec_support(internal) RETURNS internal
	AS 'MODULE_PATHNAME', 'hnsw_sparsevec_support' LANGUAGE C;

-- hydex opclasses for pgvector types

CREATE OPERATOR CLASS vector_l2_ops
	FOR TYPE @extschema:vector@.vector USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<->) (@extschema:vector@.vector, @extschema:vector@.vector) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.vector_l2_squared_distance(@extschema:vector@.vector, @extschema:vector@.vector);

CREATE OPERATOR CLASS vector_ip_ops
	FOR TYPE @extschema:vector@.vector USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<#>) (@extschema:vector@.vector, @extschema:vector@.vector) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.vector_negative_inner_product(@extschema:vector@.vector, @extschema:vector@.vector);

CREATE OPERATOR CLASS vector_cosine_ops
	FOR TYPE @extschema:vector@.vector USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<=>) (@extschema:vector@.vector, @extschema:vector@.vector) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.vector_negative_inner_product(@extschema:vector@.vector, @extschema:vector@.vector),
	FUNCTION 2 @extschema:vector@.vector_norm(@extschema:vector@.vector);

CREATE OPERATOR CLASS vector_l1_ops
	FOR TYPE @extschema:vector@.vector USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<+>) (@extschema:vector@.vector, @extschema:vector@.vector) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.l1_distance(@extschema:vector@.vector, @extschema:vector@.vector);

CREATE OPERATOR CLASS halfvec_l2_ops
	FOR TYPE @extschema:vector@.halfvec USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<->) (@extschema:vector@.halfvec, @extschema:vector@.halfvec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.halfvec_l2_squared_distance(@extschema:vector@.halfvec, @extschema:vector@.halfvec),
	FUNCTION 3 hydex_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_ip_ops
	FOR TYPE @extschema:vector@.halfvec USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<#>) (@extschema:vector@.halfvec, @extschema:vector@.halfvec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.halfvec_negative_inner_product(@extschema:vector@.halfvec, @extschema:vector@.halfvec),
	FUNCTION 3 hydex_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_cosine_ops
	FOR TYPE @extschema:vector@.halfvec USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<=>) (@extschema:vector@.halfvec, @extschema:vector@.halfvec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.halfvec_negative_inner_product(@extschema:vector@.halfvec, @extschema:vector@.halfvec),
	FUNCTION 2 @extschema:vector@.l2_norm(@extschema:vector@.halfvec),
	FUNCTION 3 hydex_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_l1_ops
	FOR TYPE @extschema:vector@.halfvec USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<+>) (@extschema:vector@.halfvec, @extschema:vector@.halfvec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.l1_distance(@extschema:vector@.halfvec, @extschema:vector@.halfvec),
	FUNCTION 3 hydex_halfvec_support(internal);

CREATE OPERATOR CLASS bit_hamming_ops
	FOR TYPE pg_catalog.bit USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<~>) (pg_catalog.bit, pg_catalog.bit) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.hamming_distance(pg_catalog.bit, pg_catalog.bit),
	FUNCTION 3 hydex_bit_support(internal);

CREATE OPERATOR CLASS bit_jaccard_ops
	FOR TYPE pg_catalog.bit USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<%>) (pg_catalog.bit, pg_catalog.bit) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.jaccard_distance(pg_catalog.bit, pg_catalog.bit),
	FUNCTION 3 hydex_bit_support(internal);

CREATE OPERATOR CLASS sparsevec_l2_ops
	FOR TYPE @extschema:vector@.sparsevec USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<->) (@extschema:vector@.sparsevec, @extschema:vector@.sparsevec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.sparsevec_l2_squared_distance(@extschema:vector@.sparsevec, @extschema:vector@.sparsevec),
	FUNCTION 3 hydex_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_ip_ops
	FOR TYPE @extschema:vector@.sparsevec USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<#>) (@extschema:vector@.sparsevec, @extschema:vector@.sparsevec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.sparsevec_negative_inner_product(@extschema:vector@.sparsevec, @extschema:vector@.sparsevec),
	FUNCTION 3 hydex_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_cosine_ops
	FOR TYPE @extschema:vector@.sparsevec USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<=>) (@extschema:vector@.sparsevec, @extschema:vector@.sparsevec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.sparsevec_negative_inner_product(@extschema:vector@.sparsevec, @extschema:vector@.sparsevec),
	FUNCTION 2 @extschema:vector@.l2_norm(@extschema:vector@.sparsevec),
	FUNCTION 3 hydex_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_l1_ops
	FOR TYPE @extschema:vector@.sparsevec USING hydex AS
	OPERATOR 1 OPERATOR(@extschema:vector@.<+>) (@extschema:vector@.sparsevec, @extschema:vector@.sparsevec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 @extschema:vector@.l1_distance(@extschema:vector@.sparsevec, @extschema:vector@.sparsevec),
	FUNCTION 3 hydex_sparsevec_support(internal);

-- RRF fusion score function (planner hook / custom scan marker)

CREATE FUNCTION rrf(
	emb1   @extschema:vector@.sparsevec,
	op1    pg_catalog.regoperator,
	q1     @extschema:vector@.sparsevec,
	emb2   @extschema:vector@.vector,
	op2    pg_catalog.regoperator,
	q2     @extschema:vector@.vector,
	k      integer            DEFAULT 60,
	w1     double precision   DEFAULT 0.5,
	w2     double precision   DEFAULT 0.5,
	cand1  integer            DEFAULT 200,
	cand2  integer            DEFAULT 200
)
	RETURNS double precision
AS 'MODULE_PATHNAME', 'rrf'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION rrf(
	emb1   @extschema:vector@.vector,
	op1    pg_catalog.regoperator,
	q1     @extschema:vector@.vector,
	emb2   @extschema:vector@.sparsevec,
	op2    pg_catalog.regoperator,
	q2     @extschema:vector@.sparsevec,
	k      integer            DEFAULT 60,
	w1     double precision   DEFAULT 0.5,
	w2     double precision   DEFAULT 0.5,
	cand1  integer            DEFAULT 200,
	cand2  integer            DEFAULT 200
)
	RETURNS double precision
AS 'MODULE_PATHNAME', 'rrf'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION rrf(
	emb1   @extschema:vector@.sparsevec,
	op1    pg_catalog.regoperator,
	q1     @extschema:vector@.sparsevec,
	emb2   @extschema:vector@.sparsevec,
	op2    pg_catalog.regoperator,
	q2     @extschema:vector@.sparsevec,
	k      integer            DEFAULT 60,
	w1     double precision   DEFAULT 0.5,
	w2     double precision   DEFAULT 0.5,
	cand1  integer            DEFAULT 200,
	cand2  integer            DEFAULT 200
)
	RETURNS double precision
AS 'MODULE_PATHNAME', 'rrf'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION rrf(
	emb1   @extschema:vector@.vector,
	op1    pg_catalog.regoperator,
	q1     @extschema:vector@.vector,
	emb2   @extschema:vector@.vector,
	op2    pg_catalog.regoperator,
	q2     @extschema:vector@.vector,
	k      integer            DEFAULT 60,
	w1     double precision   DEFAULT 0.5,
	w2     double precision   DEFAULT 0.5,
	cand1  integer            DEFAULT 200,
	cand2  integer            DEFAULT 200
)
	RETURNS double precision
AS 'MODULE_PATHNAME', 'rrf'
LANGUAGE C STRICT STABLE;
