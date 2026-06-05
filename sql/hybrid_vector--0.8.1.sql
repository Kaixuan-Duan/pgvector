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
	FOR TYPE public.vector USING hydex AS
	OPERATOR 1 <-> (public.vector, public.vector) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.vector_l2_squared_distance(public.vector, public.vector);

CREATE OPERATOR CLASS vector_ip_ops
	FOR TYPE public.vector USING hydex AS
	OPERATOR 1 <#> (public.vector, public.vector) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.vector_negative_inner_product(public.vector, public.vector);

CREATE OPERATOR CLASS vector_cosine_ops
	FOR TYPE public.vector USING hydex AS
	OPERATOR 1 <=> (public.vector, public.vector) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.vector_negative_inner_product(public.vector, public.vector),
	FUNCTION 2 public.vector_norm(public.vector);

CREATE OPERATOR CLASS vector_l1_ops
	FOR TYPE public.vector USING hydex AS
	OPERATOR 1 <+> (public.vector, public.vector) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.l1_distance(public.vector, public.vector);

CREATE OPERATOR CLASS halfvec_l2_ops
	FOR TYPE public.halfvec USING hydex AS
	OPERATOR 1 <-> (public.halfvec, public.halfvec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.halfvec_l2_squared_distance(public.halfvec, public.halfvec),
	FUNCTION 3 hydex_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_ip_ops
	FOR TYPE public.halfvec USING hydex AS
	OPERATOR 1 <#> (public.halfvec, public.halfvec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.halfvec_negative_inner_product(public.halfvec, public.halfvec),
	FUNCTION 3 hydex_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_cosine_ops
	FOR TYPE public.halfvec USING hydex AS
	OPERATOR 1 <=> (public.halfvec, public.halfvec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.halfvec_negative_inner_product(public.halfvec, public.halfvec),
	FUNCTION 2 public.l2_norm(public.halfvec),
	FUNCTION 3 hydex_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_l1_ops
	FOR TYPE public.halfvec USING hydex AS
	OPERATOR 1 <+> (public.halfvec, public.halfvec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.l1_distance(public.halfvec, public.halfvec),
	FUNCTION 3 hydex_halfvec_support(internal);

CREATE OPERATOR CLASS bit_hamming_ops
	FOR TYPE pg_catalog.bit USING hydex AS
	OPERATOR 1 <~> (pg_catalog.bit, pg_catalog.bit) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.hamming_distance(pg_catalog.bit, pg_catalog.bit),
	FUNCTION 3 hydex_bit_support(internal);

CREATE OPERATOR CLASS bit_jaccard_ops
	FOR TYPE pg_catalog.bit USING hydex AS
	OPERATOR 1 <%> (pg_catalog.bit, pg_catalog.bit) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.jaccard_distance(pg_catalog.bit, pg_catalog.bit),
	FUNCTION 3 hydex_bit_support(internal);

CREATE OPERATOR CLASS sparsevec_l2_ops
	FOR TYPE public.sparsevec USING hydex AS
	OPERATOR 1 <-> (public.sparsevec, public.sparsevec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.sparsevec_l2_squared_distance(public.sparsevec, public.sparsevec),
	FUNCTION 3 hydex_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_ip_ops
	FOR TYPE public.sparsevec USING hydex AS
	OPERATOR 1 <#> (public.sparsevec, public.sparsevec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.sparsevec_negative_inner_product(public.sparsevec, public.sparsevec),
	FUNCTION 3 hydex_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_cosine_ops
	FOR TYPE public.sparsevec USING hydex AS
	OPERATOR 1 <=> (public.sparsevec, public.sparsevec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.sparsevec_negative_inner_product(public.sparsevec, public.sparsevec),
	FUNCTION 2 public.l2_norm(public.sparsevec),
	FUNCTION 3 hydex_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_l1_ops
	FOR TYPE public.sparsevec USING hydex AS
	OPERATOR 1 <+> (public.sparsevec, public.sparsevec) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 public.l1_distance(public.sparsevec, public.sparsevec),
	FUNCTION 3 hydex_sparsevec_support(internal);

-- RRF fusion score function (planner hook / custom scan marker)

CREATE FUNCTION rrf(
	emb1   public.sparsevec,
	op1    pg_catalog.regoperator,
	q1     public.sparsevec,
	emb2   public.vector,
	op2    pg_catalog.regoperator,
	q2     public.vector,
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
	emb1   public.vector,
	op1    pg_catalog.regoperator,
	q1     public.vector,
	emb2   public.sparsevec,
	op2    pg_catalog.regoperator,
	q2     public.sparsevec,
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
	emb1   public.sparsevec,
	op1    pg_catalog.regoperator,
	q1     public.sparsevec,
	emb2   public.sparsevec,
	op2    pg_catalog.regoperator,
	q2     public.sparsevec,
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
	emb1   public.vector,
	op1    pg_catalog.regoperator,
	q1     public.vector,
	emb2   public.vector,
	op2    pg_catalog.regoperator,
	q2     public.vector,
	k      integer            DEFAULT 60,
	w1     double precision   DEFAULT 0.5,
	w2     double precision   DEFAULT 0.5,
	cand1  integer            DEFAULT 200,
	cand2  integer            DEFAULT 200
)
	RETURNS double precision
AS 'MODULE_PATHNAME', 'rrf'
LANGUAGE C STRICT STABLE;
