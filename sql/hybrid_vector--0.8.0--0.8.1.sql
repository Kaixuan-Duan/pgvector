-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION hybrid_vector UPDATE TO '0.8.1'" to load this file. \quit

ALTER EXTENSION hybrid_vector DROP FUNCTION rrf(sparsevec, regoperator, sparsevec, vector, regoperator, vector, integer, double precision, double precision, integer, integer);
DROP FUNCTION rrf(sparsevec, regoperator, sparsevec, vector, regoperator, vector, integer, double precision, double precision, integer, integer);

ALTER EXTENSION hybrid_vector DROP FUNCTION rrf(vector, regoperator, vector, sparsevec, regoperator, sparsevec, integer, double precision, double precision, integer, integer);
DROP FUNCTION rrf(vector, regoperator, vector, sparsevec, regoperator, sparsevec, integer, double precision, double precision, integer, integer);

ALTER EXTENSION hybrid_vector DROP FUNCTION rrf(sparsevec, regoperator, sparsevec, sparsevec, regoperator, sparsevec, integer, double precision, double precision, integer, integer);
DROP FUNCTION rrf(sparsevec, regoperator, sparsevec, sparsevec, regoperator, sparsevec, integer, double precision, double precision, integer, integer);

ALTER EXTENSION hybrid_vector DROP FUNCTION rrf(vector, regoperator, vector, vector, regoperator, vector, integer, double precision, double precision, integer, integer);
DROP FUNCTION rrf(vector, regoperator, vector, vector, regoperator, vector, integer, double precision, double precision, integer, integer);

CREATE FUNCTION linear(
    emb1   sparsevec,
    op1    regoperator,
    q1     sparsevec,
    emb2   vector,
    op2    regoperator,
    q2     vector,
    k      integer            DEFAULT 60,
    w1     double precision   DEFAULT 0.5,
    w2     double precision   DEFAULT 0.5,
    cand1  integer            DEFAULT 200,
    cand2  integer            DEFAULT 200
)
    RETURNS double precision
AS 'MODULE_PATHNAME', 'linear'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION linear(
    emb1   vector,
    op1    regoperator,
    q1     vector,
    emb2   sparsevec,
    op2    regoperator,
    q2     sparsevec,
    k      integer            DEFAULT 60,
    w1     double precision   DEFAULT 0.5,
    w2     double precision   DEFAULT 0.5,
    cand1  integer            DEFAULT 200,
    cand2  integer            DEFAULT 200
)
    RETURNS double precision
AS 'MODULE_PATHNAME', 'linear'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION linear(
    emb1   sparsevec,
    op1    regoperator,
    q1     sparsevec,
    emb2   sparsevec,
    op2    regoperator,
    q2     sparsevec,
    k      integer            DEFAULT 60,
    w1     double precision   DEFAULT 0.5,
    w2     double precision   DEFAULT 0.5,
    cand1  integer            DEFAULT 200,
    cand2  integer            DEFAULT 200
)
    RETURNS double precision
AS 'MODULE_PATHNAME', 'linear'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION linear(
    emb1   vector,
    op1    regoperator,
    q1     vector,
    emb2   vector,
    op2    regoperator,
    q2     vector,
    k      integer            DEFAULT 60,
    w1     double precision   DEFAULT 0.5,
    w2     double precision   DEFAULT 0.5,
    cand1  integer            DEFAULT 200,
    cand2  integer            DEFAULT 200
)
    RETURNS double precision
AS 'MODULE_PATHNAME', 'linear'
LANGUAGE C STRICT STABLE;
