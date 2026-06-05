DO $$
BEGIN
	RAISE EXCEPTION 'Automatic upgrade from hybrid_vector 0.8.0 to 0.8.1 is not supported. Use a fresh install or perform a manual migration because hybrid_vector no longer owns pgvector SQL types and functions.';
END
$$;
