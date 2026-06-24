-- Run AFTER the HNSW index is built (supabase_hnsw_index.sql).
-- Updates match_medical_chunks to set ef_search = 100 for better recall.
--
-- ef_search controls how many candidates the HNSW graph explores at query time.
-- Higher value = better recall (finds the true nearest neighbours more reliably)
--             = slightly slower (but still ms-range)
-- 100 is a good balance for medical search where recall matters more than raw speed.

CREATE OR REPLACE FUNCTION match_medical_chunks(
  query_embedding vector(384),
  match_count     int     DEFAULT 10,
  filter_body_system text  DEFAULT NULL,
  filter_type        text  DEFAULT NULL
)
RETURNS TABLE (
  id          text,
  text        text,
  source      text,
  type        text,
  body_system text,
  similarity  float
)
LANGUAGE plpgsql
AS $$
BEGIN
  -- Set HNSW ef_search for this query (higher = better recall)
  PERFORM set_config('hnsw.ef_search', '100', true);

  RETURN QUERY
  SELECT
    mc.id,
    mc.text,
    mc.source,
    mc.type,
    mc.body_system,
    1 - (mc.embedding <=> query_embedding) AS similarity
  FROM medical_chunks mc
  WHERE
    (filter_body_system IS NULL OR mc.body_system = filter_body_system)
    AND (filter_type IS NULL OR mc.type = filter_type)
  ORDER BY mc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
