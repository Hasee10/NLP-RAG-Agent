-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- Adds an HNSW index for approximate nearest-neighbour search on the embedding column.
-- Without this, pgvector does a sequential scan over all 38K+ rows on every query.
--
-- HNSW vs IVFFlat:
--   HNSW  — better recall, faster queries, no training step. Best for <1M vectors.
--   IVFFlat — faster to build, slightly lower recall. Use for >1M vectors.
--
-- Parameters:
--   m              = number of bidirectional links per layer (16 is standard)
--   ef_construction = size of dynamic candidate list during build (64 = good recall/speed balance)
--
-- Build time: ~2-5 min for 38K rows on Supabase free tier. Run during off-peak.
-- After the index is built, query latency drops from ~500ms to ~20-50ms.

-- Step 1: Set a higher maintenance memory for the index build (optional but helpful)
SET maintenance_work_mem = '256MB';

-- Step 2: Create the HNSW index
CREATE INDEX IF NOT EXISTS medical_chunks_embedding_hnsw
ON medical_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Step 3: After index is built, also index the body_system column
--         so filtered queries (body_system = 'oral') use the index efficiently
CREATE INDEX IF NOT EXISTS medical_chunks_body_system_idx
ON medical_chunks (body_system);

-- Step 4: Tune the query-time ef_search parameter (higher = better recall, slower)
-- This is a session-level setting; set it in the match_medical_chunks function instead:
-- ALTER FUNCTION match_medical_chunks SET hnsw.ef_search = 100;

-- Verify:
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'medical_chunks';
