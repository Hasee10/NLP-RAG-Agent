-- Medical RAG — Supabase pgvector setup
-- Run this in the Supabase SQL Editor once before ingestion.
-- Model: all-MiniLM-L6-v2 → 384-dim embeddings

-- 1. Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Drop and recreate table (safe: table is empty before first ingest)
DROP TABLE IF EXISTS medical_chunks CASCADE;
CREATE TABLE medical_chunks (
    id             TEXT PRIMARY KEY,
    text           TEXT NOT NULL,
    source         TEXT,
    type           TEXT,
    body_system    TEXT,
    severity_level INT,
    tags           TEXT[],
    embedding      VECTOR(384)
);

-- 3. IVFFlat index for fast ANN search (lists = sqrt(37516) ≈ 100)
CREATE INDEX IF NOT EXISTS medical_chunks_embedding_idx
    ON medical_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 4. Match function — supports optional body_system and type filters
CREATE OR REPLACE FUNCTION match_medical_chunks(
    query_embedding     VECTOR(384),
    match_count         INT      DEFAULT 10,
    filter_body_system  TEXT     DEFAULT NULL,
    filter_type         TEXT     DEFAULT NULL
)
RETURNS TABLE (
    id             TEXT,
    text           TEXT,
    source         TEXT,
    type           TEXT,
    body_system    TEXT,
    severity_level INT,
    tags           TEXT[],
    similarity     FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        mc.id,
        mc.text,
        mc.source,
        mc.type,
        mc.body_system,
        mc.severity_level,
        mc.tags,
        1 - (mc.embedding <=> query_embedding) AS similarity
    FROM medical_chunks mc
    WHERE
        (filter_body_system IS NULL OR mc.body_system = filter_body_system)
        AND (filter_type IS NULL OR mc.type = filter_type)
    ORDER BY mc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
