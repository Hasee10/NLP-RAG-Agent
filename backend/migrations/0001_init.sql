-- Supabase / pgvector schema for the RAG sentiment agent.
-- Run this in the Supabase dashboard → SQL Editor (or via the Supabase MCP).
-- Embedding dim = 128 (the project's own Task-A encoder).

create extension if not exists vector;

create table if not exists reviews (
    id           bigint generated always as identity primary key,
    text         text not null,
    sentiment    text,
    length_label int,
    embedding    vector(128),
    created_at   timestamptz default now()
);

-- Approximate-nearest-neighbour index for cosine distance.
-- (Build/refresh after bulk ingest; lists ~= sqrt(rows).)
create index if not exists reviews_embedding_idx
    on reviews using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Cosine top-k retrieval, called from the backend via supabase.rpc('match_reviews').
create or replace function match_reviews(
    query_embedding vector(128),
    match_count     int default 5
)
returns table (
    id           bigint,
    text         text,
    sentiment    text,
    length_label int,
    similarity   float
)
language sql stable
as $$
    select
        r.id,
        r.text,
        r.sentiment,
        r.length_label,
        1 - (r.embedding <=> query_embedding) as similarity
    from reviews r
    where r.embedding is not null
    order by r.embedding <=> query_embedding
    limit match_count;
$$;
