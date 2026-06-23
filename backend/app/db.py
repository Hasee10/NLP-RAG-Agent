"""Supabase access layer (service-role client + pgvector helpers)."""

from functools import lru_cache

from supabase import create_client, Client

from .config import get_settings


@lru_cache
def get_client() -> Client:
    s = get_settings()
    # service_role key — server-side only; bypasses RLS for ingest + retrieval.
    return create_client(s.supabase_url, s.supabase_service_role_key)


def upsert_reviews(rows: list[dict]) -> int:
    """rows: [{text, sentiment, length_label, embedding: list[float]}]"""
    if not rows:
        return 0
    client = get_client()
    client.table("reviews").insert(rows).execute()
    return len(rows)


def _to_pgvector(vec: list[float]) -> str:
    """pgvector RPC params must be passed as a string literal '[v1,v2,...]',
    not a JSON array (PostgREST won't bind a JSON array to a vector parameter)."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def match_reviews(query_embedding: list[float], match_count: int = 5) -> list[dict]:
    """Cosine top-k via the match_reviews SQL function (see migrations/0001_init.sql)."""
    client = get_client()
    resp = client.rpc(
        "match_reviews",
        {"query_embedding": _to_pgvector(query_embedding), "match_count": match_count},
    ).execute()
    return resp.data or []


def count_reviews() -> int:
    client = get_client()
    resp = client.table("reviews").select("id", count="exact").limit(1).execute()
    return resp.count or 0
