"""RAG orchestration — the 'tools' layer that wires embeddings + retrieval + LLM.

This is the single place where the pipeline lives:

    embed(review)  ->  match_reviews(vector)  ->  LLM grounds an explanation
"""

from . import db
from .embeddings import get_embedder
from .llm import generate_explanation


def ingest(reviews: list[dict]) -> dict:
    """reviews: [{text, sentiment?, length_label?}] -> embed + insert into Supabase."""
    embedder = get_embedder()
    texts = [r["text"] for r in reviews]
    vecs = embedder.embed(texts)
    rows = [
        {
            "text": r["text"],
            "sentiment": r.get("sentiment"),
            "length_label": r.get("length_label"),
            "embedding": vec.tolist(),
        }
        for r, vec in zip(reviews, vecs)
    ]
    inserted = db.upsert_reviews(rows)
    return {"inserted": inserted, "total_in_db": db.count_reviews()}


async def answer(review: str, top_k: int = 5) -> dict:
    """Full RAG query: encode -> retrieve -> generate grounded explanation."""
    embedder = get_embedder()
    qvec = embedder.embed_one(review)
    predicted = embedder.predict_sentiment(review)
    retrieved = db.match_reviews(qvec, match_count=top_k)
    llm_out = await generate_explanation(review, retrieved)
    return {
        "review": review,
        "predicted_sentiment": predicted,
        "retrieved": retrieved,
        "explanation": llm_out["text"],
        "model": llm_out["model"],
    }
