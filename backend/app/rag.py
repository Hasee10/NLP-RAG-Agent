"""RAG orchestration — the 'tools' layer that wires embeddings + retrieval + LLM.

This is the single place where the pipeline lives:

    embed(review)  ->  match_reviews(vector)  ->  LLM grounds an explanation
"""

import logging
import re

from . import db
from .embeddings import get_embedder
from .llm import generate_explanation

logger = logging.getLogger("rag")

_STOP = set(
    "the a an and or but if of to in on for with from by as at is are was were be this that it its "
    "they them you your i we our my me have has had do does did will would can could should not no".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", str(text).lower()) if len(w) >= 3 and w not in _STOP}


def _retrieve_relevant(qvec: list[float], review: str, top_k: int) -> list[dict]:
    """Over-fetch the sentiment-nearest neighbours, then re-rank by lexical topic
    overlap with the query (the encoder embeds by sentiment, so this restores
    topic relevance), with embedding similarity as the tiebreak."""
    pool = db.match_reviews(qvec, match_count=max(top_k * 6, 30))
    if len(pool) <= top_k:
        return pool
    qwords = _content_words(review)

    def relevance(r: dict):
        overlap = len(qwords & _content_words(r.get("text", "")))
        return (overlap, float(r.get("similarity") or 0.0))

    return sorted(pool, key=relevance, reverse=True)[:top_k]


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


async def _explain(review: str, sentiment: str, retrieved: list[dict], force_llm: bool):
    """Generate the grounded explanation. Primary = from-scratch decoder (Task C);
    fallback = Groq LLM; final safety net = rule-based. Returns (text, generator, model)."""
    if not force_llm:
        try:
            from .generator import get_generator
            text = get_generator().generate(review, sentiment, retrieved)
            if text and len(text.split()) >= 3:
                return text, "decoder", "from-scratch-decoder"
            logger.warning("decoder output too short (%r); falling back to LLM", text)
        except Exception:
            logger.exception("decoder generation failed; falling back to LLM")

    try:
        out = await generate_explanation(review, retrieved)
        return out["text"], "llm", out["model"]
    except Exception:
        logger.exception("LLM fallback failed; using rule-based explanation")
        return (
            f"This review reads as {sentiment.lower()} based on the retrieved similar reviews.",
            "rule-based",
            "none",
        )


async def answer(review: str, top_k: int = 5, force_llm: bool = False) -> dict:
    """Full RAG query: encode -> retrieve -> generate grounded explanation."""
    embedder = get_embedder()
    qvec = embedder.embed_one(review)
    predicted = embedder.predict_sentiment(review)
    retrieved = _retrieve_relevant(qvec, review, top_k)
    explanation, generator, model = await _explain(review, predicted, retrieved, force_llm)
    return {
        "review": review,
        "predicted_sentiment": predicted,
        "retrieved": retrieved,
        "explanation": explanation,
        "generator": generator,
        "model": model,
    }
