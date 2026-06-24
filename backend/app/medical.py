"""Medical RAG — retrieve from Supabase pgvector + generate via OpenRouter."""

import logging
import os
import re

from sentence_transformers import SentenceTransformer
from supabase import create_client
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DISCLAIMER = (
    "[!] This information is for educational purposes only and does not constitute "
    "medical advice, diagnosis, or treatment. Always consult a licensed healthcare "
    "professional for personal medical decisions."
)

SYSTEM_PROMPT = """You are a grounded medical information assistant.
Rules:
1. Answer ONLY from the provided <sources>. Do not add outside knowledge.
2. Cite each source you use as [1], [2], etc., matching the source numbers.
3. If the sources don't contain enough information to answer, say so clearly.
4. Never diagnose, prescribe, or give personal medical advice.
5. Keep answers concise (3-5 sentences) and accurate.
6. Always end with a brief summary sentence.
"""

SYSTEM_KW = {
    "cardiovascular": ["heart", "cardiac", "artery", "blood pressure", "coronary", "hypertension"],
    "respiratory": ["lung", "asthma", "respiratory", "pneumonia", "copd", "breath"],
    "neurological": ["brain", "nerve", "seizure", "stroke", "alzheimer", "parkinson", "migraine"],
    "gastrointestinal": ["stomach", "liver", "intestin", "bowel", "colon", "digest", "hepat"],
    "endocrine": ["diabetes", "thyroid", "insulin", "hormone", "adrenal"],
    "renal": ["kidney", "renal", "bladder", "urinary"],
    "musculoskeletal": ["bone", "joint", "muscle", "arthritis", "spine", "fracture"],
    "reproductive": ["pregnan", "ovar", "uterus", "prostate", "menstru"],
    "integumentary": ["skin", "rash", "dermat", "eczema"],
    "immune": ["immune", "hiv", "lupus", "allergy", "autoimmune"],
}

_model: SentenceTransformer | None = None
_sb = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("loading medical embedding model...")
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def _get_sb():
    global _sb
    if _sb is None:
        _sb = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _sb


def _classify(query: str) -> str | None:
    q = query.lower()
    for system, kws in SYSTEM_KW.items():
        if any(k in q for k in kws):
            return system
    return None


def _content_words(text: str) -> set[str]:
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "have",
                 "has", "had", "do", "does", "did", "will", "would", "could",
                 "of", "in", "on", "at", "to", "for", "with", "by", "and",
                 "or", "not", "this", "that", "it", "what", "how", "why"}
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in stopwords and len(w) > 2}


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    model = _get_model()
    sb = _get_sb()

    qvec = model.encode(query, normalize_embeddings=True).tolist()
    body_system = _classify(query)
    fetch_count = max(top_k * 6, 30)

    params: dict = {"query_embedding": qvec, "match_count": fetch_count}
    if body_system:
        params["filter_body_system"] = body_system

    try:
        results = sb.rpc("match_medical_chunks", params).execute().data or []
    except Exception:
        log.exception("Supabase RPC failed; retrying unfiltered")
        results = sb.rpc("match_medical_chunks", {
            "query_embedding": qvec, "match_count": fetch_count
        }).execute().data or []

    qwords = _content_words(query)

    def rank(r):
        return (len(qwords & _content_words(r.get("text", ""))), float(r.get("similarity") or 0))

    results.sort(key=rank, reverse=True)
    return results[:top_k]


async def answer(query: str, top_k: int = 5) -> dict:
    body_system = _classify(query)
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "I could not find relevant medical information for your query.",
            "citations": [],
            "disclaimer": DISCLAIMER,
            "chunks": [],
            "model": "none",
            "body_system": body_system,
            "chunk_type": None,
            "sources_used": 0,
        }

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GROQ_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/auto")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sources = "\n\n".join(
        f"[{i+1}] ({c.get('source','unknown')} / {c.get('type','')}): {c.get('text','')[:600]}"
        for i, c in enumerate(chunks)
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\n<sources>\n{sources}\n</sources>"},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    answer_text = resp.choices[0].message.content.strip()
    citation_ids = [c["id"] for c in chunks if f"[{chunks.index(c)+1}]" in answer_text]
    clean_chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]

    return {
        "answer": answer_text,
        "citations": citation_ids,
        "disclaimer": DISCLAIMER,
        "chunks": clean_chunks,
        "model": model,
        "body_system": body_system,
        "chunk_type": None,
        "sources_used": len(chunks),
    }
