"""Phase 5 — retrieval: query classification + metadata-filtered semantic search.

Exposes:
    retrieve(query: str, top_k: int = 5) -> list[dict]

Query classification is keyword-based (fast, no extra model):
    - body_system filter: cardiovascular / neurological / etc.
    - type filter:        drug / symptom / qa_pair / anatomy / diagnosis

Usage (standalone test):
    python -m medical-rag.retrieve "What are the symptoms of diabetes?"
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retrieve")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE = "medical_chunks"

# ── keyword → body_system (mirrors medquad collector) ──────────────────────
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

TYPE_KW = {
    "drug": ["drug", "medication", "dose", "dosage", "pill", "tablet", "injection", "side effect",
             "contraindication", "prescription", "antibiotic", "vaccine"],
    "symptom": ["symptom", "sign", "feel", "pain", "ache", "swell", "fever", "nausea"],
    "anatomy": ["anatomy", "organ", "structure", "bone", "muscle", "nerve", "tissue"],
    "diagnosis": ["diagnos", "test", "biopsy", "scan", "mri", "ct scan", "blood test", "screening"],
}

_model: SentenceTransformer | None = None
_sb = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def _get_sb():
    global _sb
    if _sb is None:
        _sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    return _sb


def classify_query(query: str) -> tuple[str | None, str | None]:
    """Return (body_system, chunk_type) filters, or None if no strong signal."""
    q = query.lower()
    body_system = None
    for system, kws in SYSTEM_KW.items():
        if any(k in q for k in kws):
            body_system = system
            break

    chunk_type = None
    for ctype, kws in TYPE_KW.items():
        if any(k in q for k in kws):
            chunk_type = ctype
            break
    # qa_pair is the catch-all for general questions
    if chunk_type is None and re.search(r"\b(what|how|why|when|is|are|can|does|do)\b", q):
        chunk_type = "qa_pair"

    return body_system, chunk_type


def _content_words(text: str) -> set[str]:
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "of", "in", "on",
                 "at", "to", "for", "with", "by", "from", "and", "or", "but",
                 "not", "this", "that", "it", "its", "they", "their", "what",
                 "how", "why", "when", "where", "which", "who"}
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in stopwords and len(w) > 2}


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    model = _get_model()
    sb = _get_sb()

    qvec = model.encode(query, normalize_embeddings=True).tolist()
    body_system, chunk_type = classify_query(query)
    log.info("query classified → body_system=%s  type=%s", body_system, chunk_type)

    # Over-fetch for lexical re-ranking
    fetch_count = max(top_k * 6, 30)

    # Build Supabase RPC call — match_medical_chunks function (created in Phase 4 SQL)
    params: dict = {"query_embedding": qvec, "match_count": fetch_count}
    if body_system:
        params["filter_body_system"] = body_system
    if chunk_type:
        params["filter_type"] = chunk_type

    try:
        resp = sb.rpc("match_medical_chunks", params).execute()
        results = resp.data or []
    except Exception:
        log.exception("Supabase RPC failed; trying unfiltered fallback")
        resp = sb.rpc("match_medical_chunks", {
            "query_embedding": qvec, "match_count": fetch_count
        }).execute()
        results = resp.data or []

    # Lexical re-rank: boost chunks sharing content words with query
    qwords = _content_words(query)
    def relevance(r):
        overlap = len(qwords & _content_words(r.get("text", "")))
        sim = float(r.get("similarity") or 0.0)
        return (overlap, sim)

    results.sort(key=relevance, reverse=True)
    return results[:top_k]


def main():
    import sys
    query = " ".join(sys.argv[1:]) or "What are the symptoms of diabetes?"
    log.info("query: %r", query)
    results = retrieve(query)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] sim={r.get('similarity', 0):.3f}  type={r.get('type')}  system={r.get('body_system')}")
        print(f"    {r['text'][:200]}...")


if __name__ == "__main__":
    main()
