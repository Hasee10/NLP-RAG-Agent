import logging
import os
import re

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retrieve")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
        log.info("loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def _get_sb():
    global _sb
    if _sb is None:
        _sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    return _sb


def classify_query(query: str) -> tuple[str | None, str | None]:
    q = query.lower()
    body_system = None
    for system, kws in SYSTEM_KW.items():
        if any(k in q for k in kws):
            body_system = system
            break
    return body_system, None


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

    fetch_count = max(top_k * 6, 30)
    params: dict = {"query_embedding": qvec, "match_count": fetch_count}
    if body_system:
        params["filter_body_system"] = body_system

    try:
        resp = sb.rpc("match_medical_chunks", params).execute()
        results = resp.data or []
    except Exception:
        log.exception("Supabase RPC failed; trying unfiltered fallback")
        resp = sb.rpc("match_medical_chunks", {
            "query_embedding": qvec, "match_count": fetch_count
        }).execute()
        results = resp.data or []

    qwords = _content_words(query)

    def relevance(r):
        overlap = len(qwords & _content_words(r.get("text", "")))
        sim = float(r.get("similarity") or 0.0)
        return (overlap, sim)

    results.sort(key=relevance, reverse=True)
    return results[:top_k]
