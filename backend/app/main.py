"""FastAPI entrypoint.

Run locally:
    cd backend
    uvicorn app.main:app --reload --port 8000

Docs at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from . import db, rag
from .schemas import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse,
)

settings = get_settings()
app = FastAPI(title="RAG Sentiment Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.app_env, "model": settings.openrouter_model}


@app.get("/stats")
def stats():
    try:
        return {"reviews_in_db": db.count_reviews()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Supabase error: {e}")


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        result = rag.ingest([r.model_dump() for r in req.reviews])
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        result = await rag.answer(req.review, top_k=req.top_k)
        return QueryResponse(**result)
    except RuntimeError as e:           # e.g. missing OpenRouter key
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
