"""FastAPI entrypoint.

Run locally:
    cd backend
    uvicorn app.main:app --reload --port 8000

Docs at http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from . import db, rag
from .schemas import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="RAG Sentiment Agent",
    version="1.0.0",
    # Hide internal details from OpenAPI docs in production
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url="/redoc" if settings.app_env != "production" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.app_env}


@app.get("/stats")
def stats():
    try:
        return {"reviews_in_db": db.count_reviews()}
    except Exception:
        logger.exception("stats: database error")
        raise HTTPException(status_code=502, detail="Service temporarily unavailable")


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        result = rag.ingest([r.model_dump() for r in req.reviews])
        return IngestResponse(**result)
    except Exception:
        logger.exception("ingest: failed")
        raise HTTPException(status_code=502, detail="Ingest failed")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        result = await rag.answer(req.review, top_k=req.top_k, force_llm=req.force_llm)
        return QueryResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("query: pipeline error")
        raise HTTPException(status_code=502, detail="Query failed")
