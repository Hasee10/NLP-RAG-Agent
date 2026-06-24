"""Medical RAG query API — end-to-end: embed → retrieve → generate.

Exposes:
    GET  /health
    POST /query  {"query": "...", "top_k": 5, "force_llm": false}
             ->  {"answer": "...", "citations": [...], "disclaimer": "...",
                  "chunks": [...], "model": "...", "body_system": "...", "type": "..."}

Run (dev):
    uvicorn medical-rag.query:app --port 8001 --reload
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT.parent / ".env", override=False)

from retrieve import classify_query, retrieve  # noqa: E402
from generate import generate_answer, DISCLAIMER  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("query")

app = FastAPI(title="Medical RAG API", version="1.0")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    force_llm: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    disclaimer: str
    chunks: list[dict]
    model: str
    body_system: str | None
    chunk_type: str | None
    sources_used: int


@app.get("/health")
def health():
    return {"status": "ok", "service": "medical-rag"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    body_system, chunk_type = classify_query(req.query)
    chunks = retrieve(req.query, top_k=req.top_k)

    if not chunks:
        return QueryResponse(
            answer="I could not find relevant medical information for your query.",
            citations=[],
            disclaimer=DISCLAIMER,
            chunks=[],
            model="none",
            body_system=body_system,
            chunk_type=chunk_type,
            sources_used=0,
        )

    result = await generate_answer(req.query, chunks)

    # Strip embeddings from returned chunks (large + not needed by client)
    clean_chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        disclaimer=result["disclaimer"],
        chunks=clean_chunks,
        model=result["model"],
        body_system=body_system,
        chunk_type=chunk_type,
        sources_used=result["sources_used"],
    )
