import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retrieve import classify_query, retrieve
from generate import generate_answer, DISCLAIMER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")

app = FastAPI(title="Medical RAG API", version="1.0")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


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
