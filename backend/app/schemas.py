"""Request/response models for the API."""

from pydantic import BaseModel, Field

_MAX_REVIEW_LEN = 5_000   # ~1,000 words; prevents memory/storage abuse
_MAX_INGEST_BATCH = 500   # single ingest call cap


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=_MAX_REVIEW_LEN)
    sentiment: str | None = Field(None, max_length=20)
    length_label: int | None = None


class IngestRequest(BaseModel):
    reviews: list[ReviewIn] = Field(..., min_length=1, max_length=_MAX_INGEST_BATCH)


class IngestResponse(BaseModel):
    inserted: int
    total_in_db: int


class QueryRequest(BaseModel):
    review: str = Field(..., min_length=3, max_length=_MAX_REVIEW_LEN)
    top_k: int = Field(5, ge=1, le=20)
    force_llm: bool = Field(False, description="Skip the from-scratch decoder and use the LLM directly")


class RetrievedItem(BaseModel):
    text: str
    sentiment: str | None = None
    similarity: float | None = None


class QueryResponse(BaseModel):
    review: str
    predicted_sentiment: str
    retrieved: list[RetrievedItem]
    explanation: str
    generator: str = "decoder"   # "decoder" | "llm" | "rule-based"
    model: str
