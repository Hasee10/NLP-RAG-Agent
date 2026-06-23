"""Request/response models for the API."""

from pydantic import BaseModel, Field


class ReviewIn(BaseModel):
    text: str
    sentiment: str | None = None
    length_label: int | None = None


class IngestRequest(BaseModel):
    reviews: list[ReviewIn]


class IngestResponse(BaseModel):
    inserted: int
    total_in_db: int


class QueryRequest(BaseModel):
    review: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class RetrievedItem(BaseModel):
    text: str
    sentiment: str | None = None
    similarity: float | None = None


class QueryResponse(BaseModel):
    review: str
    predicted_sentiment: str
    retrieved: list[RetrievedItem]
    explanation: str
    model: str
