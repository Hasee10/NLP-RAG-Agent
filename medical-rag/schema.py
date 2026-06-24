"""Canonical chunk schema for the medical RAG corpus (Phase 1).

Every collector emits records validated against this schema so downstream
phases (chunking, dedup, ingest) can rely on a stable shape.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Optional

BODY_SYSTEMS = {
    "cardiovascular", "respiratory", "neurological", "musculoskeletal",
    "gastrointestinal", "endocrine", "reproductive", "integumentary",
    "renal", "immune", "general",
}

CHUNK_TYPES = {"anatomy", "drug", "symptom", "diagnosis", "qa_pair", "sentiment_map"}


@dataclass
class Chunk:
    text: str
    source: str
    type: str
    body_system: str = "general"
    severity_level: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    id: str = ""

    def __post_init__(self):
        if self.type not in CHUNK_TYPES:
            raise ValueError(f"invalid type {self.type!r}; must be one of {sorted(CHUNK_TYPES)}")
        if self.body_system not in BODY_SYSTEMS:
            self.body_system = "general"
        if self.severity_level is not None and not (1 <= self.severity_level <= 5):
            raise ValueError(f"severity_level must be 1-5 or None, got {self.severity_level}")
        self.text = " ".join(str(self.text).split())  # normalise whitespace
        if not self.id:
            self.id = hashlib.sha1(f"{self.source}:{self.text}".encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)
