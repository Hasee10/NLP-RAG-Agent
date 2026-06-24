"""Phase 4 — ingest embedded corpus into Supabase pgvector.

Creates (or reuses) a `medical_chunks` table with:
    id TEXT PRIMARY KEY
    text TEXT
    source TEXT
    type TEXT
    body_system TEXT
    severity_level INT
    tags TEXT[]
    embedding VECTOR(768)

Usage:
    python -m medical-rag.ingest [--batch-size 200] [--clear]

Env vars (in medical-rag/.env or environment):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT.parent / ".env", override=False)  # fallback to project root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest")

PROCESSED = ROOT / "data" / "processed"
TABLE = "medical_chunks"


def get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def ensure_table(sb):
    """Create the medical_chunks table + index if not exists via Supabase RPC."""
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id          TEXT PRIMARY KEY,
        text        TEXT NOT NULL,
        source      TEXT,
        type        TEXT,
        body_system TEXT,
        severity_level INT,
        tags        TEXT[],
        embedding   VECTOR(768)
    );
    CREATE INDEX IF NOT EXISTS {TABLE}_embedding_idx
        ON {TABLE} USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """
    sb.rpc("exec_sql", {"sql": ddl}).execute()
    log.info("table %s ready", TABLE)


def load_embedded(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def upsert_batch(sb, rows: list[dict]):
    records = []
    for r in rows:
        records.append({
            "id": r["id"],
            "text": r["text"],
            "source": r.get("source", ""),
            "type": r.get("type", ""),
            "body_system": r.get("body_system", "general"),
            "severity_level": r.get("severity_level"),
            "tags": r.get("tags", []),
            "embedding": r["embedding"],
        })
    sb.table(TABLE).upsert(records).execute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--clear", action="store_true", help="truncate table before ingest")
    args = ap.parse_args()

    embedded_path = PROCESSED / "corpus_embedded.jsonl"
    assert embedded_path.exists(), "corpus_embedded.jsonl not found — run embed.py first"

    sb = get_client()

    if args.clear:
        log.info("truncating %s...", TABLE)
        sb.table(TABLE).delete().neq("id", "").execute()

    log.info("loading embedded corpus...")
    rows = load_embedded(embedded_path)
    log.info("ingesting %d chunks in batches of %d...", len(rows), args.batch_size)

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        upsert_batch(sb, batch)
        if (i // args.batch_size + 1) % 10 == 0:
            log.info("  %d / %d", i + len(batch), len(rows))

    log.info("DONE  %d chunks ingested into %s", len(rows), TABLE)


if __name__ == "__main__":
    main()
