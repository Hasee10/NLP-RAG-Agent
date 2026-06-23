"""Bulk-ingest the training corpus into Supabase pgvector.

Embeds each review with the project's own encoder and inserts it into the
`reviews` table (run the migration in backend/migrations/0001_init.sql first).

Usage (from repo root, with .env populated):
    python -m backend.scripts.ingest_corpus --limit 5000 --batch 256
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# allow `python backend/scripts/ingest_corpus.py` as well as -m
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.app import db                       # noqa: E402
from backend.app.embeddings import get_embedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO_ROOT / "data" / "train.csv"))
    ap.add_argument("--limit", type=int, default=5000, help="max reviews to ingest")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"Loaded {len(df)} reviews from {args.csv}")

    embedder = get_embedder()
    total = 0
    for s in range(0, len(df), args.batch):
        chunk = df.iloc[s : s + args.batch]
        vecs = embedder.embed(chunk["text"].astype(str).tolist())
        rows = [
            {
                "text": str(r["text"]),
                "sentiment": r.get("sentiment"),
                "length_label": int(r["length_label"]) if pd.notna(r.get("length_label")) else None,
                "embedding": vec.tolist(),
            }
            for (_, r), vec in zip(chunk.iterrows(), vecs)
        ]
        db.upsert_reviews(rows)
        total += len(rows)
        print(f"  ingested {total}/{len(df)}")

    print(f"Done. {total} reviews ingested. DB now holds {db.count_reviews()}.")


if __name__ == "__main__":
    main()
