"""Phase 3 — encode corpus with off-the-shelf biomedical sentence model.

corpus.jsonl  ->  embeddings (768-dim)  ->  corpus_embedded.jsonl
                                         ->  embeddings.npy  (for quick local inspection)

Model: pritamdeka/S-PubMedBert-MS-MARCO (768-dim, CPU-runnable, no fine-tuning needed for MVP).

Usage:
    python -m medical-rag.embed [--batch-size 64] [--device cpu]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("embed")

PROCESSED = ROOT / "data" / "processed"
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_corpus(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    corpus_path = PROCESSED / "corpus.jsonl"
    assert corpus_path.exists(), f"corpus.jsonl not found — run preprocess.py first"

    log.info("loading corpus...")
    corpus = load_corpus(corpus_path)
    log.info("corpus: %d chunks", len(corpus))

    log.info("loading model %s on %s...", MODEL_NAME, args.device)
    model = SentenceTransformer(MODEL_NAME, device=args.device)

    texts = [c["text"] for c in corpus]
    log.info("encoding %d texts (batch=%d)...", len(texts), args.batch_size)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity = dot product
        convert_to_numpy=True,
    )
    log.info("embeddings shape: %s", embeddings.shape)

    # Save raw numpy array for quick inspection / offline use
    npy_path = PROCESSED / "embeddings.npy"
    np.save(npy_path, embeddings)
    log.info("saved %s", npy_path)

    # Attach embeddings to corpus records
    embedded_path = PROCESSED / "corpus_embedded.jsonl"
    with open(embedded_path, "w", encoding="utf-8") as f:
        for chunk, vec in zip(corpus, embeddings):
            chunk["embedding"] = vec.tolist()
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("saved %s", embedded_path)
    log.info("DONE  %d chunks embedded", len(corpus))


if __name__ == "__main__":
    main()
