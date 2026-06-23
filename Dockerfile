# ── RAG Sentiment backend (FastAPI + CPU PyTorch) ───────────────────────────
# Build context = repo root, because the backend imports rag_common.py and reads
# data/vocab.json + models/encoder_best.pt from the repo root.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only PyTorch FIRST so we never pull the multi-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Remaining Python deps (torch>=2.0 is already satisfied by the CPU wheel above).
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

# App code + the exact artifacts the query path needs (kept minimal on purpose).
COPY rag_common.py            ./rag_common.py
COPY data/vocab.json          ./data/vocab.json
COPY models/encoder_best.pt   ./models/encoder_best.pt
COPY backend                  ./backend

# Run from backend/ so `app.main:app` and REPO_ROOT (=/app) resolve correctly.
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
