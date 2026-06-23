---
title: RAG Sentiment Backend
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# RAG Sentiment Analysis — FastAPI Backend

Serves the from-scratch PyTorch RAG pipeline (encode → retrieve → explain).

- `GET /health` — liveness
- `GET /stats` — vectors in Supabase
- `POST /query` — `{ "review": "...", "top_k": 5 }` → sentiment + retrieved + explanation

Built from this repo's `Dockerfile` (CPU PyTorch). Set the Supabase + OpenRouter
secrets in **Settings → Variables and secrets** before first boot.
