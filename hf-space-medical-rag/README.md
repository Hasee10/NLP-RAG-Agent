---
title: Medical RAG Backend
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Medical RAG Backend

FastAPI backend for grounded medical Q&A.  
- 37,000+ MedQuAD chunks in Supabase pgvector  
- `all-MiniLM-L6-v2` semantic retrieval  
- OpenRouter LLM generation (cited, disclaimered)

## Endpoints

- `GET /health` — liveness check  
- `POST /query` — `{"query": "...", "top_k": 5}`

## Required Secrets (set in Space Settings → Variables and secrets)

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `OPENROUTER_API_KEY`
