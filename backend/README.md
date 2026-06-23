# Backend — RAG Sentiment Agent (FastAPI + Supabase + OpenRouter)

Production-style RAG service. Retrieval embeddings come from the project's **own
trained encoder** (128-dim); generation is delegated to an **OpenRouter** LLM;
vectors live in **Supabase pgvector**.

```
review ──▶ encoder (128-d vector) ──▶ Supabase match_reviews (cosine top-k) ──▶ OpenRouter LLM ──▶ grounded explanation
```

## Layout

```
backend/
├── app/
│   ├── config.py       # typed settings from repo-root .env
│   ├── embeddings.py   # serves the Task-A encoder (singleton)
│   ├── db.py           # Supabase client + pgvector helpers
│   ├── llm.py          # OpenRouter chat/completions
│   ├── rag.py          # orchestration ("tools" layer): embed → retrieve → generate
│   ├── schemas.py      # request/response models
│   └── main.py         # FastAPI app + routes
├── migrations/0001_init.sql   # pgvector table + match_reviews() function
├── scripts/ingest_corpus.py   # bulk-embed train.csv into Supabase
└── requirements.txt
```

## Setup

1. **Fill `.env`** at the repo root (already scaffolded): set `SUPABASE_DB_URL`
   password and `OPENROUTER_API_KEY`.

2. **Install deps**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Create the schema** — paste `backend/migrations/0001_init.sql` into
   Supabase → SQL Editor and run it.

4. **Ingest the corpus** (embeds with the encoder, inserts vectors)
   ```bash
   python -m backend.scripts.ingest_corpus --limit 5000
   ```

5. **Run the API**
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```
   Open http://localhost:8000/docs

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | liveness + active model |
| GET  | `/stats`  | number of reviews in the vector DB |
| POST | `/ingest` | embed + insert a batch of reviews |
| POST | `/query`  | full RAG: encode → retrieve → grounded explanation |

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"review":"This thing broke after two days, total waste of money.","top_k":5}'
```

## Security

- `service_role` key is used **server-side only**. Never send it to the frontend.
- The Vercel frontend talks to this API (and at most uses the Supabase `anon` key).
- Rotate the `service_role` key once if it was ever shared in plaintext.
