# Medical RAG (multi-domain)

A grounded **medical-information retrieval** system: it answers anatomy, symptom,
drug, and diagnosis questions using retrieved medical knowledge, with a second
"sentiment layer" that maps lay descriptions to clinical severity.

> ⚠️ **Information / education tool only.** Every answer is grounded in retrieved
> sources, cited, and carries a "consult a licensed physician" disclaimer. It does
> **not** diagnose individuals or prescribe. It refuses when context is insufficient.

## Architecture (MVP)
```
query → retrieve (Supabase pgvector, metadata-filtered) → grounded LLM (Groq) → cited answer
        domain layer  (drugs, symptoms, QA, anatomy)
        sentiment layer (lay phrase → clinical severity)
```
Embeddings: off-the-shelf biomedical sentence model (`pritamdeka/S-PubMedBert-MS-MARCO`,
768-dim, CPU-runnable). No fine-tuning required for the MVP.

## Phases
1. **Data collection** ✅ — `collectors/` → `data/raw/*.json` (canonical `schema.Chunk`)
2. **Preprocessing** ✅ — `preprocess.py` → `corpus.jsonl` + `train.jsonl` + `val.jsonl`
3. **Embeddings** ✅ — `embed.py` → `corpus_embedded.jsonl` + `embeddings.npy`
4. **Vector DB** ✅ — `ingest.py` → Supabase `medical_chunks(vector(768))`
5. **Retrieval** ✅ — `retrieve.py` → semantic + keyword-classified metadata filter
6. **Generation** ✅ — `generate.py` + `query.py` → grounded, cited, disclaimered API
7. Monitoring/eval — deferred to post-MVP

## Phase 1 sources (free / legitimate)
| Source | Type | Records |
|---|---|---|
| OpenFDA drug labels | `drug` | ~450 |
| MedQuAD (NIH/CDC/NLM) | `qa_pair` | ~16,400 |
| Human Phenotype Ontology | `symptom` | ~17,300 |
| Curated sentiment→severity | `sentiment_map` | 30 |

Phase 2 output (gitignored, regenerable): 37,516 deduplicated corpus chunks.

Deferred (redundant with MedQuAD for MVP): MedlinePlus, Gray's Anatomy, UMLS,
MIMIC-III (credentialed), DrugBank-full (licensed).

## Full pipeline run order
```bash
pip install -r requirements.txt

# Phase 1 — collect
python -m medical-rag.collectors.openfda_drugs --limit 300
python -m medical-rag.collectors.medquad_qa
python -m medical-rag.collectors.hpo_symptoms
python -m medical-rag.collectors.sentiment_severity
python -m medical-rag.tests.validate_phase1   # verify

# Phase 2 — preprocess (slow: BM25 mining ~30 min on CPU)
python -m medical-rag.preprocess --max-anchors 6000

# Phase 3 — embed (requires sentence-transformers + torch)
python -m medical-rag.embed

# Phase 4 — ingest to Supabase (run supabase_setup.sql in SQL Editor first)
python -m medical-rag.ingest --clear

# Phase 5+6 — run the query API
uvicorn medical-rag.query:app --port 8001
# Test: curl -X POST http://localhost:8001/query -H "Content-Type: application/json" \
#            -d '{"query": "What are the symptoms of diabetes?"}'
```

## Env vars (medical-rag/.env)
```
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
GROQ_API_KEY=...
```
Output: `data/raw/{source}.json` (gitignored — regenerable).
