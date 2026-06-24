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
2. Preprocessing — chunk (200–300 tok), dedup (MinHash), hard negatives (BM25)
3. Embeddings — encode corpus with the biomedical model
4. Vector DB — Supabase pgvector table `medical_chunks(vector(768))` + metadata filters
5. Retrieval — semantic + metadata filter (+ reranker later)
6. Generation — grounded, cited, disclaimered (Groq, configurable)
7. Monitoring/eval — deferred to post-MVP

## Phase 1 sources (free / legitimate)
| Source | Type | Records |
|---|---|---|
| OpenFDA drug labels | `drug` | ~450 |
| MedQuAD (NIH/CDC/NLM) | `qa_pair` | ~16,400 |
| Human Phenotype Ontology | `symptom` | ~17,300 |
| Curated sentiment→severity | `sentiment_map` | 30 |

Deferred (redundant with MedQuAD for MVP): MedlinePlus, Gray's Anatomy, UMLS,
MIMIC-III (credentialed), DrugBank-full (licensed).

## Run
```bash
pip install -r requirements.txt
python collectors/openfda_drugs.py --limit 300
python collectors/medquad_qa.py
python collectors/hpo_symptoms.py
python collectors/sentiment_severity.py
```
Output: `data/raw/{source}.json` (gitignored — regenerable).
