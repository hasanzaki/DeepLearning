# IIUM Agentic Telephony — Vector DB Project Plan

## Overview

Build an agentic telephony system for IIUM using VAPI as the voice/telephony layer,
backed by a vector database of IIUM academic documents (Google Drive corpus).

## Context

- **Institution**: IIUM (International Islamic University Malaysia)
- **Telephony**: VAPI API (cloud-based, no self-hosted infra needed)
- **Document corpus**: Hundreds of files from Google Drive
- **Malay content**: Minimal (mostly English)
- **Repo**: Temporary home in `iium-vectordb/` subfolder; migrate to `iium-acad/telephony` later

---

## Architecture

```
Google Drive
     │
     ▼
[1] Ingestion         ← download, parse, chunk docs
     │
     ▼
[2] Embeddings        ← embed chunks (OpenAI / local model)
     │
     ▼
[3] Vector DB         ← store & index (Chroma / Qdrant / Pinecone)
     │
     ▼
[4] Retrieval API     ← FastAPI query endpoint
     │
     ▼
[5] VAPI Integration  ← tool/function call from VAPI assistant
     │
     ▼
[6] Caller            ← student/staff asking IIUM questions by phone
```

---

## Phases

### Phase 1 — Ingestion
- [ ] Authenticate with Google Drive API (service account)
- [ ] List & download all files from target folder(s)
- [ ] Parse: PDF → text, DOCX → text, plain text pass-through
- [ ] Chunk text (512 tokens, 50-token overlap)
- [ ] Store raw chunks as JSON/JSONL

**Output**: `ingestion/chunks.jsonl`

### Phase 2 — Embeddings
- [ ] Choose embedding model (default: `text-embedding-3-small`)
- [ ] Batch-embed all chunks
- [ ] Cache embeddings (avoid re-embedding on re-runs)

**Output**: `embeddings/embedded_chunks.jsonl`

### Phase 3 — Vector DB
- [ ] Choose DB (default: **Chroma** — local, zero-infra)
- [ ] Ingest embedded chunks with metadata (source, page, date)
- [ ] Persist to disk

**Output**: `vectordb/chroma_db/`

### Phase 4 — Retrieval API
- [ ] FastAPI app with `/query` endpoint
- [ ] Top-k semantic search
- [ ] Return source + excerpt for citation

**Output**: `api/main.py`

### Phase 5 — VAPI Integration
- [ ] Define VAPI assistant config (system prompt, voice, language)
- [ ] Register retrieval API as a VAPI tool/function
- [ ] Test end-to-end: phone call → VAPI → retrieval → spoken answer
- [ ] Handle fallback for out-of-scope questions

**Output**: `api/vapi_config.json`, `docs/vapi_setup.md`

### Phase 6 — Evaluation & Stress Test
- [ ] Define test questions (10–20 IIUM FAQs)
- [ ] Measure retrieval accuracy (hit rate, MRR)
- [ ] Simulate concurrent VAPI calls

**Output**: `evaluation/`, `stress_test/`

---

## Stack

| Layer       | Tool                        |
|-------------|-----------------------------|
| Telephony   | VAPI API                    |
| Voice LLM   | GPT-4o / Claude (via VAPI)  |
| Embeddings  | text-embedding-3-small      |
| Vector DB   | Chroma (local)              |
| API         | FastAPI + uvicorn           |
| Drive sync  | Google Drive API v3         |
| Language    | Python 3.10+                |

---

## File Structure

```
iium-vectordb/
├── plan.md                  ← this file
├── config/
│   ├── settings.py          ← env vars, model names, paths
│   └── .env.example         ← required env vars template
├── ingestion/
│   ├── gdrive_downloader.py ← download from Google Drive
│   ├── parser.py            ← PDF/DOCX/TXT → text
│   └── chunker.py           ← text → chunks
├── embeddings/
│   └── embedder.py          ← embed chunks, cache results
├── vectordb/
│   └── ingest.py            ← load chunks into Chroma
├── api/
│   ├── main.py              ← FastAPI retrieval endpoint
│   └── vapi_config.json     ← VAPI assistant definition
├── evaluation/
│   └── eval.py              ← retrieval accuracy tests
├── stress_test/
│   └── load_test.py         ← concurrent query simulation
└── docs/
    ├── vapi_setup.md        ← VAPI integration guide
    └── gdrive_setup.md      ← Google Drive API setup guide
```

---

## Next Steps (immediate)

1. Set up `config/settings.py` and `.env.example`
2. Build `ingestion/gdrive_downloader.py` (needs GDrive link/folder ID)
3. Build `ingestion/parser.py` + `chunker.py`
4. Build `embeddings/embedder.py`
5. Build `vectordb/ingest.py`
6. Build `api/main.py`
7. Configure VAPI assistant + tool call

---

## Notes

- GDrive folder ID / link: **TBD** (provide when starting ingestion phase)
- Target: working demo call answering IIUM academic questions
- Migration to `iium-acad/telephony` repo: after demo is stable
