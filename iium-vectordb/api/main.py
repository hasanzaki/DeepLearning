"""
Phase 4: FastAPI retrieval API for IIUM Vector DB.

Endpoints:
  POST /query   — semantic search, returns top-k chunks
  GET  /health  — liveness check

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

VAPI will call POST /query with a JSON body:
  { "query": "What is the deadline for course registration?" }
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    API_HOST,
    API_PORT,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IIUM VectorDB API",
    description="Semantic retrieval over IIUM academic documents",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Startup: load Chroma collection once
# ---------------------------------------------------------------------------

_collection = None
_openai_client = None


def _get_collection():
    global _collection
    if _collection is None:
        try:
            import chromadb
        except ImportError:
            raise RuntimeError("Install chromadb: pip install chromadb")

        persist_path = Path(CHROMA_PERSIST_DIR)
        if not persist_path.exists():
            raise RuntimeError(
                f"ChromaDB not found at {persist_path}. Run vectordb/ingest.py first."
            )

        client = chromadb.PersistentClient(path=str(persist_path))
        _collection = client.get_collection(name=COLLECTION_NAME)
        logger.info(
            "Loaded Chroma collection '%s' with %d vectors", COLLECTION_NAME, _collection.count()
        )
    return _collection


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to config/.env")
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Install openai: pip install openai")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    filename: str
    page: int
    score: float          # cosine similarity (higher = more relevant)


class QueryResponse(BaseModel):
    query: str
    results: list[ChunkResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    collection = _get_collection()
    return {"status": "ok", "vectors": collection.count()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    # Embed the query
    try:
        client = _get_openai_client()
        response = client.embeddings.create(
            input=[request.query],
            model=EMBEDDING_MODEL,
        )
        query_embedding = response.data[0].embedding
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Embedding error: {exc}")

    # Query Chroma
    try:
        collection = _get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("Chroma query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Vector DB error: {exc}")

    # Build response (distance → similarity: cosine distance = 1 - similarity)
    chunks = []
    ids       = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
        chunks.append(ChunkResult(
            chunk_id = chunk_id,
            text     = text,
            filename = meta.get("filename", ""),
            page     = meta.get("page", 0),
            score    = round(1 - dist, 4),
        ))

    return QueryResponse(query=request.query, results=chunks)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
