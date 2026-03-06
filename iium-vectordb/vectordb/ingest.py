"""
Phase 3: Load embedded chunks into ChromaDB.

Features:
  - Idempotent: skips chunk_ids already in the collection
  - Persists to disk (no server needed)
  - Stores metadata: source filename, page, chunk_index

Usage:
    python vectordb/ingest.py
    python vectordb/ingest.py --input embeddings/embedded_chunks.jsonl
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import CHROMA_PERSIST_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "embeddings" / "embedded_chunks.jsonl"


# ---------------------------------------------------------------------------
# Load embedded chunks from JSONL
# ---------------------------------------------------------------------------

def _load_embedded_chunks(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d embedded chunks from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Ingest into Chroma
# ---------------------------------------------------------------------------

def ingest(
    input_path: Path = DEFAULT_INPUT,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    try:
        import chromadb
    except ImportError:
        raise ImportError("Install chromadb: pip install chromadb")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Embedded chunks not found at {input_path}. Run embeddings/embedder.py first."
        )

    records = _load_embedded_chunks(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    # Connect to / create persistent Chroma collection
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Find which chunk_ids are already in the collection
    existing_ids: set[str] = set()
    if collection.count() > 0:
        existing = collection.get(include=[])   # IDs only
        existing_ids = set(existing["ids"])
    logger.info("Collection '%s': %d existing vectors", collection_name, len(existing_ids))

    # Filter to only new records
    new_records = [r for r in records if r["chunk_id"] not in existing_ids]
    logger.info("New chunks to ingest: %d", len(new_records))

    if not new_records:
        logger.info("All chunks already in collection. Nothing to do.")
        return

    # Batch upsert (Chroma recommends ≤5000 per call)
    batch_size = 500
    for i in range(0, len(new_records), batch_size):
        batch = new_records[i: i + batch_size]
        collection.add(
            ids        = [r["chunk_id"]    for r in batch],
            embeddings = [r["embedding"]   for r in batch],
            documents  = [r["text"]        for r in batch],
            metadatas  = [
                {
                    "source":      r["source"],
                    "filename":    r["filename"],
                    "page":        r["page"],
                    "chunk_index": r["chunk_index"],
                    "char_offset": r["char_offset"],
                    "model":       r.get("model", ""),
                }
                for r in batch
            ],
        )
        logger.info(
            "Ingested batch %d/%d (%d chunks)",
            i // batch_size + 1,
            (len(new_records) + batch_size - 1) // batch_size,
            len(batch),
        )

    logger.info("\n✓ Vector DB ingest complete")
    logger.info("  New vectors added : %d", len(new_records))
    logger.info("  Total in collection: %d", collection.count())
    logger.info("  Persisted to      : %s", persist_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IIUM VectorDB — Phase 3 Ingest")
    parser.add_argument("--input",      type=Path, default=DEFAULT_INPUT,      help="Path to embedded_chunks.jsonl")
    parser.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR,           help="ChromaDB persist directory")
    parser.add_argument("--collection",  default=COLLECTION_NAME,              help="Chroma collection name")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ingest(
        input_path=args.input,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
