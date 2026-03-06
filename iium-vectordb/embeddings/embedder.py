"""
Phase 2: Embed chunks using OpenAI text-embedding-3-small.

Features:
  - Batch embedding (configurable batch size)
  - Cache: skips already-embedded chunk_ids on re-run
  - Output: embeddings/embedded_chunks.jsonl

Usage:
    python embeddings/embedder.py
    python embeddings/embedder.py --input ingestion/chunks.jsonl --output embeddings/embedded_chunks.jsonl
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from ingestion.chunker import load_chunks, Chunk

logger = logging.getLogger(__name__)

DEFAULT_INPUT  = Path(__file__).resolve().parents[1] / "ingestion" / "chunks.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "embedded_chunks.jsonl"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cached_ids(output_path: Path) -> set[str]:
    """Return chunk_ids already present in the output file."""
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["chunk_id"])
                except (KeyError, json.JSONDecodeError):
                    pass
    logger.info("Cache: %d already-embedded chunks found in %s", len(ids), output_path)
    return ids


# ---------------------------------------------------------------------------
# OpenAI embedding
# ---------------------------------------------------------------------------

def _embed_batch(client, texts: list[str], model: str) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts. Retries once on rate limit."""
    for attempt in range(3):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as exc:
            if attempt < 2:
                wait = 2 ** attempt * 5   # 5s, 10s
                logger.warning("Embedding error (attempt %d/3): %s — retrying in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Main embed function
# ---------------------------------------------------------------------------

def embed(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> Path:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to config/.env")

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load chunks
    chunks: list[Chunk] = load_chunks(input_path)
    if not chunks:
        raise ValueError(f"No chunks found in {input_path}")

    # Filter out already-embedded chunks (cache)
    cached_ids = _load_cached_ids(output_path)
    pending = [c for c in chunks if c.chunk_id not in cached_ids]
    logger.info(
        "Chunks total: %d | cached: %d | to embed: %d",
        len(chunks), len(cached_ids), len(pending),
    )

    if not pending:
        logger.info("All chunks already embedded. Nothing to do.")
        return output_path

    # Embed in batches, append to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedded_count = 0

    with output_path.open("a", encoding="utf-8") as out_f:
        for i in range(0, len(pending), batch_size):
            batch: list[Chunk] = pending[i: i + batch_size]
            texts = [c.text for c in batch]

            logger.info(
                "Embedding batch %d/%d (chunks %d–%d)...",
                i // batch_size + 1,
                (len(pending) + batch_size - 1) // batch_size,
                i + 1,
                min(i + batch_size, len(pending)),
            )

            embeddings = _embed_batch(client, texts, model)

            for chunk, embedding in zip(batch, embeddings):
                record = chunk.to_dict()
                record["embedding"] = embedding
                record["model"] = model
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            embedded_count += len(batch)
            out_f.flush()

    logger.info("\n✓ Embedding complete")
    logger.info("  Newly embedded : %d", embedded_count)
    logger.info("  Total in file  : %d", embedded_count + len(cached_ids))
    logger.info("  Output         : %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IIUM VectorDB — Phase 2 Embeddings")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT,  help="Path to chunks.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to embedded_chunks.jsonl")
    parser.add_argument("--model",  default=EMBEDDING_MODEL,           help="OpenAI embedding model")
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE, help="Chunks per API call")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    embed(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
