"""
Split ParsedDocuments into overlapping text chunks suitable for embedding.

Each Chunk carries metadata needed for retrieval citation:
  - chunk_id: deterministic hash of (source, page, offset)
  - source: original file path
  - page: page number within the source document
  - chunk_index: position of this chunk within the page
  - text: the chunk content
  - char_offset: character offset of this chunk within the page text

Chunking strategy: character-based with token-aware sizing.
Default: ~512 tokens ≈ 2048 characters, 50-token overlap ≈ 200 characters.
"""
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)

# Rough chars-per-token ratio for English text
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    chunk_id: str
    source: str
    filename: str
    page: int
    chunk_index: int
    char_offset: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[tuple[int, str]]:
    """
    Split text into (char_offset, chunk_text) pairs.
    Tries to break at sentence/paragraph boundaries when possible.
    """
    if len(text) <= chunk_chars:
        return [(0, text)]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars

        if end >= len(text):
            # Last chunk — take the rest
            chunks.append((start, text[start:]))
            break

        # Try to break at a paragraph boundary
        break_at = _find_break(text, end, window=200)
        chunk_text = text[start:break_at].strip()
        if chunk_text:
            chunks.append((start, chunk_text))

        start = break_at - overlap_chars
        if start <= chunks[-1][0]:
            # Avoid infinite loop if we can't advance
            start = break_at

    return chunks


def _find_break(text: str, pos: int, window: int) -> int:
    """
    Look backwards from pos within window characters for a good break point.
    Priority: double newline > single newline > period > space.
    """
    segment = text[max(0, pos - window): pos]
    for sep in ["\n\n", "\n", ". ", " "]:
        idx = segment.rfind(sep)
        if idx != -1:
            return max(0, pos - window) + idx + len(sep)
    return pos  # Hard break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    doc: ParsedDocument,
    chunk_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    chunk_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    filename = Path(doc.source).name
    chunks = []

    for page_num, page_text in doc.pages:
        splits = _split_text(page_text, chunk_chars, overlap_chars)
        for i, (offset, text) in enumerate(splits):
            chunk_id = _make_id(doc.source, page_num, offset)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source=doc.source,
                filename=filename,
                page=page_num,
                chunk_index=i,
                char_offset=offset,
                text=text,
            ))

    logger.debug("%s → %d chunks (across %d pages)", filename, len(chunks), doc.page_count)
    return chunks


def chunk_documents(
    docs: list[ParsedDocument],
    chunk_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_tokens, overlap_tokens))
    logger.info(
        "Chunked %d documents → %d total chunks (size=%d tok, overlap=%d tok)",
        len(docs), len(all_chunks), chunk_tokens, overlap_tokens,
    )
    return all_chunks


def _make_id(source: str, page: int, offset: int) -> str:
    key = f"{source}|p{page}|o{offset}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), output_path)


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk(**json.loads(line)))
    logger.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from ingestion.parser import parse_directory

    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ingestion/raw_docs")
    out_path = Path("ingestion/chunks.jsonl")

    docs = parse_directory(raw_dir)
    chunks = chunk_documents(docs)
    save_chunks(chunks, out_path)
    print(f"\n{len(chunks)} chunks saved to {out_path}")
