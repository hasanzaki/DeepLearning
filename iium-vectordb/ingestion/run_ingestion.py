"""
Phase 1 pipeline: Download → Parse → Chunk → Save

Run from repo root:
    python ingestion/run_ingestion.py

Flags:
    --skip-download   Re-use already-downloaded files in raw_docs/
    --folder-id ID    Override GDRIVE_FOLDER_ID from env
    --output PATH     Override default chunks.jsonl path
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import GDRIVE_FOLDER_ID
from ingestion.gdrive_downloader import download
from ingestion.parser import parse_directory
from ingestion.chunker import chunk_documents, save_chunks

RAW_DOCS_DIR = Path(__file__).resolve().parent / "raw_docs"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "chunks.jsonl"

logger = logging.getLogger(__name__)


def run(
    skip_download: bool = False,
    folder_id: str | None = None,
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    folder_id = folder_id or GDRIVE_FOLDER_ID

    # ── Step 1: Download ──────────────────────────────────────────────────
    if skip_download:
        logger.info("Skipping download. Using existing files in %s", RAW_DOCS_DIR)
    else:
        logger.info("=== Step 1/3: Downloading from Google Drive ===")
        files = download(folder_id=folder_id, dest_dir=RAW_DOCS_DIR)
        logger.info("Download complete: %d files", len(files))

    # ── Step 2: Parse ─────────────────────────────────────────────────────
    logger.info("=== Step 2/3: Parsing documents ===")
    docs = parse_directory(RAW_DOCS_DIR)
    if not docs:
        logger.error("No documents parsed. Check raw_docs/ directory.")
        sys.exit(1)

    # ── Step 3: Chunk ─────────────────────────────────────────────────────
    logger.info("=== Step 3/3: Chunking ===")
    chunks = chunk_documents(docs)

    # ── Save ──────────────────────────────────────────────────────────────
    save_chunks(chunks, output)

    logger.info("\n✓ Ingestion complete")
    logger.info("  Documents : %d", len(docs))
    logger.info("  Chunks    : %d", len(chunks))
    logger.info("  Output    : %s", output)

    return output


def main():
    parser = argparse.ArgumentParser(description="IIUM VectorDB — Phase 1 Ingestion")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Drive download, use existing raw_docs/")
    parser.add_argument("--folder-id", default=None,
                        help="Override Google Drive folder ID")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL path for chunks")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    run(
        skip_download=args.skip_download,
        folder_id=args.folder_id,
        output=args.output,
    )


if __name__ == "__main__":
    main()
