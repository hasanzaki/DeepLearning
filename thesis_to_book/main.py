#!/usr/bin/env python3
"""
thesis_to_book — Convert academic theses into a formatted Word book.

Usage:
    python main.py --config examples/sample_config.yaml
    python main.py --config my_book.yaml --output output/my_book.docx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from src.extractors import DOCXExtractor, PDFExtractor
from src.generators import BookGenerator
from src.processors import AIRewriter, StructureAnalyzer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_extractor(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PDFExtractor()
    if ext in (".docx", ".doc"):
        return DOCXExtractor()
    raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .docx")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert academic theses into a formatted Word book."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file (see examples/sample_config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the output path from the config file",
    )
    parser.add_argument(
        "--no-toc",
        action="store_true",
        help="Skip generating the Table of Contents",
    )
    parser.add_argument(
        "--no-bibliography",
        action="store_true",
        help="Skip generating the consolidated bibliography",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip generating the index",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and structure theses but skip AI rewriting (for testing)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: config file not found: {cfg_path}", file=sys.stderr)
        return 1

    cfg = load_config(str(cfg_path))
    output_cfg = cfg.get("output", {})
    ai_cfg = cfg.get("ai", {})
    book_cfg = cfg.get("book", {})
    theses_cfg = cfg.get("theses", [])

    output_path = args.output or output_cfg.get("path", "output/academic_book.docx")
    include_toc = (not args.no_toc) and output_cfg.get("include_toc", True)
    include_bibliography = (not args.no_bibliography) and output_cfg.get("include_bibliography", True)
    include_index = (not args.no_index) and output_cfg.get("include_index", True)

    if not theses_cfg:
        print("Error: no theses defined in config file.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Initialise pipeline components
    # ------------------------------------------------------------------ #
    analyzer = StructureAnalyzer()
    rewriter = AIRewriter(
        model=ai_cfg.get("model", "claude-opus-4-6"),
        rewrite_level=ai_cfg.get("rewrite_level", "medium"),
        style_guide=ai_cfg.get("style_guide", ""),
    )
    generator = BookGenerator(book_config=book_cfg)

    # ------------------------------------------------------------------ #
    # Process each thesis
    # ------------------------------------------------------------------ #
    rewritten_chapters = []
    for thesis_cfg in tqdm(theses_cfg, desc="Processing theses"):
        thesis_path = thesis_cfg.get("path", "")
        author = thesis_cfg.get("author", "Unknown Author")
        chapter_num = thesis_cfg.get("chapter_number", len(rewritten_chapters) + 1)
        chapter_title = thesis_cfg.get("chapter_title", f"Chapter {chapter_num}")
        original_title = thesis_cfg.get("original_title", "")

        print(f"\n[{chapter_num}] Extracting: {thesis_path}")
        try:
            extractor = get_extractor(thesis_path)
            thesis = extractor.extract(thesis_path, author=author, title=original_title)
        except FileNotFoundError as e:
            print(f"  Warning: {e} — skipping.", file=sys.stderr)
            continue
        except Exception as e:
            print(f"  Error extracting {thesis_path}: {e} — skipping.", file=sys.stderr)
            continue

        print(f"  Structuring chapter…")
        chapter = analyzer.analyze(thesis, chapter_num, chapter_title)

        if args.dry_run:
            print(f"  [dry-run] Skipping AI rewrite.")
            from src.processors.ai_rewriter import RewrittenChapter
            rewritten = RewrittenChapter(
                chapter_number=chapter.chapter_number,
                chapter_title=chapter.chapter_title,
                author=chapter.author,
                abstract=chapter.abstract,
                introduction=chapter.introduction,
                body_sections=chapter.body_sections,
                conclusion=chapter.conclusion,
                references=chapter.references,
            )
        else:
            print(f"  Rewriting with Claude ({ai_cfg.get('model', 'claude-opus-4-6')})…")
            rewritten = rewriter.rewrite_chapter(chapter)

        rewritten_chapters.append(rewritten)

    if not rewritten_chapters:
        print("Error: no chapters were processed.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Generate the book
    # ------------------------------------------------------------------ #
    print(f"\nGenerating book → {output_path}")
    saved_path = generator.generate(
        chapters=rewritten_chapters,
        output_path=output_path,
        include_toc=include_toc,
        include_bibliography=include_bibliography,
        include_index=include_index,
    )
    print(f"Done! Book saved to: {saved_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
