"""Extract structured text from PDF thesis files using pdfplumber."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


@dataclass
class ExtractedSection:
    heading: str
    level: int  # 1 = chapter, 2 = section, 3 = subsection
    content: str


@dataclass
class ExtractedThesis:
    source_path: str
    title: str
    author: str
    abstract: str
    sections: list[ExtractedSection] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    raw_text: str = ""


# Patterns that commonly signal headings in thesis PDFs
_HEADING_PATTERNS = [
    re.compile(r"^(CHAPTER\s+\d+[\.:]\s*.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(\d+\.\s+[A-Z][^\n]{3,80})$", re.MULTILINE),
    re.compile(r"^(\d+\.\d+\s+[A-Z][^\n]{3,80})$", re.MULTILINE),
    re.compile(r"^(\d+\.\d+\.\d+\s+[A-Z][^\n]{3,80})$", re.MULTILINE),
]

_ABSTRACT_PATTERN = re.compile(
    r"(?:abstract|ABSTRACT)\s*\n(.*?)(?=\n(?:chapter|introduction|\d+\.|keywords|table of contents))",
    re.IGNORECASE | re.DOTALL,
)

_REFERENCES_PATTERN = re.compile(
    r"(?:references|bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n(.*?)$",
    re.IGNORECASE | re.DOTALL,
)


def _detect_heading_level(line: str) -> int:
    if re.match(r"^CHAPTER\s+\d+", line, re.IGNORECASE):
        return 1
    if re.match(r"^\d+\.\s+[A-Z]", line):
        return 1
    if re.match(r"^\d+\.\d+\s+[A-Z]", line):
        return 2
    if re.match(r"^\d+\.\d+\.\d+\s+[A-Z]", line):
        return 3
    return 2


def _split_into_sections(text: str) -> list[ExtractedSection]:
    """Split raw text into sections based on heading detection."""
    lines = text.splitlines()
    sections: list[ExtractedSection] = []
    current_heading = "Preamble"
    current_level = 1
    current_lines: list[str] = []

    def _flush():
        nonlocal current_heading, current_level, current_lines
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                ExtractedSection(
                    heading=current_heading,
                    level=current_level,
                    content=content,
                )
            )
        current_lines = []

    for line in lines:
        stripped = line.strip()
        is_heading = any(pat.match(stripped) for pat in _HEADING_PATTERNS)
        if is_heading and len(stripped) > 4:
            _flush()
            current_heading = stripped
            current_level = _detect_heading_level(stripped)
        else:
            current_lines.append(line)

    _flush()
    return sections


class PDFExtractor:
    """Extract text and structure from a PDF thesis."""

    def extract(self, pdf_path: str, author: str = "", title: str = "") -> ExtractedThesis:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages_text: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

        raw_text = "\n".join(pages_text)

        abstract = self._extract_abstract(raw_text)
        references = self._extract_references(raw_text)

        # Remove the references block from the body before sectioning
        body_text = re.split(
            r"\n(?:references|bibliography)\s*\n",
            raw_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        sections = _split_into_sections(body_text)

        inferred_title = title or self._infer_title(pages_text)

        return ExtractedThesis(
            source_path=str(path),
            title=inferred_title,
            author=author,
            abstract=abstract,
            sections=sections,
            references=references,
            raw_text=raw_text,
        )

    def _extract_abstract(self, text: str) -> str:
        match = _ABSTRACT_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        # Fallback: grab up to 800 chars after "abstract"
        idx = text.lower().find("abstract")
        if idx != -1:
            return text[idx + 8 : idx + 808].strip()
        return ""

    def _extract_references(self, text: str) -> list[str]:
        match = _REFERENCES_PATTERN.search(text)
        if not match:
            return []
        ref_block = match.group(1).strip()
        # Split on newlines that look like numbered or bracketed references
        entries = re.split(r"\n(?=\[\d+\]|\d+\.\s)", ref_block)
        return [e.strip() for e in entries if e.strip()]

    def _infer_title(self, pages_text: list[str]) -> str:
        """Use the first non-trivial line of the first page as the title."""
        if not pages_text:
            return "Untitled Thesis"
        for line in pages_text[0].splitlines():
            line = line.strip()
            if len(line) > 10:
                return line
        return "Untitled Thesis"
