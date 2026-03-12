"""Extract structured text from Word (.docx) thesis files using python-docx."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

from .pdf_extractor import ExtractedSection, ExtractedThesis

# Word heading style names
_HEADING_STYLES = {"Heading 1", "Heading 2", "Heading 3", "Title"}

# Built-in style level map
_STYLE_LEVEL = {
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Title": 0,
}


def _para_text(para) -> str:
    return "".join(run.text for run in para.runs).strip()


def _is_heading(para) -> bool:
    return para.style.name in _HEADING_STYLES


class DOCXExtractor:
    """Extract text and structure from a Word thesis document."""

    def extract(self, docx_path: str, author: str = "", title: str = "") -> ExtractedThesis:
        path = Path(docx_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX not found: {docx_path}")

        doc = Document(str(path))
        sections: list[ExtractedSection] = []
        references: list[str] = []
        abstract = ""
        doc_title = title
        in_references = False

        current_heading = "Preamble"
        current_level = 1
        current_paragraphs: list[str] = []

        def _flush():
            nonlocal current_heading, current_level, current_paragraphs
            content = "\n\n".join(p for p in current_paragraphs if p)
            if content:
                sections.append(
                    ExtractedSection(
                        heading=current_heading,
                        level=current_level,
                        content=content,
                    )
                )
            current_paragraphs = []

        for para in doc.paragraphs:
            text = _para_text(para)
            if not text:
                continue

            style_name = para.style.name

            # Capture document title
            if style_name == "Title" and not doc_title:
                doc_title = text
                continue

            # Detect references section
            if re.match(r"^(references|bibliography)$", text, re.IGNORECASE):
                _flush()
                in_references = True
                continue

            if in_references:
                references.append(text)
                continue

            # Detect abstract
            if re.match(r"^abstract$", text, re.IGNORECASE):
                # Next paragraphs until next heading are abstract
                abstract_parts: list[str] = []
                continue

            if _is_heading(para):
                _flush()
                current_heading = text
                current_level = _STYLE_LEVEL.get(style_name, 2)
            else:
                current_paragraphs.append(text)

        _flush()

        # If no title found, try to infer from first heading
        if not doc_title and sections:
            doc_title = sections[0].heading

        # Extract abstract from "Abstract" section if it was captured as a section
        for sec in sections:
            if re.match(r"^abstract$", sec.heading, re.IGNORECASE):
                abstract = sec.content
                break

        raw_text = "\n\n".join(
            f"{sec.heading}\n{sec.content}" for sec in sections
        )

        return ExtractedThesis(
            source_path=str(path),
            title=doc_title or "Untitled Thesis",
            author=author,
            abstract=abstract,
            sections=sections,
            references=references,
            raw_text=raw_text,
        )
