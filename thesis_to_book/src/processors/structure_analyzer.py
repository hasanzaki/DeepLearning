"""Analyze and normalize thesis structure into a canonical book chapter layout."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..extractors.pdf_extractor import ExtractedSection, ExtractedThesis


@dataclass
class BookChapter:
    chapter_number: int
    chapter_title: str
    author: str
    original_title: str
    abstract: str
    introduction: str
    body_sections: list[tuple[str, str]]  # (heading, content) pairs
    conclusion: str
    references: list[str]


# Section headings that typically belong to the introduction
_INTRO_KEYWORDS = re.compile(
    r"^(introduction|background|motivation|overview|preface|foreword)",
    re.IGNORECASE,
)

# Section headings that typically form the conclusion
_CONCLUSION_KEYWORDS = re.compile(
    r"^(conclusion|summary|discussion|future\s+work|limitations|closing)",
    re.IGNORECASE,
)

# Sections to skip entirely (admin content)
_SKIP_KEYWORDS = re.compile(
    r"^(table of contents|list of (figures|tables|abbreviations)|acknowledgements?|dedication|declaration|approval)",
    re.IGNORECASE,
)


def _is_intro(heading: str) -> bool:
    return bool(_INTRO_KEYWORDS.search(heading))


def _is_conclusion(heading: str) -> bool:
    return bool(_CONCLUSION_KEYWORDS.search(heading))


def _should_skip(heading: str) -> bool:
    return bool(_SKIP_KEYWORDS.search(heading))


class StructureAnalyzer:
    """Convert an ExtractedThesis into a structured BookChapter."""

    def analyze(
        self,
        thesis: ExtractedThesis,
        chapter_number: int,
        chapter_title: str,
    ) -> BookChapter:
        intro_parts: list[str] = []
        conclusion_parts: list[str] = []
        body_sections: list[tuple[str, str]] = []

        for sec in thesis.sections:
            if _should_skip(sec.heading):
                continue
            if _is_intro(sec.heading):
                intro_parts.append(sec.content)
            elif _is_conclusion(sec.heading):
                conclusion_parts.append(sec.content)
            else:
                body_sections.append((sec.heading, sec.content))

        return BookChapter(
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            author=thesis.author,
            original_title=thesis.title,
            abstract=thesis.abstract,
            introduction="\n\n".join(intro_parts),
            body_sections=body_sections,
            conclusion="\n\n".join(conclusion_parts),
            references=thesis.references,
        )
