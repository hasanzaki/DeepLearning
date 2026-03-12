"""Use the Claude API to rewrite thesis sections into cohesive book prose."""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from .structure_analyzer import BookChapter

# Token budget per section to avoid exceeding context limits
_MAX_INPUT_CHARS = 12_000


@dataclass
class RewrittenChapter:
    chapter_number: int
    chapter_title: str
    author: str
    abstract: str
    introduction: str
    body_sections: list[tuple[str, str]]  # (heading, rewritten content)
    conclusion: str
    references: list[str]
    index_terms: list[str] = field(default_factory=list)


class AIRewriter:
    """Rewrite extracted thesis chapters using Claude."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        rewrite_level: str = "medium",
        style_guide: str = "",
    ):
        self._client = anthropic.Anthropic()
        self._model = model
        self._rewrite_level = rewrite_level
        self._style_guide = style_guide

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite_chapter(self, chapter: BookChapter) -> RewrittenChapter:
        abstract = self._rewrite_abstract(chapter)
        introduction = self._rewrite_section(
            "Introduction", chapter.introduction, chapter
        )
        body_sections = [
            (heading, self._rewrite_section(heading, content, chapter))
            for heading, content in chapter.body_sections
        ]
        conclusion = self._rewrite_section(
            "Conclusion", chapter.conclusion, chapter
        )
        index_terms = self._extract_index_terms(chapter)

        return RewrittenChapter(
            chapter_number=chapter.chapter_number,
            chapter_title=chapter.chapter_title,
            author=chapter.author,
            abstract=abstract,
            introduction=introduction,
            body_sections=body_sections,
            conclusion=conclusion,
            references=chapter.references,
            index_terms=index_terms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        level_instructions = {
            "light": (
                "Make minimal edits: fix grammar and spelling, improve sentence flow, "
                "but preserve the author's original voice and structure."
            ),
            "medium": (
                "Restructure paragraphs for better academic clarity, improve transitions "
                "between ideas, and ensure consistent formal tone. Preserve technical "
                "accuracy and all key arguments."
            ),
            "heavy": (
                "Fully rewrite the content into polished book-quality academic prose. "
                "Improve argument structure, eliminate redundancy, and ensure the section "
                "reads as a unified contribution to the book."
            ),
        }
        instruction = level_instructions.get(self._rewrite_level, level_instructions["medium"])
        style = f"\n\nStyle guide: {self._style_guide}" if self._style_guide else ""
        return (
            f"You are an expert academic editor preparing a scholarly book. "
            f"Your task: {instruction}{style}\n\n"
            "Output ONLY the rewritten text. Do not include headings, preamble, "
            "or commentary. Preserve all technical terminology and citations."
        )

    def _rewrite_section(
        self, heading: str, content: str, chapter: BookChapter
    ) -> str:
        if not content.strip():
            return ""

        truncated = content[:_MAX_INPUT_CHARS]
        prompt = (
            f"Chapter {chapter.chapter_number}: \"{chapter.chapter_title}\" "
            f"(adapted from thesis by {chapter.author})\n\n"
            f"Section: {heading}\n\n"
            f"{truncated}"
        )

        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def _rewrite_abstract(self, chapter: BookChapter) -> str:
        if not chapter.abstract.strip():
            return ""

        prompt = (
            f"Rewrite the following thesis abstract as a chapter abstract for a book. "
            f"Chapter title: \"{chapter.chapter_title}\". "
            f"Author: {chapter.author}.\n\n"
            f"{chapter.abstract[:_MAX_INPUT_CHARS]}"
        )
        message = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=(
                "You are an academic editor. Rewrite the abstract to fit a book chapter "
                "context. Output only the rewritten abstract text."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def _extract_index_terms(self, chapter: BookChapter) -> list[str]:
        """Ask Claude to extract key index terms from the chapter body."""
        sample_text = "\n".join(
            content for _, content in chapter.body_sections[:3]
        )[:4000]

        if not sample_text.strip():
            return []

        prompt = (
            f"From the following text of chapter \"{chapter.chapter_title}\", "
            "extract 10-20 important technical terms suitable for a book index. "
            "Return one term per line, no bullets or numbering.\n\n"
            f"{sample_text}"
        )
        message = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system="You are an academic indexer. Output only the index terms, one per line.",
            messages=[{"role": "user", "content": prompt}],
        )
        terms = message.content[0].text.strip().splitlines()
        return [t.strip() for t in terms if t.strip()]
