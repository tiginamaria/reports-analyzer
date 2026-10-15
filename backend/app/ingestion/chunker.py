"""Fixed-size chunking with overlap.

Each page's body text (tables removed) is cut into windows of `max_words` words,
consecutive windows sharing `overlap_words`, so a sentence cut by a window boundary is whole
in the next one. Each detected table is a chunk of its own, never cut, and is indexed only
once. No document structure is inferred.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from app.ingestion.parser import ParsedDocument
from app.textutil import word_count


@dataclass
class ChunkDraft:
    page: int
    kind: Literal["text", "table"]
    content: str

    @property
    def word_count(self) -> int:
        return word_count(self.content)


def chunk_document(
    doc: ParsedDocument, max_words: int, overlap_words: int, min_words: int
) -> list[ChunkDraft]:
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    tables_by_page: dict[int, list[str]] = defaultdict(list)
    for table in doc.tables:
        tables_by_page[table.page].append(table.text)

    chunks: list[ChunkDraft] = []
    for index, body in enumerate(doc.bodies):
        page = index + 1
        for window in _windows(body.split(), max_words, overlap_words):
            chunks.append(ChunkDraft(page, "text", " ".join(window)))
        for table_text in tables_by_page[page]:
            chunks.append(ChunkDraft(page, "table", table_text))
    return [chunk for chunk in chunks if chunk.word_count >= min_words]


def _windows(words: list[str], size: int, overlap: int) -> Iterator[list[str]]:
    """Consecutive slices of `size` words, each starting `overlap` words before the previous
    one ended. The last slice ends at the last word."""
    step = size - overlap
    for start in range(0, len(words), step):
        yield words[start : start + size]
        if start + size >= len(words):
            return


def contextual_header(company: str, year: int, page: int) -> str:
    """Prepended to a chunk before embedding so the vector carries its context."""
    return f"[{company} | Annual Report {year} | p. {page}]"
