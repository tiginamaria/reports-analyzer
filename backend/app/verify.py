"""Verbatim check: is a quoted passage really present in the source text?

Exact match after normalisation is "verified". A fuzzy match above the threshold is
"fuzzy" (shown with a warning). Anything else is "failed".
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from sqlalchemy.orm import Session

from app.models import Chunk, Page
from app.textutil import normalize

Status = Literal["verified", "fuzzy", "failed"]

MIN_QUOTE_CHARS = 3


@dataclass
class VerifyResult:
    status: Status
    score: float


def verify_quote(quote: str, sources: list[str], fuzzy_threshold: float = 0.92) -> VerifyResult:
    """Check the quote against each source; the first exact match wins, otherwise the best
    fuzzy score decides."""
    needle = normalize(quote)
    if len(needle) < MIN_QUOTE_CHARS:
        return VerifyResult("failed", 0.0)
    best = 0.0
    for source in sources:
        haystack = normalize(source)
        if needle in haystack:
            return VerifyResult("verified", 1.0)
        best = max(best, _fuzzy_ratio(needle, haystack))
    if best >= fuzzy_threshold:
        return VerifyResult("fuzzy", best)
    return VerifyResult("failed", best)


def verify_against_chunk(db: Session, chunk: Chunk, quote: str, fuzzy_threshold: float) -> Status:
    """Verify a quote cited from a chunk against the stored text of the chunk's page, the
    source of truth, and against the chunk itself: a table chunk is text clipped to the table
    region, which can read differently from the page's block text."""
    page = (
        db.query(Page)
        .filter(Page.report_id == chunk.report_id, Page.page_no == chunk.page)
        .one_or_none()
    )
    sources = [page.text] if page else []
    return verify_quote(quote, [*sources, chunk.content], fuzzy_threshold).status


def _fuzzy_ratio(needle: str, haystack: str) -> float:
    """Best similarity between the needle and any window of the haystack of similar length."""
    if not haystack:
        return 0.0
    if len(needle) >= len(haystack):
        return SequenceMatcher(None, needle, haystack, autojunk=False).ratio()
    # Anchor on the first word to avoid scanning every offset.
    first_word = needle.split(" ", 1)[0]
    window = len(needle) + 4
    best = 0.0
    start = 0
    while (idx := haystack.find(first_word, start)) >= 0:
        candidate = haystack[idx : idx + window]
        best = max(best, SequenceMatcher(None, needle, candidate, autojunk=False).ratio())
        if best > 0.99:
            break
        start = idx + 1
    return best
