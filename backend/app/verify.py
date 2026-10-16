"""Verbatim check: is a quoted passage really present in the source text?

Exact match after normalisation is "verified". A fuzzy match above the threshold is
"fuzzy" (shown with a warning). Anything else is "failed".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz
from sqlalchemy.orm import Session

from app.models import Chunk, Page
from app.textutil import normalize

Status = Literal["verified", "fuzzy", "failed"]

MIN_QUOTE_CHARS = 3
# Similarity at or above which a non-exact quote counts as "fuzzy" rather than "failed".
FUZZY_THRESHOLD = 0.92


@dataclass
class VerifyResult:
    status: Status
    score: float


def verify_quote(
    quote: str, sources: list[str], fuzzy_threshold: float = FUZZY_THRESHOLD
) -> VerifyResult:
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


def verify_against_chunk(db: Session, chunk: Chunk, quote: str) -> Status:
    """Verify a quote cited from a chunk against the stored text of the chunk's page, the
    source of truth, and against the chunk itself: a table chunk is text clipped to the table
    region, which can read differently from the page's block text."""
    page = (
        db.query(Page)
        .filter(Page.report_id == chunk.report_id, Page.page_no == chunk.page)
        .one_or_none()
    )
    sources = [page.text] if page else []
    return verify_quote(quote, [*sources, chunk.content]).status


def _fuzzy_ratio(needle: str, haystack: str) -> float:
    """Best similarity (0..1) between the needle and any equally long window of the haystack.
    A needle longer than the haystack is compared whole: `partial_ratio` would otherwise slide
    the haystack over the needle and score a short source inside a long invented quote as 1."""
    if not haystack:
        return 0.0
    if len(needle) >= len(haystack):
        return fuzz.ratio(needle, haystack) / 100
    return fuzz.partial_ratio(needle, haystack) / 100
