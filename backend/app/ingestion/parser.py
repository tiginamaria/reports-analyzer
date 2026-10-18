"""PDF parsing with PyMuPDF: the text of every page, plus the tables found on it.

Page text is kept verbatim (lightly cleaned) because it is the source of truth for quote
verification. For tables only the detected boundary is used and the raw text inside it is
stored, because PyMuPDF's cell extraction often merges columns of borderless tables.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.textutil import clean_page_text

PROGRESS_EVERY_PAGES = 10
TEXT_BLOCK = 0  # PyMuPDF block type; 1 is an image

# A detected table is kept only if it has at least this many rows and numeric cells; layout
# grids and navigation bars fail the numeric test. Changing these changes what gets indexed,
# so already ingested reports must be re-ingested.
TABLE_MIN_ROWS = 2
TABLE_MIN_NUMERIC_CELLS = 3


@dataclass
class Table:
    page: int  # 1-based
    text: str  # raw text inside the table boundary, as PyMuPDF returns it


@dataclass
class ParsedDocument:
    pages: list[str]  # full page text in reading order, index 0 == page 1
    bodies: list[str]  # the same pages with table regions removed; what gets chunked
    tables: list[Table] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


ProgressFn = Callable[[int, int], None]  # (pages done, total pages)


def parse_pdf(path: Path, progress: ProgressFn | None = None) -> ParsedDocument:
    doc = ParsedDocument(pages=[], bodies=[])
    with pymupdf.open(path) as pdf:
        for index in range(pdf.page_count):
            page = pdf[index]
            page_no = index + 1
            rects = _table_rects(page, TABLE_MIN_ROWS, TABLE_MIN_NUMERIC_CELLS)
            doc.pages.append(_page_text(page))
            doc.bodies.append(_page_text(page, exclude=rects))
            doc.tables.extend(Table(page_no, text) for text in _table_texts(page, rects))
            if progress and (page_no % PROGRESS_EVERY_PAGES == 0 or page_no == pdf.page_count):
                progress(page_no, pdf.page_count)
    return doc


def _page_text(page: pymupdf.Page, exclude: Sequence[pymupdf.Rect] = ()) -> str:
    """Text blocks in reading order (top to bottom, left to right), skipping blocks that lie
    inside `exclude` regions. Sorting whole blocks keeps each paragraph intact; sorting
    lines would interleave the columns of a two-column page."""
    paragraphs = []
    for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks", sort=True):
        if block_type != TEXT_BLOCK:
            continue
        if any(rect.intersects(pymupdf.Rect(x0, y0, x1, y1)) for rect in exclude):
            continue
        paragraphs.append(text)
    return clean_page_text("\n\n".join(paragraphs))


def _table_rects(page: pymupdf.Page, min_rows: int, min_numeric_cells: int) -> list[pymupdf.Rect]:
    """Boundaries of real tables. Layout grids and navigation bars also come back as tables,
    so a table must have enough rows and numeric cells to be kept."""
    rects = []
    for table in page.find_tables().tables:
        rows = table.extract()
        numeric_cells = sum(1 for row in rows for cell in row if cell and re.search(r"\d", cell))
        if len(rows) >= min_rows and numeric_cells >= min_numeric_cells:
            rects.append(pymupdf.Rect(table.bbox))
    return rects


def _table_texts(page: pymupdf.Page, rects: Sequence[pymupdf.Rect]) -> list[str]:
    """The raw text inside each table boundary, as PyMuPDF returns it."""
    texts = (clean_page_text(page.get_text("text", clip=rect)) for rect in rects)
    return [text for text in texts if text]
