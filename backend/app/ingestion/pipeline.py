"""Ingestion pipeline: parse -> detect metadata -> store pages -> chunk -> embed -> extract.

Runs as a background task with its own database session. Status and progress are written
to the report row after every stage so the UI can poll.
"""

import logging
import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.extraction.extractor import detect_metadata, extract_report
from app.ingestion.chunker import chunk_document, contextual_header
from app.ingestion.parser import ParsedDocument, parse_pdf
from app.llm import get_provider
from app.models import Chunk, Page, Report, ReportStatus
from app.retrieval.embedder import get_embedder

log = logging.getLogger(__name__)

# Progress range (start, end) of each stage, in percent of the whole ingestion.
PROGRESS_RANGES = {
    ReportStatus.PARSING: (0, 30),
    ReportStatus.CHUNKING: (30, 35),
    ReportStatus.EMBEDDING: (35, 80),
    ReportStatus.EXTRACTING: (80, 100),
    ReportStatus.READY: (100, 100),
}

MAX_ERROR_LENGTH = 2000
EMBED_SLICE = 512  # chunks per embedding call; see `_embed`


def ingest_report(report_id: str, path: Path) -> None:
    """Background-task entry point. Any failure is recorded on the report, never raised."""
    with SessionLocal() as db:
        report = db.get(Report, report_id)
        if report is None:
            return
        try:
            _run_stages(db, report, path)
        except Exception as exc:  # any failure is shown in the UI
            db.rollback()
            _mark_failed(db, report_id, exc)


def _run_stages(db: Session, report: Report, path: Path) -> None:
    doc = _parse(db, report, path)
    _detect_metadata(report, doc)

    _set_progress(db, report, ReportStatus.CHUNKING)
    _store_pages(db, report, doc)
    chunks = _chunk(db, report, doc)

    _set_progress(db, report, ReportStatus.EMBEDDING)
    _embed(db, report, chunks)

    _set_progress(db, report, ReportStatus.EXTRACTING)
    extract_report(db, report, get_provider())

    _set_progress(db, report, ReportStatus.READY, fraction=1.0)
    log.info("Ingested %s (%d pages, %d chunks)", report.filename, doc.page_count, len(chunks))


def _mark_failed(db: Session, report_id: str, exc: Exception) -> None:
    report = db.get(Report, report_id)
    if report is None:
        log.info("Report %s was deleted during ingestion; stopping", report_id)
        return
    log.error("Ingestion failed for %s", report_id, exc_info=exc)
    report.status = ReportStatus.FAILED
    report.progress = 0
    report.error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_LENGTH]
    db.commit()


def _set_progress(db: Session, report: Report, stage: ReportStatus, fraction: float = 0.0) -> None:
    """Record the current stage and how far into it (0..1) ingestion is."""
    start, end = PROGRESS_RANGES[stage]
    report.status = stage
    report.progress = int(start + (end - start) * fraction)
    db.commit()


def _parse(db: Session, report: Report, path: Path) -> ParsedDocument:
    _set_progress(db, report, ReportStatus.PARSING)

    def on_progress(pages_done: int, total_pages: int) -> None:
        _set_progress(db, report, ReportStatus.PARSING, pages_done / total_pages)

    return parse_pdf(path, on_progress)


def _detect_metadata(report: Report, doc: ParsedDocument) -> None:
    """Ask the LLM for company, fiscal year and title from the first pages. If that fails
    the report keeps year 0 and gets the file name as its company, so it stays visible."""
    try:
        meta = detect_metadata(get_provider(), doc.pages, report.filename)
    except Exception as exc:  # detection is best effort
        log.warning("Metadata detection failed for %s: %s", report.filename, exc)
        report.company = Path(report.filename).stem
        return
    report.company = meta.company
    report.fiscal_year = meta.fiscal_year
    report.title = meta.title


def _store_pages(db: Session, report: Report, doc: ParsedDocument) -> None:
    """Page texts are the source of truth for verbatim checks and page-level citations."""
    report.page_count = doc.page_count
    db.add_all(
        Page(report_id=report.id, page_no=index + 1, text=text)
        for index, text in enumerate(doc.pages)
    )


def _chunk(db: Session, report: Report, doc: ParsedDocument) -> list[Chunk]:
    drafts = chunk_document(doc)
    prefix = _chunk_id_prefix(report)
    chunks = [
        Chunk(
            id=f"{prefix}-p{draft.page}-c{number}",
            report_id=report.id,
            page=draft.page,
            kind=draft.kind,
            content=draft.content,
            word_count=draft.word_count,
        )
        for number, draft in enumerate(drafts)
    ]
    db.add_all(chunks)
    return chunks


def _chunk_id_prefix(report: Report) -> str:
    """Chunk ids are cited by the LLM, so they are readable (`shell-2025-...`). The report
    id fragment keeps them unique when several PDFs share a company and year."""
    slug = re.sub(r"[^a-z0-9]+", "-", report.company.lower()).strip("-")[:20]
    return f"{slug}-{report.fiscal_year}-{report.id[:6]}"


def _embed(db: Session, report: Report, chunks: list[Chunk], slice_size: int = EMBED_SLICE) -> None:
    """Embed in slices of `slice_size` chunks."""
    embedder = get_embedder()
    for start in range(0, len(chunks), slice_size):
        batch = chunks[start : start + slice_size]
        texts = [_embedding_input(report, chunk) for chunk in batch]
        for chunk, vector in zip(batch, embedder.embed_documents(texts), strict=True):
            chunk.embedding = vector
        _set_progress(db, report, ReportStatus.EMBEDDING, (start + len(batch)) / len(chunks))


def _embedding_input(report: Report, chunk: Chunk) -> str:
    """What gets embedded: the chunk text behind a header naming the report and page, so the
    vector carries its context."""
    return f"{contextual_header(report.company, report.fiscal_year, chunk.page)}\n{chunk.content}"
