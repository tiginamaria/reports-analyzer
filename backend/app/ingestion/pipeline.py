"""Ingestion pipeline: parse -> detect metadata -> store pages -> chunk -> embed -> extract.

Runs as a background task with its own database session. Status, progress and the latest
activity are written to the report row after every page, every embedding batch and every
extraction step, so the UI can show what is happening and how far along it is.

The report's status tracks its index: `ready` means the pages, chunks and embeddings are
complete and the report can be searched. How each registry entry's extraction went is
recorded separately, per entry, in `extractions` (see `extractor.extract_report`); a failed
entry never fails the report and can be extracted again with `rerun_extraction`.

The pipeline is async: database writes await the session, while the blocking work (PyMuPDF,
the embedding model or endpoint, the LLM) runs in worker threads. Callbacks those threads
need (progress after every page, cancellation before every batch) are bridged back to the
loop with `concurrency.sync_callback`, so they still write progress and check the database.

Deleting the report cancels its ingestion: the task checks `cancellation.check` between
inference batches and before every write and stops without touching the database.
"""

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.concurrency import sync_callback
from app.db import SessionLocal
from app.extraction.extractor import (
    detect_metadata,
    extract_report,
    record_failure,
    start_extraction,
)
from app.ingestion import cancellation
from app.ingestion.cancellation import IngestionCancelled
from app.ingestion.chunker import chunk_document, contextual_header
from app.ingestion.parser import ParsedDocument, parse_pdf
from app.llm import LLMError, get_provider
from app.models import Chunk, Page, Report, ReportStatus
from app.retrieval.embedder import EmbeddingError, get_embedder, models_to_load

log = logging.getLogger(__name__)

# Progress range (start, end) of each stage, in percent of the whole ingestion.
PROGRESS_RANGES = {
    ReportStatus.PENDING: (0, 0),
    ReportStatus.PARSING: (0, 30),
    ReportStatus.CHUNKING: (30, 35),
    ReportStatus.EMBEDDING: (35, 80),
    ReportStatus.EXTRACTING: (80, 100),
    ReportStatus.READY: (100, 100),
}

# How a stage is named in an error message.
STAGE_LABELS = {
    ReportStatus.PENDING: "Starting",
    ReportStatus.PARSING: "Parsing",
    ReportStatus.CHUNKING: "Chunking",
    ReportStatus.EMBEDDING: "Embedding",
    ReportStatus.EXTRACTING: "Extraction",
}
RETRY_ACTION = "Delete the report and upload it again."

MAX_ERROR_LENGTH = 2000


async def ingest_report(report_id: str, path: Path) -> None:
    """Background-task entry point. Any failure is recorded on the report, never raised;
    a report deleted meanwhile stops its ingestion quietly."""
    async with SessionLocal() as db:
        report = await db.get(Report, report_id)
        if report is None:
            return
        cancellation.begin(report_id)
        try:
            await _run_stages(db, report, path)
        except IngestionCancelled:
            await db.rollback()
            log.info("Report %s was deleted during ingestion; stopping", report_id)
        except Exception as exc:  # any failure is shown in the UI
            stage = ReportStatus(report.status) if report.status in STAGE_LABELS else None
            await db.rollback()
            await _mark_failed(db, report_id, exc, stage)
        finally:
            cancellation.clear(report_id)


async def _run_stages(db: AsyncSession, report: Report, path: Path) -> None:
    check = partial(cancellation.check, db, report.id)
    report.started_at = datetime.now(UTC)

    doc = await _parse(db, report, path)
    await check()
    await _detect_metadata(db, report, doc)
    await check()

    await _set_progress(db, report, ReportStatus.CHUNKING, activity="Storing pages")
    _store_pages(db, report, doc)
    chunks = _chunk(db, report, doc)
    await check()

    await _set_progress(db, report, ReportStatus.EMBEDDING, total=len(chunks))
    await _embed(db, report, chunks)

    await _set_progress(db, report, ReportStatus.EXTRACTING, activity="Starting extraction")
    await _extract(db, report)

    await _set_progress(db, report, ReportStatus.READY, fraction=1.0)
    log.info("Ingested %s (%d pages, %d chunks)", report.filename, doc.page_count, len(chunks))


async def _extract(db: AsyncSession, report: Report, keys: list[str] | None = None) -> None:
    """The extraction stage. The index is complete by now, so a failure here (no provider
    configured, the reranker, the database) does not fail the report: every entry without
    an outcome is recorded as failed, which the UI shows next to a retry."""
    check = partial(cancellation.check, db, report.id)

    async def on_progress(done: int, total: int, activity: str) -> None:
        await _set_progress(
            db, report, ReportStatus.EXTRACTING, done=done, total=total, activity=activity
        )

    try:
        await extract_report(
            db, report, get_provider(), keys=keys, check=check, on_progress=on_progress
        )
    except IngestionCancelled:
        raise
    except Exception as exc:
        await db.rollback()
        log.error("Extraction failed for %s", report.filename, exc_info=exc)
        await record_failure(db, report, exc, keys=keys)


async def begin_reextraction(db: AsyncSession, report: Report, keys: list[str]) -> None:
    """What the retry endpoint does before scheduling `rerun_extraction`: the entries become
    `pending` and the report `extracting`, so the response already shows the retry running."""
    await start_extraction(db, report, keys)
    await _set_progress(db, report, ReportStatus.EXTRACTING, activity="Starting extraction")
    await db.refresh(report, ["extractions"])  # the response shows the rows just set to pending


async def rerun_extraction(report_id: str, keys: list[str]) -> None:
    """Background-task entry point of the retry endpoint: extract `keys` again on an indexed
    report. Its pages, chunks and embeddings are untouched; only the entries' datapoints are
    replaced. The report is `ready` again afterwards whatever the entries' outcomes."""
    async with SessionLocal() as db:
        report = await db.get(Report, report_id)
        if report is None:
            return
        cancellation.begin(report_id)
        try:
            await _extract(db, report, keys)
            await _set_progress(db, report, ReportStatus.READY, fraction=1.0)
        except IngestionCancelled:
            await db.rollback()
            log.info("Report %s was deleted during re-extraction; stopping", report_id)
        except Exception as exc:  # the database itself; the entries stay pending
            await db.rollback()
            log.error("Re-extraction of %s failed for %s", keys, report_id, exc_info=exc)
        finally:
            cancellation.clear(report_id)


async def _mark_failed(
    db: AsyncSession, report_id: str, exc: Exception, stage: ReportStatus | None = None
) -> None:
    report = await db.get(Report, report_id)
    if report is None:
        log.info("Report %s was deleted during ingestion; stopping", report_id)
        return
    log.error("Ingestion failed for %s", report_id, exc_info=exc)
    report.status = ReportStatus.FAILED
    report.progress = 0
    report.activity = None
    report.error = failure_message(exc, stage)
    await db.commit()


def failure_message(exc: Exception, stage: ReportStatus | None) -> str:
    """`Embedding failed: <cause>. Delete the report and upload it again.`: the stage, a cause
    a user can act on (provider errors carry their own message; anything else its type) and
    what to do next."""
    cause = (
        str(exc) if isinstance(exc, EmbeddingError | LLMError) else f"{type(exc).__name__}: {exc}"
    )
    label = STAGE_LABELS.get(stage, "Ingestion") if stage else "Ingestion"
    return f"{label} failed: {cause.rstrip('.')}. {RETRY_ACTION}"[:MAX_ERROR_LENGTH]


async def _set_progress(
    db: AsyncSession,
    report: Report,
    stage: ReportStatus,
    fraction: float = 0.0,
    *,
    done: int = 0,
    total: int = 0,
    activity: str | None = None,
) -> None:
    """Record the current stage, how far into it ingestion is (`done` of `total` units, or
    `fraction` 0..1 when the stage has no units) and what it is doing right now."""
    if total:
        fraction = done / total
    start, end = PROGRESS_RANGES[stage]
    report.status = stage
    report.progress = int(start + (end - start) * fraction)
    report.done = done
    report.total = total
    report.activity = activity
    await db.commit()


async def _parse(db: AsyncSession, report: Report, path: Path) -> ParsedDocument:
    """Parse the PDF in a worker thread. After every page the thread reports back to the
    loop, which checks for cancellation and records the progress before the next page."""
    await _set_progress(db, report, ReportStatus.PARSING, activity="Opening the PDF")
    started = time.perf_counter()

    async def on_progress(pages_done: int, total_pages: int) -> None:
        await cancellation.check(db, report.id)
        await _set_progress(
            db,
            report,
            ReportStatus.PARSING,
            done=pages_done,
            total=total_pages,
            activity=f"Parsed page {pages_done} of {total_pages} ({_elapsed(started)})",
        )

    return await asyncio.to_thread(parse_pdf, path, sync_callback(on_progress))


async def _detect_metadata(db: AsyncSession, report: Report, doc: ParsedDocument) -> None:
    """Ask the LLM for company, fiscal year and title from the first pages. If that fails
    the report keeps year 0 and gets the file name as its company, so it stays visible."""
    await _set_progress(
        db, report, ReportStatus.PARSING, fraction=1.0, activity="Detecting company and year"
    )
    try:
        meta = await asyncio.to_thread(detect_metadata, get_provider(), doc.pages, report.filename)
    except Exception as exc:  # detection is best effort
        log.warning("Metadata detection failed for %s: %s", report.filename, exc)
        report.company = Path(report.filename).stem
        return
    report.company = meta.company
    report.fiscal_year = meta.fiscal_year
    report.title = meta.title


def _store_pages(db: AsyncSession, report: Report, doc: ParsedDocument) -> None:
    """Page texts are the source of truth for verbatim checks and page-level citations."""
    report.page_count = doc.page_count
    db.add_all(
        Page(report_id=report.id, page_no=index + 1, text=text)
        for index, text in enumerate(doc.pages)
    )


def _chunk(db: AsyncSession, report: Report, doc: ParsedDocument) -> list[Chunk]:
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


async def _embed(
    db: AsyncSession, report: Report, chunks: list[Chunk], slice_size: int | None = None
) -> None:
    """Embed the chunks in slices of `slice_size` (the embedder's own by default: one model
    batch locally, one window of concurrent requests hosted). Each slice is embedded in a
    worker thread; after every slice the vectors are persisted and the progress updated.
    Before every slice, and before every batch inside it, the cancellation check runs (the
    embedder gets it bridged to the loop), so a deleted report stops within one batch."""
    check = partial(cancellation.check, db, report.id)
    total = len(chunks)
    pending = models_to_load()
    if pending:
        await _set_progress(
            db, report, ReportStatus.EMBEDDING, total=total, activity=f"Loading {pending[0]}"
        )
    embedder = await asyncio.to_thread(get_embedder)
    slice_size = slice_size or embedder.slice_size
    started = time.perf_counter()
    report.embedding_model = None  # not searchable until every vector is written
    for start in range(0, total, slice_size):
        await check()
        batch = chunks[start : start + slice_size]
        texts = [_embedding_input(report, chunk) for chunk in batch]
        batch_started = time.perf_counter()
        vectors = await asyncio.to_thread(embedder.embed_documents, texts, sync_callback(check))
        for chunk, vector in zip(batch, vectors, strict=True):
            chunk.embedding = vector
        await check()  # before persisting: the report may have gone while the batch ran
        done = start + len(batch)
        await _set_progress(
            db,
            report,
            ReportStatus.EMBEDDING,
            done=done,
            total=total,
            activity=(
                f"Embedded chunks {start + 1}-{done} of {total} in "
                f"{time.perf_counter() - batch_started:.1f} s ({_elapsed(started)} so far)"
            ),
        )
    report.embedding_model = embedder.signature


def _embedding_input(report: Report, chunk: Chunk) -> str:
    """What gets embedded: the chunk text behind a header naming the report and page, so the
    vector carries its context."""
    return f"{contextual_header(report.company, report.fiscal_year, chunk.page)}\n{chunk.content}"


def _elapsed(started: float) -> str:
    seconds = int(time.perf_counter() - started)
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"
