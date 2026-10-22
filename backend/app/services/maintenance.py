"""What the process does at startup, and what the status endpoint reports."""

import logging

from app.config import get_settings
from app.db import SessionLocal
from app.ingestion.pipeline import RETRY_ACTION, STAGE_LABELS
from app.llm import get_provider
from app.models import ExtractionStatus, ReportStatus
from app.repositories.reports import ReportRepository
from app.schemas import StatusOut

log = logging.getLogger(__name__)


async def recover_interrupted_ingestions() -> None:
    """Ingestion runs in-process, so a report left mid-way by a previous process can never
    finish on its own. A report that was still building its index is marked failed. One
    that was already extracting has a complete index: it becomes `ready`, and the entries
    it was extracting are marked failed so the UI offers a retry instead of a re-upload."""
    async with SessionLocal() as db:
        reports = ReportRepository(db)
        interrupted = await reports.in_progress()
        for report in interrupted:
            if report.status == ReportStatus.EXTRACTING:
                report.status = ReportStatus.READY
                report.progress = 100
            else:
                stage = STAGE_LABELS.get(ReportStatus(report.status), "Ingestion").lower()
                report.status = ReportStatus.FAILED
                report.activity = None
                report.error = (
                    f"Ingestion was interrupted by a restart during {stage}. {RETRY_ACTION}"
                )
        pending = await reports.pending_extractions()
        for extraction in pending:
            extraction.status = ExtractionStatus.ERROR
            extraction.message = "Extraction was interrupted by a restart. Retry it."
        if interrupted or pending:
            await db.commit()
            log.warning(
                "Recovered %d interrupted ingestion(s) and %d pending extraction(s)",
                len(interrupted),
                len(pending),
            )


async def reports_to_reembed() -> int:
    """Indexed reports whose vectors come from another embedding model than the configured
    one. They are not searched until `python -m app.reembed` rebuilds them."""
    async with SessionLocal() as db:
        signature = get_settings().embedding_signature
        return await ReportRepository(db).count_indexed_with_other_embedding(signature)


async def warn_about_stale_embeddings() -> None:
    stale = await reports_to_reembed()
    if stale:
        log.warning(
            "%d indexed report(s) were embedded with another model than %s and will not be "
            "searched; run `python -m app.reembed` to rebuild their vectors",
            stale,
            get_settings().embedding_signature,
        )


async def system_status() -> StatusOut:
    settings = get_settings()
    provider = get_provider()
    return StatusOut(
        llm_provider=provider.name,
        llm_model=provider.model_name(),
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_signature,
        reranker_model=settings.reranker_name,
        reports_to_reembed=await reports_to_reembed(),
    )
