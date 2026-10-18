"""FastAPI application: routers, startup, health and status."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, datapoints, reports
from app.config import get_settings
from app.db import SessionLocal
from app.llm import get_provider
from app.models import (
    IN_PROGRESS_STATUSES,
    Extraction,
    ExtractionStatus,
    Report,
    ReportStatus,
)
from app.schemas import StatusOut

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("LLM provider: %s", get_settings().llm_provider)
    _recover_interrupted_ingestions()
    yield


def _recover_interrupted_ingestions() -> None:
    """Ingestion runs in-process, so a report left mid-way by a previous process can never
    finish on its own. A report that was still building its index is marked failed. One
    that was already extracting has a complete index: it becomes `ready`, and the entries
    it was extracting are marked failed so the UI offers a retry instead of a re-upload."""
    with SessionLocal() as db:
        interrupted = db.query(Report).filter(Report.status.in_(IN_PROGRESS_STATUSES)).all()
        for report in interrupted:
            if report.status == ReportStatus.EXTRACTING:
                report.status = ReportStatus.READY
                report.progress = 100
            else:
                report.status = ReportStatus.FAILED
                report.error = "Ingestion was interrupted by a restart. Delete and upload again."
        pending = db.query(Extraction).filter(Extraction.status == ExtractionStatus.PENDING).all()
        for extraction in pending:
            extraction.status = ExtractionStatus.ERROR
            extraction.message = "Extraction was interrupted by a restart. Retry it."
        if interrupted or pending:
            db.commit()
            log.warning(
                "Recovered %d interrupted ingestion(s) and %d pending extraction(s)",
                len(interrupted),
                len(pending),
            )


app = FastAPI(title="Annual Report RAG Analyzer", version="0.1.0", lifespan=lifespan)
app.include_router(reports.router)
app.include_router(datapoints.router)
app.include_router(chat.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusOut)
def status() -> StatusOut:
    settings = get_settings()
    provider = get_provider()
    return StatusOut(
        llm_provider=provider.name,
        llm_model=provider.model_name(),
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )
