"""FastAPI application: routers, startup, health and status."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, datapoints, reports
from app.config import get_settings
from app.db import SessionLocal
from app.llm import get_provider
from app.models import IN_PROGRESS_STATUSES, Report, ReportStatus
from app.schemas import StatusOut

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("LLM provider: %s", get_settings().llm_provider)
    _fail_interrupted_ingestions()
    yield


def _fail_interrupted_ingestions() -> None:
    """Ingestion runs in-process, so a report left mid-way by a previous process can never
    finish. Mark it failed so the UI does not show it as busy forever."""
    with SessionLocal() as db:
        interrupted = db.query(Report).filter(Report.status.in_(IN_PROGRESS_STATUSES)).all()
        for report in interrupted:
            report.status = ReportStatus.FAILED
            report.error = "Ingestion was interrupted by a restart. Delete and upload again."
        if interrupted:
            db.commit()
            log.warning("Marked %d interrupted ingestion(s) as failed", len(interrupted))


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
