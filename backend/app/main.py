"""FastAPI application: routers, error handlers and startup."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.routers import chat, datapoints, reports, system
from app.config import get_settings
from app.services.maintenance import recover_interrupted_ingestions, warn_about_stale_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("LLM provider: %s", settings.llm_provider)
    log.info(
        "Embeddings: %s; reranker: %s",
        settings.embedding_signature,
        settings.reranker_name or "none",
    )
    await recover_interrupted_ingestions()
    await warn_about_stale_embeddings()
    yield


app = FastAPI(title="Annual Report RAG Analyzer", version="0.1.0", lifespan=lifespan)
register_error_handlers(app)
app.include_router(system.router)
app.include_router(reports.router)
app.include_router(datapoints.router)
app.include_router(chat.router)
