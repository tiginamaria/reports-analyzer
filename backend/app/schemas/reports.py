"""Reports: index status, upload results and the extraction retry request."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ExtractionOut(BaseModel):
    """Outcome of one registry entry on a report; mirrors `ExtractionStatus`."""

    model_config = ConfigDict(from_attributes=True)

    key: str
    status: str
    message: str | None
    updated_at: datetime


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    company: str
    fiscal_year: int
    title: str | None
    filename: str
    page_count: int
    status: str
    progress: int
    # Progress detail of the current stage: units done of total (pages, chunks, entries),
    # the worker's latest activity and when ingestion started. `updated_at` moves with them.
    done: int
    total: int
    activity: str | None
    started_at: datetime | None
    # Which vectors the report holds; searchable only when it equals the status endpoint's
    # `embedding_model`, otherwise it needs re-embedding (`python -m app.reembed`).
    embedding_model: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    # One entry per registry key once extraction has started; an entry with datapoints and
    # one whose extraction failed are told apart here, not by counting datapoints.
    extractions: list[ExtractionOut]


class ReportDetail(ReportOut):
    chunk_count: int
    datapoint_count: int


class ExtractRequest(BaseModel):
    """Registry keys to extract again; None means every entry whose last attempt failed."""

    keys: list[str] | None = None


class RejectedFile(BaseModel):
    filename: str
    reason: str


class UploadResult(BaseModel):
    started: list[ReportOut]
    rejected: list[RejectedFile]
