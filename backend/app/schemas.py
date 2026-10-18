"""Pydantic schemas for the HTTP API."""

from datetime import datetime
from typing import Any

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
    error: str | None
    created_at: datetime
    updated_at: datetime
    # One entry per registry key once extraction has started; an entry with datapoints and
    # one whose extraction failed are told apart here, not by counting datapoints.
    extractions: list[ExtractionOut]


class ReportDetail(ReportOut):
    chunk_count: int
    datapoint_count: int


class DatapointOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    report_id: str
    key: str
    label: str
    value: str | None
    numeric_value: float | None
    unit: str | None
    period: str | None
    qualifiers: dict[str, Any]
    quote: str | None
    chunk_id: str | None
    page: int | None
    verification: str
    model: str | None


class DatapointWithReport(DatapointOut):
    company: str
    fiscal_year: int


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    status: str | None
    citations: list[dict[str, Any]]
    meta: dict[str, Any]
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    report_ids: list[str] | None = None


class ExtractRequest(BaseModel):
    """Registry keys to extract again; None means every entry whose last attempt failed."""

    keys: list[str] | None = None


class RejectedFile(BaseModel):
    filename: str
    reason: str


class UploadResult(BaseModel):
    started: list[ReportOut]
    rejected: list[RejectedFile]


class StatusOut(BaseModel):
    llm_provider: str
    llm_model: str
    embedding_model: str
    reranker_model: str | None
