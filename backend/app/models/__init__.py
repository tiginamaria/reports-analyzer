"""SQLAlchemy models. See docs/DESIGN.md, section "Data model".

One module per aggregate; everything is re-exported here so `from app.models import Report`
keeps working for the pipelines, the evaluation scripts and Alembic."""

from app.models.base import Base
from app.models.conversation import Conversation, Message
from app.models.report import (
    IN_PROGRESS_STATUSES,
    INDEXED_STATUSES,
    Chunk,
    Datapoint,
    Extraction,
    ExtractionStatus,
    Page,
    Report,
    ReportStatus,
)

__all__ = [
    "IN_PROGRESS_STATUSES",
    "INDEXED_STATUSES",
    "Base",
    "Chunk",
    "Conversation",
    "Datapoint",
    "Extraction",
    "ExtractionStatus",
    "Message",
    "Page",
    "Report",
    "ReportStatus",
]
