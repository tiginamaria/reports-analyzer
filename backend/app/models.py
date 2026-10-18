"""SQLAlchemy models. See docs/MVP_DESIGN.md, section "Data model"."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class ReportStatus(StrEnum):
    """Lifecycle of a report. Ingestion moves it through the stages in this order."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    EXTRACTING = "extracting"
    READY = "ready"
    FAILED = "failed"


# Ingestion has not finished (or was cut short by a restart).
IN_PROGRESS_STATUSES = (
    ReportStatus.PENDING,
    ReportStatus.PARSING,
    ReportStatus.CHUNKING,
    ReportStatus.EMBEDDING,
    ReportStatus.EXTRACTING,
)

# The index is complete and the report can be searched; extraction only adds datapoints.
INDEXED_STATUSES = (ReportStatus.READY, ReportStatus.EXTRACTING)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    company: Mapped[str] = mapped_column(String(200), index=True)
    fiscal_year: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str | None] = mapped_column(String(300))
    filename: Mapped[str] = mapped_column(String(500))
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default=ReportStatus.PENDING)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Children are removed by the database's ON DELETE CASCADE (passive_deletes), so deleting
    # a report never loads its thousands of chunks and their vectors into memory first.
    pages: Mapped[list[Page]] = relationship(
        back_populates="report", cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="report", cascade="all, delete-orphan", passive_deletes=True
    )
    datapoints: Mapped[list[Datapoint]] = relationship(
        back_populates="report", cascade="all, delete-orphan", passive_deletes=True
    )


class Page(Base):
    """Full text of one PDF page: the source of truth for verbatim quote checks and for
    pinning a citation to the exact page."""

    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("report_id", "page_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    report: Mapped[Report] = relationship(back_populates="pages")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"), index=True)
    page: Mapped[int] = mapped_column(Integer)  # chunks never span pages
    kind: Mapped[str] = mapped_column(String(10), default="text")  # text | table
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(), nullable=True)
    # tsv is a generated column (see migration); mapped as deferred read-only
    word_count: Mapped[int] = mapped_column(Integer, default=0)

    report: Mapped[Report] = relationship(back_populates="chunks")


class Datapoint(Base):
    __tablename__ = "datapoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(60), index=True)
    label: Mapped[str] = mapped_column(String(300))
    value: Mapped[str | None] = mapped_column(String(500))
    numeric_value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(100))
    period: Mapped[str | None] = mapped_column(String(100))
    qualifiers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    quote: Mapped[str | None] = mapped_column(Text)
    chunk_id: Mapped[str | None] = mapped_column(String(80))
    page: Mapped[int | None] = mapped_column(Integer)
    verification: Mapped[str] = mapped_column(String(20))  # verified | fuzzy | failed
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    report: Mapped[Report] = relationship(back_populates="datapoints")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(300), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(20))
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
