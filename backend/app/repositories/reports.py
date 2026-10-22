"""Queries on reports and their children."""

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    IN_PROGRESS_STATUSES,
    INDEXED_STATUSES,
    Chunk,
    Datapoint,
    Extraction,
    ExtractionStatus,
    Report,
)


class ReportRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list(self) -> Sequence[Report]:
        result = await self.db.execute(
            select(Report)
            .options(selectinload(Report.extractions))
            .order_by(Report.company, Report.fiscal_year.desc(), Report.filename)
        )
        return result.scalars().all()

    async def get(self, report_id: str, *, with_extractions: bool = False) -> Report | None:
        """The report, with its extraction outcomes loaded when the caller will read them
        (lazy loading is not available on an async session)."""
        options = [selectinload(Report.extractions)] if with_extractions else []
        return await self.db.get(Report, report_id, options=options)

    async def by_hash(self, digest: str) -> Report | None:
        result = await self.db.execute(select(Report).where(Report.file_hash == digest))
        return result.scalar_one_or_none()

    def add(self, report: Report) -> None:
        self.db.add(report)

    async def delete(self, report: Report) -> None:
        await self.db.delete(report)

    async def chunk_count(self, report_id: str) -> int:
        stmt = select(func.count(Chunk.id)).where(Chunk.report_id == report_id)
        return (await self.db.execute(stmt)).scalar() or 0

    async def datapoint_count(self, report_id: str) -> int:
        stmt = select(func.count(Datapoint.id)).where(Datapoint.report_id == report_id)
        return (await self.db.execute(stmt)).scalar() or 0

    async def in_progress(self) -> Sequence[Report]:
        """Reports whose ingestion has not finished."""
        stmt = select(Report).where(Report.status.in_(IN_PROGRESS_STATUSES))
        return (await self.db.execute(stmt)).scalars().all()

    async def pending_extractions(self) -> Sequence[Extraction]:
        stmt = select(Extraction).where(Extraction.status == ExtractionStatus.PENDING)
        return (await self.db.execute(stmt)).scalars().all()

    async def count_indexed_with_other_embedding(self, signature: str) -> int:
        """Indexed reports whose vectors come from another embedding model than `signature`."""
        stmt = (
            select(func.count())
            .select_from(Report)
            .where(Report.status.in_(INDEXED_STATUSES), Report.embedding_model != signature)
        )
        return (await self.db.execute(stmt)).scalar() or 0
