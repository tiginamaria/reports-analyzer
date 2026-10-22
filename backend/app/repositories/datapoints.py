"""Queries on pre-extracted datapoints."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Datapoint, Report


class DatapointRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_with_reports(
        self, report_id: str | None = None, key: str | None = None
    ) -> Sequence[tuple[Datapoint, Report]]:
        """Datapoints joined with their report, ordered the way the data page shows them."""
        stmt = select(Datapoint, Report).join(Report, Report.id == Datapoint.report_id)
        if report_id:
            stmt = stmt.where(Datapoint.report_id == report_id)
        if key:
            stmt = stmt.where(Datapoint.key == key)
        stmt = stmt.order_by(
            Report.company, Report.fiscal_year.desc(), Datapoint.key, Datapoint.page
        )
        return [(datapoint, report) for datapoint, report in (await self.db.execute(stmt)).all()]
