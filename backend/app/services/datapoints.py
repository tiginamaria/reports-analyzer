"""Pre-extracted datapoints and the registry they come from."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.extraction.extractor import load_registry
from app.repositories.datapoints import DatapointRepository
from app.schemas import DatapointOut, DatapointWithReport


class DatapointService:
    def __init__(self, db: AsyncSession) -> None:
        self.datapoints = DatapointRepository(db)

    async def list_all(
        self, report_id: str | None = None, key: str | None = None
    ) -> list[DatapointWithReport]:
        rows = await self.datapoints.list_with_reports(report_id, key)
        return [
            DatapointWithReport(
                **DatapointOut.model_validate(datapoint).model_dump(),
                company=report.company,
                fiscal_year=report.fiscal_year,
            )
            for datapoint, report in rows
        ]

    @staticmethod
    def registry() -> list[dict]:
        """The datapoints every report is scanned for, as the UI needs them for its columns."""
        return [
            {"key": spec.key, "label": spec.label, "multi": spec.multi} for spec in load_registry()
        ]
