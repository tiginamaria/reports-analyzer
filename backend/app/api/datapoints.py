"""Pre-extracted datapoints and the registry they come from."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.extraction.extractor import load_registry
from app.models import Datapoint, Report
from app.schemas import DatapointOut, DatapointWithReport

router = APIRouter(prefix="/api/datapoints", tags=["datapoints"])


@router.get("", response_model=list[DatapointWithReport])
def list_datapoints(
    report_id: str | None = None, key: str | None = None, db: Session = Depends(get_db)
) -> list[DatapointWithReport]:
    query = db.query(Datapoint, Report).join(Report, Report.id == Datapoint.report_id)
    if report_id:
        query = query.filter(Datapoint.report_id == report_id)
    if key:
        query = query.filter(Datapoint.key == key)
    rows = query.order_by(
        Report.company, Report.fiscal_year.desc(), Datapoint.key, Datapoint.page
    ).all()
    return [
        DatapointWithReport(
            **DatapointOut.model_validate(datapoint).model_dump(),
            company=report.company,
            fiscal_year=report.fiscal_year,
        )
        for datapoint, report in rows
    ]


@router.get("/registry")
def registry() -> list[dict]:
    """The datapoints every report is scanned for, as the UI needs them for its columns."""
    return [{"key": spec.key, "label": spec.label, "multi": spec.multi} for spec in load_registry()]
