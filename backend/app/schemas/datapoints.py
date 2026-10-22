"""Pre-extracted datapoints."""

from typing import Any

from pydantic import BaseModel, ConfigDict


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
