"""Structured outputs of the extraction models."""

from pydantic import BaseModel, Field


class Qualifier(BaseModel):
    name: str
    value: str


class ExtractedItem(BaseModel):
    label: str = Field(description="Short name of the datapoint or goal")
    value: str | None = Field(description="Value exactly as written in the quote")
    unit: str | None
    period: str | None = Field(description="Fiscal year, target year or date the value refers to")
    qualifiers: list[Qualifier]
    quote: str = Field(description="Verbatim sentence or table row copied from one chunk")
    chunk_id: str = Field(description="The id of the chunk the quote is copied from")


class ExtractionResult(BaseModel):
    found: bool
    items: list[ExtractedItem]


class ReportMetadata(BaseModel):
    company: str
    fiscal_year: int
    title: str | None
