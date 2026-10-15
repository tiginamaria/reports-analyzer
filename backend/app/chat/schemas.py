from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    standalone_question: str = Field(
        description="The question rewritten to stand alone, with pronouns resolved"
    )
    companies: list[str] = Field(description="Company names the question is about; empty if any")
    fiscal_year: int | None


class Citation(BaseModel):
    chunk_id: str
    quote: str = Field(description="Verbatim passage copied from that chunk")


class ChatAnswer(BaseModel):
    status: Literal["answered", "partial", "not_found"]
    answer: str
    citations: list[Citation]
