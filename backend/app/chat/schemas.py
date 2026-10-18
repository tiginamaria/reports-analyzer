"""Structured outputs of the chat models: the query rewrite and the answer."""

from typing import Literal

from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    company: str | None = Field(
        description="Company this search is about, e.g. 'ASML'; null when neither the question "
        "nor the conversation names one"
    )
    fiscal_year: int | None = Field(
        description="Fiscal year of the annual report to search, e.g. 2025; null when no year is "
        "stated or implied. Never a target year such as 2030"
    )
    question: str = Field(
        description="A standalone English question about one topic in that report, naming the "
        "company and the year, e.g. 'How many employees did ASML have at the end of 2025?'"
    )

    @property
    def search_text(self) -> str:
        """The question as searched: unchanged when it names the company and year, otherwise
        with the missing one appended, phrased like the chunk headers ('Annual Report 2025')
        so the search still lands on the right report."""
        parts = [self.question]
        if self.company and self.company.lower() not in self.question.lower():
            parts.append(self.company)
        if self.fiscal_year and str(self.fiscal_year) not in self.question:
            parts.append(f"annual report {self.fiscal_year}")
        return " ".join(parts)


class Query(BaseModel):
    standalone_question: str = Field(
        description="The whole question rewritten to stand alone in the user's language, with "
        "pronouns resolved; the answer is written for this question"
    )
    subqueries: list[SubQuery] = Field(
        description="The searches to run: one per company, fiscal year and topic the question "
        "asks about. Never two companies, two years or two topics in one subquery"
    )


class Citation(BaseModel):
    chunk_id: str = Field(
        description="Id of the chunk the quote is copied from, as in its '[chunk ...]' header"
    )
    quote: str = Field(
        description="Verbatim passage copied character for character from that chunk, "
        "preferably the sentence or table row that contains the figure"
    )


class ChatAnswer(BaseModel):
    status: Literal["answered", "partial", "not_found"] = Field(
        description="'answered' if the chunks answer the question, 'partial' if they answer "
        "part of it or give only a related figure, 'not_found' if they do not answer it"
    )
    answer: str = Field(
        description="Short plain prose in the user's language, every figure with its unit and "
        "period (e.g. 'EUR 1.2 billion in 2025'); for 'not_found', say the report does not "
        "state it"
    )
    citations: list[Citation] = Field(description="One citation per claim; empty for 'not_found'")
