"""Chat pipeline: analyse -> retrieve -> rerank -> generate -> verify.

Conversation handling follows the usual RAG pattern: the small model rewrites the latest
question so it stands alone (retrieval needs a standalone query), then the main model gets
the transcript as real conversation turns, token-budgeted, with the chunks retrieved for this
question in the final user message. Earlier turns carry only question and answer text, never
their chunks, so each answer is grounded in and verified against this turn's evidence alone.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from sqlalchemy.orm import Session

from app.chat.schemas import ChatAnswer, Citation, QueryAnalysis
from app.config import get_settings
from app.llm import ChatTurn, LLMError, LLMProvider, Role
from app.models import INDEXED_STATUSES, Message, Report
from app.retrieval.search import Hit, format_context, retrieve
from app.textutil import normalize
from app.verify import verify_against_chunk

log = logging.getLogger(__name__)

EventType = Literal["conversation", "stage", "answer", "error"]


@dataclass
class ChatEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


def run_chat(
    db: Session,
    provider: LLMProvider,
    question: str,
    history: list[Message],
    report_ids: list[str] | None,
) -> Iterator[ChatEvent]:
    """Yield `stage` events as the pipeline progresses, then exactly one `answer` or `error`."""
    settings = get_settings()
    yield ChatEvent("stage", {"name": "analyzing"})

    turns = select_history(history, settings.history_max_tokens)
    analysis = _analyse(provider, question, turns)
    # Only fully indexed reports are searchable. Chunks of reports that are still embedding,
    # or that failed, are in the table too, so retrieval is always scoped explicitly.
    indexed = db.query(Report).filter(Report.status.in_(INDEXED_STATUSES)).all()
    scope = _resolve_scope(analysis, indexed, report_ids)

    yield ChatEvent(
        "stage",
        {
            "name": "retrieving",
            "query": analysis.standalone_question,
            "reports": [report.id for report in scope],
        },
    )
    if not scope:
        if report_ids:
            reason = "The selected reports are not ready yet or no longer exist."
        else:
            reason = "No report has been ingested yet."
        meta: dict[str, Any] = {"skipped_llm": True, "empty_scope": True}
        yield ChatEvent("answer", _abstain(db, reason, analysis, meta))
        return

    hits = retrieve(db, analysis.standalone_question, report_ids=[report.id for report in scope])
    best_score = max((hit.score for hit in hits), default=0.0)
    if not hits or best_score < settings.rerank_min_score:
        reason = "The ingested reports do not appear to cover this. Not stated in the report."
        meta = {"best_rerank_score": best_score, "skipped_llm": True}
        yield ChatEvent("answer", _abstain(db, reason, analysis, meta, hits))
        return

    yield ChatEvent("stage", {"name": "generating", "chunks": len(hits)})
    try:
        answer = provider.complete_json(
            system=settings.prompt_answer,
            user=_answer_prompt(analysis.standalone_question, hits),
            schema=ChatAnswer,
            max_tokens=4000,
            history=turns,
        )
    except LLMError as exc:
        yield ChatEvent("error", {"message": str(exc)})
        return
    meta = {"best_rerank_score": best_score}
    yield ChatEvent("answer", _finalize(db, answer, hits, analysis, meta))


def estimate_tokens(text: str) -> int:
    """Cheap upper-bound estimate, good enough for a budget (English prose runs ~4 chars/token)."""
    return len(text) // 3 + 1


def select_history(history: list[Message], max_tokens: int) -> list[ChatTurn]:
    """The most recent whole exchanges (a user message and its answer) that fit the token
    budget, oldest first. Chunks and citations of earlier turns are not replayed: only the text."""
    kept: list[list[ChatTurn]] = []
    used = 0
    for exchange in reversed(_exchanges(history)):
        cost = sum(estimate_tokens(turn.content) for turn in exchange)
        if used + cost > max_tokens:
            break
        kept.insert(0, exchange)
        used += cost
    return [turn for exchange in kept for turn in exchange]


def _exchanges(history: list[Message]) -> list[list[ChatTurn]]:
    """Group messages into exchanges: a user message plus the assistant reply, if it got one."""
    exchanges: list[list[ChatTurn]] = []
    for message in history:
        if message.role not in ("user", "assistant"):
            continue
        turn = ChatTurn(cast(Role, message.role), message.content)
        if turn.role == "user" or not exchanges:
            exchanges.append([turn])
        else:
            exchanges[-1].append(turn)
    return exchanges


def _analyse(provider: LLMProvider, question: str, turns: list[ChatTurn]) -> QueryAnalysis:
    """Standalone question plus company and year hints from the small model. Falls back to
    the raw question when the call fails, so the chat still works."""
    transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
    user = f"Conversation so far:\n{transcript or '(none)'}\n\nQuestion: {question}"
    try:
        return provider.complete_json(
            system=get_settings().prompt_query_analysis,
            user=user,
            schema=QueryAnalysis,
            small=True,
            max_tokens=600,
        )
    except LLMError as exc:
        log.warning("Query analysis failed, using raw question: %s", exc)
        return QueryAnalysis(standalone_question=question, companies=[], fiscal_year=None)


def _resolve_scope(
    analysis: QueryAnalysis, reports: list[Report], explicit_ids: list[str] | None
) -> list[Report]:
    """The indexed reports to search. An explicit filter wins and is never widened: if it
    matches no indexed report the result is empty and the caller abstains. Without one, the
    company and year from the query narrow the set when they match something."""
    if explicit_ids:
        wanted = set(explicit_ids)
        return [report for report in reports if report.id in wanted]
    scope = reports
    if analysis.companies:
        by_company = [r for r in scope if _company_matches(r.company, analysis.companies)]
        scope = by_company or scope
    if analysis.fiscal_year:
        by_year = [r for r in scope if r.fiscal_year == analysis.fiscal_year]
        scope = by_year or scope
    return scope


def _company_matches(company: str, names: list[str]) -> bool:
    """Loose match so that "shell" finds "Shell plc" and "CM.com" finds "CM com"."""
    target = _compact_name(company)
    for name in names:
        candidate = _compact_name(name)
        if candidate and (candidate in target or target in candidate):
            return True
    return False


def _compact_name(name: str) -> str:
    return normalize(name).replace(".", "").replace(" ", "")


def _answer_prompt(question: str, hits: list[Hit]) -> str:
    """The final user turn: this question's evidence, then the standalone question."""
    return f"Chunks:\n\n{format_context(hits)}\n\nQuestion: {question}"


def _abstain(
    db: Session,
    reason: str,
    analysis: QueryAnalysis,
    meta: dict[str, Any],
    hits: list[Hit] | None = None,
) -> dict[str, Any]:
    """A `not_found` answer produced without calling the main model."""
    answer = ChatAnswer(status="not_found", answer=reason, citations=[])
    return _finalize(db, answer, hits or [], analysis, meta)


def _finalize(
    db: Session,
    answer: ChatAnswer,
    hits: list[Hit],
    analysis: QueryAnalysis,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Verify every citation against the retrieved chunks and build the answer payload.
    An answer whose quotes all failed verification is downgraded to `not_found`."""
    hits_by_chunk = {hit.chunk.id: hit for hit in hits}
    citations = [_verify_citation(db, c, hits_by_chunk.get(c.chunk_id)) for c in answer.citations]
    trusted = [c for c in citations if c["verification"] in ("verified", "fuzzy")]

    status = answer.status
    if status == "answered" and citations and not trusted:
        status = "not_found"
    return {
        "status": status,
        "answer": answer.answer,
        "citations": citations,
        "meta": {
            **meta,
            "standalone_question": analysis.standalone_question,
            "retrieved_chunks": [hit.chunk.id for hit in hits],
            "dropped_citations": len(citations) - len(trusted),
        },
    }


def _verify_citation(db: Session, citation: Citation, hit: Hit | None) -> dict[str, Any]:
    """The citation as stored and shown, with its verification status and page. `hit` is
    None when the model cited a chunk it was never given."""
    if hit is None:
        return {"chunk_id": citation.chunk_id, "quote": citation.quote, "verification": "failed"}
    status = verify_against_chunk(
        db, hit.chunk, citation.quote, get_settings().fuzzy_quote_threshold
    )
    return {
        "chunk_id": citation.chunk_id,
        "quote": citation.quote,
        "verification": status,
        "report_id": hit.report.id,
        "company": hit.report.company,
        "fiscal_year": hit.report.fiscal_year,
        "page": hit.chunk.page,
    }
