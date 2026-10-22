"""Chat pipeline: rewrite query -> retrieve -> rerank -> generate -> verify.

Conversation handling follows the usual RAG pattern: the small model rewrites the latest
question so it stands alone (retrieval needs a standalone query), seeing every earlier
question and the last few answers; then the main model gets the whole conversation as real
turns, with the chunks retrieved for this question in the final user message. Earlier turns
carry only question and answer text, never their chunks, so each answer is grounded in and
verified against this turn's evidence alone.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import prompts
from app.chat.schemas import ChatAnswer, Citation, Query
from app.config import get_settings
from app.llm import ChatTurn, LLMError, LLMProvider, Role
from app.models import IN_PROGRESS_STATUSES, INDEXED_STATUSES, Message, Report, ReportStatus
from app.retrieval.embedder import get_embedder, get_reranker, models_to_load
from app.retrieval.search import Hit, format_context, retrieve
from app.verify import verify_against_chunk

log = logging.getLogger(__name__)

EventType = Literal["conversation", "stage", "answer", "error"]

# How much of the conversation the rewrite gets with a new question (see `_rewrite_turns`);
# the main model gets the whole conversation.
REWRITE_QUESTIONS = 30  # last questions sent to the rewrite
REWRITE_ANSWER_MESSAGES = 6  # last messages whose answers are sent too, about three answers
MAX_SUBQUERIES = 10  # subqueries searched, each keeping RERANK_TOP_K chunks; later ones are dropped
# Shown instead of an answer none of whose quotes could be found in the report.
UNVERIFIED_ANSWER = (
    "Not stated in the report: the answer's quotes could not be found in the report text."
)

# Greetings, thanks and acknowledgements: answered at once, without the model or retrieval.
SMALL_TALK = re.compile(
    r"^(?:"
    r"(?:hi|hello|hey|hiya|howdy|yo|hallo|hoi|greetings|good\s*(?:morning|afternoon|evening|day))"
    r"(?:\s+(?:there|all|again|everyone))?"
    r"|(?:thanks|thank\s+you|thx|ty|cheers|bedankt|dank\s*je(?:\s*wel)?)"
    r"(?:\s+(?:a\s+lot|so\s+much|very\s+much))?"
    r"|ok|okay|great|cool|nice|perfect|bye|goodbye|see\s+you|good\s*night"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)

# Why a question was answered without searching; stored in the answer's `meta.reason`.
Reason = Literal[
    "greeting",  # small talk
    "no_reports",  # nothing uploaded (or the selected reports no longer exist)
    "reports_processing",  # reports exist but none is indexed yet
    "reports_failed",  # every report failed to ingest
    "reports_need_reembedding",  # indexed with another embedding model than the configured one
    "no_evidence",  # searched; every chunk scored below the floor
]


@dataclass
class ChatEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


async def run_chat(
    db: AsyncSession,
    provider: LLMProvider,
    question: str,
    history: list[Message],
    report_ids: list[str] | None,
) -> AsyncIterator[ChatEvent]:
    """Yield `stage` events as the pipeline progresses, then exactly one `answer` or `error`.

    Database work awaits the session; the model calls (loading local models, the rewrite,
    the answer) are blocking and run in worker threads.

    Before anything costly runs, small talk gets a canned reply and the library is checked:
    with nothing to search the answer says what is missing (no report, reports still
    processing, failed, or embedded with another model) without calling the model or
    loading any local model. Local models that still have to load are reported as their
    own stage, so a first question after a start does not look stuck."""
    library = await _library(db, report_ids)
    if SMALL_TALK.match(question):
        yield ChatEvent(
            "answer", _quick_answer(question, "greeting", _greeting(library), "answered")
        )
        return
    if (unavailable := library.unavailable()) is not None:
        reason, text = unavailable
        yield ChatEvent("answer", _quick_answer(question, reason, text))
        return

    if pending := models_to_load():
        yield ChatEvent("stage", {"name": "loading_models", "models": pending})
        await asyncio.to_thread(get_embedder)
        await asyncio.to_thread(get_reranker)

    yield ChatEvent("stage", {"name": "analyzing"})
    query = await asyncio.to_thread(_rewrite_query, provider, question, history)
    yield ChatEvent("stage", {"name": "retrieving", "query": query.standalone_question})

    hits = await _retrieve_all(db, query, report_ids)
    if not hits:  # every chunk scored below the floor
        text = "The ingested reports do not appear to cover this. Not stated in the report."
        if library.processing:
            text += f" {library.describe_processing()}"
        yield ChatEvent("answer", await _abstain(db, text, query, "no_evidence"))
        return

    yield ChatEvent("stage", {"name": "generating", "chunks": len(hits)})
    try:
        answer = await asyncio.to_thread(
            _generate_answer, provider, query.standalone_question, hits, history
        )
    except LLMError as exc:
        yield ChatEvent("error", {"message": str(exc)})
        return
    yield ChatEvent("answer", await _finalize(db, answer, hits, query))


@dataclass
class Library:
    """What the user's question can be searched against: the reports within the filter
    (all reports without one), by what state they are in."""

    searchable: list[str] = field(default_factory=list)  # labels; indexed with the right vectors
    processing: list[str] = field(default_factory=list)  # "ABN AMRO 2025 (embedding, 59%)"
    failed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)  # indexed with another embedding model
    filtered: bool = False

    def unavailable(self) -> tuple[Reason, str] | None:
        """The reason nothing can be searched and what to tell the user, or None when at
        least one report is searchable."""
        if self.searchable:
            return None
        if self.processing:
            return "reports_processing", (
                f"{self.describe_processing()} Ask again once a report is ready."
            )
        if self.stale:
            verb = "was" if len(self.stale) == 1 else "were"
            return "reports_need_reembedding", (
                f"{_count(len(self.stale), 'report')} ({_names(self.stale)}) {verb} indexed with "
                "a different embedding model than the one configured now. Re-embed "
                "(python -m app.reembed) before asking again."
            )
        if self.failed:
            return "reports_failed", (
                f"Ingestion failed for {_names(self.failed)}. Delete the report and upload it again."
            )
        if self.filtered:
            return "no_reports", (
                "The selected reports no longer exist. Choose another report or upload one."
            )
        return "no_reports", (
            "No report has been uploaded yet. Upload an annual report on the Reports page, "
            "then ask again."
        )

    def describe_processing(self) -> str:
        return f"{_count(len(self.processing), 'report')} still being processed: {_names(self.processing)}."


def _library_statement(report_ids: list[str] | None) -> Select:
    """The reports within the user's filter (all without one), by what the `Library` sorts on."""
    stmt = select(
        Report.company, Report.fiscal_year, Report.status, Report.progress, Report.embedding_model
    )
    if report_ids:
        stmt = stmt.where(Report.id.in_(report_ids))
    return stmt


async def _library(db: AsyncSession, report_ids: list[str] | None) -> Library:
    """One query over the reports (within the user's filter) sorted into the `Library`."""
    signature = get_settings().embedding_signature
    rows = await db.execute(_library_statement(report_ids))
    library = Library(filtered=bool(report_ids))
    for company, year, status, progress, embedding in rows.all():
        label = " ".join(part for part in (company, str(year or "")) if part) or "a report"
        if status in INDEXED_STATUSES and embedding == signature:
            library.searchable.append(label)
        elif status in INDEXED_STATUSES:
            library.stale.append(label)
        elif status in IN_PROGRESS_STATUSES:
            library.processing.append(f"{label} ({status}, {progress}%)")
        elif status == ReportStatus.FAILED:
            library.failed.append(label)
    return library


def _greeting(library: Library) -> str:
    if library.searchable:
        return (
            "Hello! Ask me about the ingested annual reports, for example a figure, a target or "
            f"a comparison. Reports ready to search: {_names(library.searchable)}."
        )
    if library.processing:
        return f"Hello! {library.describe_processing()} Ask me about them once they are ready."
    return "Hello! Upload an annual report on the Reports page, then ask me about it."


def _names(labels: list[str], limit: int = 5) -> str:
    shown = ", ".join(labels[:limit])
    return shown if len(labels) <= limit else f"{shown} and {len(labels) - limit} more"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _conversation_turns(history: list[Message]) -> list[ChatTurn]:
    """The whole conversation as the main model gets it with a new question: questions
    (user) and answers (assistant), starting with a question. Only the text is sent, never
    the chunks of earlier turns. Sending everything keeps the prefix append-only, so the
    provider's prompt cache serves it."""
    turns = [
        ChatTurn(cast(Role, m.role), m.content) for m in history if m.role in ("user", "assistant")
    ]
    while turns and turns[0].role == "assistant":
        turns.pop(0)
    return turns


def _rewrite_turns(
    history: list[Message],
    questions: int = REWRITE_QUESTIONS,
    answer_messages: int = REWRITE_ANSWER_MESSAGES,
) -> list[ChatTurn]:
    """The conversation as the rewrite sees it: the last `questions` questions, and answers
    only within the last `answer_messages` messages, in order.

    A follow-up like "And emissions?" refers to a company and year that may have been named
    many turns back, and only in a question. Questions are short, so all of them are kept;
    answers are long and only matter for what was just said. The raw questions are kept
    rather than their earlier rewrites, so a wrong rewrite never feeds later ones."""
    messages = [m for m in history if m.role in ("user", "assistant")]
    question_indexes = [i for i, m in enumerate(messages) if m.role == "user"]
    first = question_indexes[-questions] if len(question_indexes) > questions else 0
    recent = max(first, len(messages) - answer_messages)
    return [
        ChatTurn(cast(Role, m.role), m.content)
        for i, m in enumerate(messages[first:], start=first)
        if m.role == "user" or i >= recent
    ]


def _rewrite_query(provider: LLMProvider, question: str, history: list[Message]) -> Query:
    """The small model rewrites the question to stand alone and splits it into subqueries,
    one per company, year and topic. Falls back to the raw question when the call fails, so
    the chat still works. It sees the part of the conversation `_rewrite_turns` selects.
    Blocking: run it in a worker thread."""
    turns = _rewrite_turns(history)
    transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
    user = f"Conversation so far:\n{transcript or '(none)'}\n\nQuestion: {question}"
    try:
        return provider.complete_json(
            system=prompts.QUERY_REWRITE,
            user=user,
            schema=Query,
            small=True,
        )
    except LLMError as exc:
        log.warning("Query rewrite failed, using raw question: %s", exc)
        return Query(standalone_question=question, subqueries=[])


async def _retrieve_all(
    db: AsyncSession,
    query: Query,
    report_ids: list[str] | None,
    max_subqueries: int = MAX_SUBQUERIES,
) -> list[Hit]:
    """Retrieve for every subquery (or the standalone question if there are none): each keeps
    its own top chunks, and they are merged best first, a chunk found by two subqueries kept
    once. Each subquery is searched by its `search_text`, which always names its company and
    year. At most `max_subqueries` are searched. `report_ids` is the user's report filter;
    retrieval itself keeps to indexed reports."""
    if len(query.subqueries) > max_subqueries:
        log.warning("Searching %d of %d subqueries", max_subqueries, len(query.subqueries))
    subqueries = query.subqueries[:max_subqueries]
    searches = [subquery.search_text for subquery in subqueries] or [query.standalone_question]
    best: dict[str, Hit] = {}
    for search in searches:
        for hit in await retrieve(db, search, report_ids):
            if hit.chunk.id not in best or best[hit.chunk.id].score < hit.score:
                best[hit.chunk.id] = hit
    return sorted(best.values(), key=lambda hit: hit.score, reverse=True)


def _answer_prompt(question: str, hits: list[Hit]) -> str:
    """The final user turn: this question's chunks, then the standalone question."""
    return f"Chunks:\n\n{format_context(hits)}\n\nQuestion: {question}"


def _generate_answer(
    provider: LLMProvider, question: str, hits: list[Hit], history: list[Message]
) -> ChatAnswer:
    """The main model's structured answer. The conversation goes as earlier turns; the final
    user turn holds this question's chunks, then the standalone question. Raises `LLMError`
    when the call fails. Blocking: run it in a worker thread."""
    return provider.complete_json(
        system=prompts.ANSWER,
        user=_answer_prompt(question, hits),
        schema=ChatAnswer,
        max_tokens=4000,
        history=_conversation_turns(history),
    )


async def _abstain(db: AsyncSession, text: str, query: Query, reason: Reason) -> dict[str, Any]:
    """A `not_found` answer produced after retrieval but without calling the main model."""
    answer = ChatAnswer(status="not_found", answer=text, citations=[])
    return await _finalize(db, answer, [], query, reason)


def _quick_answer(
    question: str, reason: Reason, text: str, status: str = "not_found"
) -> dict[str, Any]:
    """An answer produced before any model call or retrieval (small talk, nothing to search)."""
    return {
        "status": status,
        "answer": text,
        "citations": [],
        "meta": {
            "reason": reason,
            "skipped_llm": True,
            "skipped_retrieval": True,
            "standalone_question": question,
            "queries": [],
            "retrieved_chunks": [],
            "dropped_citations": 0,
        },
    }


async def _finalize(
    db: AsyncSession,
    answer: ChatAnswer,
    hits: list[Hit],
    query: Query,
    reason: Reason | None = None,
) -> dict[str, Any]:
    """Verify every citation against the retrieved chunks and build the answer payload.
    An answer whose quotes all failed verification is downgraded to `not_found`, and its text
    is replaced: it states figures nothing in the report supports. The original text is kept
    in `meta` for debugging."""
    hits_by_chunk = {hit.chunk.id: hit for hit in hits}
    citations = [
        await _verify_citation(db, c, hits_by_chunk.get(c.chunk_id)) for c in answer.citations
    ]
    trusted = [c for c in citations if c["verification"] in ("verified", "fuzzy")]

    meta: dict[str, Any] = {}
    if reason:
        meta["reason"] = reason
    if hits:  # reranker probability; the RRF score when no reranker is configured (CI)
        meta["best_score"] = hits[0].score
    else:
        meta["skipped_llm"] = True
    status, text = answer.status, answer.answer
    if status == "answered" and citations and not trusted:
        status, text = "not_found", UNVERIFIED_ANSWER
        meta["unverified_answer"] = answer.answer
    return {
        "status": status,
        "answer": text,
        "citations": citations,
        "meta": {
            **meta,
            "standalone_question": query.standalone_question,
            "queries": [query.model_dump() for query in query.subqueries],
            "retrieved_chunks": [hit.chunk.id for hit in hits],
            "dropped_citations": len(citations) - len(trusted),
        },
    }


async def _verify_citation(db: AsyncSession, citation: Citation, hit: Hit | None) -> dict[str, Any]:
    """The citation as stored and shown, with its verification status and page. `hit` is
    None when the model cited a chunk it was never given."""
    if hit is None:
        return {"chunk_id": citation.chunk_id, "quote": citation.quote, "verification": "failed"}
    status = await verify_against_chunk(db, hit.chunk, citation.quote)
    return {
        "chunk_id": citation.chunk_id,
        "quote": citation.quote,
        "verification": status,
        "report_id": hit.report.id,
        "company": hit.report.company,
        "fiscal_year": hit.report.fiscal_year,
        "page": hit.chunk.page,
    }
