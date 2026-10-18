"""Chat pipeline: rewrite query -> retrieve -> rerank -> generate -> verify.

Conversation handling follows the usual RAG pattern: the small model rewrites the latest
question so it stands alone (retrieval needs a standalone query), seeing every earlier
question and the last few answers; then the main model gets the whole conversation as real
turns, with the chunks retrieved for this question in the final user message. Earlier turns
carry only question and answer text, never their chunks, so each answer is grounded in and
verified against this turn's evidence alone.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from sqlalchemy.orm import Session

from app.chat import prompts
from app.chat.schemas import ChatAnswer, Citation, Query
from app.llm import ChatTurn, LLMError, LLMProvider, Role
from app.models import INDEXED_STATUSES, Message, Report
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
    yield ChatEvent("stage", {"name": "analyzing"})

    query = _rewrite_query(provider, question, history)
    yield ChatEvent("stage", {"name": "retrieving", "query": query.standalone_question})

    hits = _retrieve_all(db, query, report_ids)
    if not hits:
        if _searchable_reports(db, report_ids):  # every chunk scored below the floor
            reason = "The ingested reports do not appear to cover this. Not stated in the report."
        else:  # no report is ready yet, or none of the selected ones
            reason = "No ingested report could be searched. Not stated in the report."
        yield ChatEvent("answer", _abstain(db, reason, query))
        return

    yield ChatEvent("stage", {"name": "generating", "chunks": len(hits)})
    try:
        answer = _generate_answer(provider, query.standalone_question, hits, history)
    except LLMError as exc:
        yield ChatEvent("error", {"message": str(exc)})
        return
    yield ChatEvent("answer", _finalize(db, answer, hits, query))


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
    the chat still works. It sees the part of the conversation `_rewrite_turns` selects."""
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


def _retrieve_all(
    db: Session,
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
        for hit in retrieve(db, search, report_ids):
            if hit.chunk.id not in best or best[hit.chunk.id].score < hit.score:
                best[hit.chunk.id] = hit
    return sorted(best.values(), key=lambda hit: hit.score, reverse=True)


def _searchable_reports(db: Session, report_ids: list[str] | None) -> bool:
    """Whether retrieval had any indexed report to search, within the user's filter if
    given. Tells an empty result that means "nothing relevant" from one that means "nothing
    to search"."""
    reports = db.query(Report.id).filter(Report.status.in_(INDEXED_STATUSES))
    if report_ids:
        reports = reports.filter(Report.id.in_(report_ids))
    return reports.first() is not None


def _answer_prompt(question: str, hits: list[Hit]) -> str:
    """The final user turn: this question's chunks, then the standalone question."""
    return f"Chunks:\n\n{format_context(hits)}\n\nQuestion: {question}"


def _generate_answer(
    provider: LLMProvider, question: str, hits: list[Hit], history: list[Message]
) -> ChatAnswer:
    """The main model's structured answer. The conversation goes as earlier turns; the final
    user turn holds this question's chunks, then the standalone question. Raises `LLMError`
    when the call fails."""
    return provider.complete_json(
        system=prompts.ANSWER,
        user=_answer_prompt(question, hits),
        schema=ChatAnswer,
        max_tokens=4000,
        history=_conversation_turns(history),
    )


def _abstain(db: Session, reason: str, query: Query) -> dict[str, Any]:
    """A `not_found` answer produced without calling the main model."""
    answer = ChatAnswer(status="not_found", answer=reason, citations=[])
    return _finalize(db, answer, [], query)


def _finalize(
    db: Session,
    answer: ChatAnswer,
    hits: list[Hit],
    query: Query,
) -> dict[str, Any]:
    """Verify every citation against the retrieved chunks and build the answer payload.
    An answer whose quotes all failed verification is downgraded to `not_found`, and its text
    is replaced: it states figures nothing in the report supports. The original text is kept
    in `meta` for debugging."""
    hits_by_chunk = {hit.chunk.id: hit for hit in hits}
    citations = [_verify_citation(db, c, hits_by_chunk.get(c.chunk_id)) for c in answer.citations]
    trusted = [c for c in citations if c["verification"] in ("verified", "fuzzy")]

    meta: dict[str, Any] = {}
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


def _verify_citation(db: Session, citation: Citation, hit: Hit | None) -> dict[str, Any]:
    """The citation as stored and shown, with its verification status and page. `hit` is
    None when the model cited a chunk it was never given."""
    if hit is None:
        return {"chunk_id": citation.chunk_id, "quote": citation.quote, "verification": "failed"}
    status = verify_against_chunk(db, hit.chunk, citation.quote)
    return {
        "chunk_id": citation.chunk_id,
        "quote": citation.quote,
        "verification": status,
        "report_id": hit.report.id,
        "company": hit.report.company,
        "fiscal_year": hit.report.fiscal_year,
        "page": hit.chunk.page,
    }
