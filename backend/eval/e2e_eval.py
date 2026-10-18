"""End-to-end evaluation of the chat pipeline on the gold questions.

Runs the real `run_chat` (analysis, retrieval, rerank, generation, verification) per
question, with configuration overrides applied by patching the module constants that the
pipeline reads at call time. Answers are graded twice: deterministically (key facts in the
answer, citation verification, cited page among the gold pages) and by an LLM judge
(correctness against the gold answer, groundedness in the verified quotes). The judge is a
different, stronger model than the generator, to limit self-preference.

    uv run python -m eval.e2e_eval baseline
    uv run python -m eval.e2e_eval baseline fts_or --limit 10
"""

import argparse
import dataclasses
import functools
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal
from unittest import mock

from pydantic import BaseModel
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

import app.chat.pipeline as pipeline
import app.retrieval.search as search
from app.chat.pipeline import run_chat
from app.config import get_settings
from app.llm import LLMProvider
from app.llm.anthropic_provider import AnthropicProvider
from app.models import Chunk, Message
from app.retrieval.embedder import get_embedder
from app.textutil import normalize
from eval.common import USAGE, instrument, report_keys, require_reranker, write_json
from eval.retrieval_eval import FTS_SQL, Question, load_questions, session_for

JUDGE_MODEL = "claude-opus-5-5"
WORKERS = 4  # judge calls; the chat runs use --workers (1 for honest latency)


@dataclass(frozen=True)
class E2EConfig:
    """Defaults are the MVP settings, so `baseline` stays comparable after the app's
    constants change; `current` is whatever `retrieval/search.py` holds now."""

    variant: str = "base"
    candidates: int = 40
    top_k: int = 8
    fts: str | None = "fts"  # tsquery from FTS_SQL replacing the app's; None keeps the app's
    model: str | None = None  # main model override


# "tuned" is the best retrieval setting from retrieval_eval plus the abstention threshold
# that separates out-of-corpus questions from every answerable one.
TUNED = E2EConfig(candidates=60, top_k=12)
CONFIGS = {
    "baseline": E2EConfig(),
    "tuned": TUNED,
    "current": E2EConfig(
        candidates=search.RETRIEVAL_CANDIDATES,
        top_k=search.RERANK_TOP_K,
        fts=None,
    ),
    "tuned_fts_or": dataclasses.replace(TUNED, fts="fts_or"),
    "tuned_haiku": dataclasses.replace(TUNED, model="claude-haiku-4-5-20251001"),
}


def _fulltext(method: str) -> Callable[..., list[str]]:
    """`search.fulltext_search` with the given tsquery construction."""

    def fulltext_search(
        db: Session, query: str, limit: int, report_ids: list[str] | None
    ) -> list[str]:
        where, params = search._report_filter(report_ids)
        rows = db.execute(
            sql(
                f"SELECT c.id FROM chunks c, (SELECT {FTS_SQL[method]} AS q) t "
                f"WHERE c.tsv @@ t.q {where} ORDER BY ts_rank_cd(c.tsv, t.q) DESC LIMIT :limit"
            ),
            {"q": query, "limit": limit, **params},
        )
        return [row[0] for row in rows]

    return fulltext_search


def provider_for(cfg: E2EConfig) -> LLMProvider:
    settings = get_settings()
    if cfg.model:
        settings = settings.model_copy(update={"anthropic_model": cfg.model})
    provider = AnthropicProvider(settings)
    instrument(provider)
    return provider


def run_question(q: Question, provider: LLMProvider, cfg: E2EConfig) -> dict[str, Any]:
    history = [Message(role=t["role"], content=t["content"]) for t in q.history]
    stages: dict[str, float] = {}
    answer: dict[str, Any] = {}
    mark = USAGE.mark()
    start = last = time.perf_counter()
    current = "start"
    with session_for(cfg.variant) as db:
        for event in run_chat(db, provider, q.question, history, None):
            now = time.perf_counter()
            stages[current] = round(now - last, 2)
            last = now
            if event.type == "stage":
                current = event.data["name"]
            elif event.type == "answer":
                answer = event.data
            elif event.type == "error":
                answer = {"status": "error", "answer": event.data["message"], "citations": []}
        ids = report_keys(db)
        retrieved = [
            {"report_id": c.report_id, "page": c.page, "content": c.content}
            for c in _chunks(db, answer.get("meta", {}).get("retrieved_chunks", []))
        ]
    stages["total"] = round(time.perf_counter() - start, 2)
    return {
        "id": q.id,
        "answer": answer,
        "stages": stages,
        "ids": ids,
        "mark": mark,
        "retrieved": retrieved,
    }


def _chunks(db: Session, chunk_ids: list[str]) -> list[Chunk]:
    by_id = {c.id: c for c in db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


def retrieval_metrics(q: Question, run: dict[str, Any]) -> dict[str, Any]:
    """What the real pipeline retrieved for this question: did a chunk land on a gold page,
    are all key facts in the retrieved text of the gold reports."""
    if not q.answerable:
        return {"retrieved_gold_page": None, "retrieved_facts": None}
    ids = run["ids"]
    gold_reports = {ids[key] for key in q.reports}
    gold_pages = {(ids[e["report"]], e["page"]) for e in q.evidence}
    retrieved = run["retrieved"]
    text = " ".join(normalize(c["content"]) for c in retrieved if c["report_id"] in gold_reports)
    return {
        "retrieved_gold_page": any((c["report_id"], c["page"]) in gold_pages for c in retrieved),
        "retrieved_facts": all(normalize(f) in text for f in q.key_facts) if q.key_facts else None,
    }


class Judgement(BaseModel):
    correctness: Literal["correct", "partial", "incorrect", "abstained"]
    grounded: bool
    explanation: str


JUDGE_SYSTEM = """You grade answers of a question-answering system over company annual reports.
You get the question, the gold answer written by an analyst, the system's answer, its status
and the quotes it cited (already checked to appear verbatim in the report).
- correctness: "correct" if the answer states the gold answer's key figures/facts (allow
  rounding and unit rewrites such as 2,519 million = 2.5 billion, and extra correct detail);
  "partial" if it gets part of it right or hedges; "incorrect" if it states a wrong figure or
  fact; "abstained" if it says the report does not state it. For a question whose gold answer
  says it is not stated in the reports, "correct" means the system abstained or clearly said
  it is not disclosed, "incorrect" means it gave a specific figure as the answer.
- grounded: true if every figure in the answer is supported by the cited quotes (or the
  answer makes no factual claim). False if a figure appears in no quote.
Be strict on numbers and periods (fiscal year), lenient on wording."""


def judge(judge_provider: LLMProvider, q: Question, answer: dict[str, Any]) -> Judgement:
    quotes = "\n".join(
        f"- [{c['verification']}] {c.get('company', '')} {c.get('fiscal_year', '')} "
        f"p{c.get('page', '?')}: {c['quote']}"
        for c in answer.get("citations", [])
    )
    gold = q.answer or "Not stated in the reports."
    user = (
        f"Question: {q.question}\n"
        + (f"Earlier conversation: {q.history}\n" if q.history else "")
        + f"Gold answer: {gold}\nNotes: {q.notes}\n\n"
        f"System status: {answer.get('status')}\nSystem answer: {answer.get('answer')}\n"
        f"Cited quotes:\n{quotes or '(none)'}"
    )
    return judge_provider.complete_json(
        system=JUDGE_SYSTEM, user=user, schema=Judgement, max_tokens=800
    )


def deterministic(q: Question, answer: dict[str, Any], ids: dict[str, str]) -> dict[str, Any]:
    citations = answer.get("citations", [])
    gold_pages = {(ids.get(e["report"]), e["page"]) for e in q.evidence}
    text = normalize(answer.get("answer", ""))
    trusted = [c for c in citations if c["verification"] in ("verified", "fuzzy")]
    return {
        "status": answer.get("status"),
        "abstained": answer.get("status") == "not_found",
        "facts_in_answer": (
            all(normalize(f) in text for f in q.key_facts) if q.key_facts else None
        ),
        "citations": len(citations),
        "verified": sum(c["verification"] == "verified" for c in citations),
        "fuzzy": sum(c["verification"] == "fuzzy" for c in citations),
        "failed": sum(c["verification"] == "failed" for c in citations),
        "cited_gold_page": (
            any((c.get("report_id"), c.get("page")) in gold_pages for c in trusted)
            if q.evidence
            else None
        ),
        "skipped_llm": bool(answer.get("meta", {}).get("skipped_llm")),
    }


def evaluate(name: str, limit: int | None, workers: int) -> dict[str, Any]:
    cfg = CONFIGS[name]
    questions = load_questions()[:limit]
    provider = provider_for(cfg)
    judge_provider = AnthropicProvider(
        get_settings().model_copy(update={"anthropic_model": JUDGE_MODEL})
    )
    # the pipeline calls `retrieve` with the module defaults; hand it one with this
    # configuration's limits bound instead
    patches: list[mock._patch] = [
        mock.patch.object(
            pipeline,
            "retrieve",
            functools.partial(search.retrieve, candidates=cfg.candidates, top_k=cfg.top_k),
        )
    ]
    if cfg.fts:
        patches.append(mock.patch.object(search, "fulltext_search", _fulltext(cfg.fts)))
    with ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        with ThreadPoolExecutor(workers) as pool:
            runs = list(pool.map(lambda q: run_question(q, provider, cfg), questions))
    # Usage is attributed per question only roughly when questions run in parallel, so cost
    # is reported per run; the judge is not instrumented and not counted.
    usage = USAGE.summary()
    with ThreadPoolExecutor(WORKERS) as pool:
        judgements = list(
            pool.map(
                lambda pair: judge(judge_provider, *pair),
                zip(questions, (r["answer"] for r in runs), strict=True),
            )
        )
    rows = []
    for q, run, verdict in zip(questions, runs, judgements, strict=True):
        rows.append(
            {
                "id": q.id,
                "type": q.type,
                "modality": q.modality,
                "question": q.question,
                "gold": q.answer,
                "answer": run["answer"].get("answer"),
                "query": run["answer"].get("meta", {}).get("standalone_question"),
                "best_score": run["answer"].get("meta", {}).get("best_score"),
                **deterministic(q, run["answer"], run["ids"]),
                **retrieval_metrics(q, run),
                "retrieved_chunks": len(run["retrieved"]),
                "judge": verdict.model_dump(),
                "cited": [
                    {
                        k: c.get(k)
                        for k in ("verification", "company", "fiscal_year", "page", "quote")
                    }
                    for c in run["answer"].get("citations", [])
                ],
                "stages": run["stages"],
            }
        )
    return {
        "config": name,
        "settings": cfg.__dict__,
        "usage": usage,
        **aggregate(rows),
        "rows": rows,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]

    def share(items: list[dict], pred: Callable[[dict], Any]) -> float | None:
        return round(sum(1 for r in items if pred(r)) / len(items), 3) if items else None

    citations = sum(r["citations"] for r in rows)
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == t]
        by_type[t] = {
            "n": len(group),
            "correct": share(group, lambda r: r["judge"]["correctness"] == "correct"),
            "correct_or_partial": share(
                group, lambda r: r["judge"]["correctness"] in ("correct", "partial")
            ),
            "retrieved_facts": share(
                [r for r in group if r.get("retrieved_facts") is not None],
                lambda r: r["retrieved_facts"],
            ),
        }
    return {
        "summary": {
            "n": len(rows),
            "answer_correct": share(answerable, lambda r: r["judge"]["correctness"] == "correct"),
            "answer_correct_or_partial": share(
                answerable, lambda r: r["judge"]["correctness"] in ("correct", "partial")
            ),
            "answer_incorrect": share(
                answerable, lambda r: r["judge"]["correctness"] == "incorrect"
            ),
            "false_abstention": share(answerable, lambda r: r["abstained"]),
            "facts_in_answer": share(
                [r for r in answerable if r["facts_in_answer"] is not None],
                lambda r: r["facts_in_answer"],
            ),
            "cited_gold_page": share(answerable, lambda r: r["cited_gold_page"]),
            "retrieved_gold_page": share(answerable, lambda r: r["retrieved_gold_page"]),
            "retrieved_facts": share(
                [r for r in answerable if r["retrieved_facts"] is not None],
                lambda r: r["retrieved_facts"],
            ),
            "unanswerable_abstained": share(unanswerable, lambda r: r["abstained"]),
            "unanswerable_correct": share(
                unanswerable, lambda r: r["judge"]["correctness"] == "correct"
            ),
            "hallucinated_on_unanswerable": share(
                unanswerable, lambda r: r["judge"]["correctness"] == "incorrect"
            ),
            "grounded": share(rows, lambda r: r["judge"]["grounded"]),
            "citations_per_answer": round(citations / len(rows), 2),
            "citation_verified": round(sum(r["verified"] for r in rows) / max(citations, 1), 3),
            "citation_fuzzy": round(sum(r["fuzzy"] for r in rows) / max(citations, 1), 3),
            "citation_failed": round(sum(r["failed"] for r in rows) / max(citations, 1), 3),
            "latency_p50_s": round(statistics.median(r["stages"]["total"] for r in rows), 2),
            "latency_p90_s": round(
                sorted(r["stages"]["total"] for r in rows)[int(0.9 * (len(rows) - 1))], 2
            ),
            "stage_p50_s": {
                stage: round(statistics.median(r["stages"].get(stage, 0) for r in rows), 2)
                for stage in ("analyzing", "retrieving", "generating")
            },
        },
        "by_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", choices=sorted(CONFIGS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1, help="questions run in parallel")
    parser.add_argument("--tag", default="", help="suffix for the result file (repeat runs)")
    args = parser.parse_args()
    require_reranker().score("warm-up", ["warm-up"])  # load both models outside the timings
    get_embedder().embed_query("warm-up")
    for name in args.configs:
        USAGE.calls.clear()
        result = evaluate(name, args.limit, args.workers)
        print(name, result["summary"], result["usage"])
        print(write_json(f"e2e_{name}{args.tag}.json", result))


if __name__ == "__main__":
    main()
