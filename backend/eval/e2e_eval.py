"""End-to-end evaluation of the chat on the gold questions.

Runs the real `run_chat` (rewrite, retrieval, rerank, generation, verification) on every
question with the app's settings, or with the retrieval parameters and the answer model
overridden for an experiment. Answers are graded deterministically (citation verification,
cited page among the gold pages, whether retrieval reached a gold page) and by an LLM judge
(correctness against the gold answer, groundedness in the cited quotes), a different and
stronger model than the generator.

    uv run python -m eval.e2e_eval                                  # the app's settings
    uv run python -m eval.e2e_eval --candidates 40 --top-k 8        # a retrieval experiment
    uv run python -m eval.e2e_eval --model claude-haiku-4-5-20251001 --tag haiku
    uv run python -m eval.e2e_eval --tag openai                     # keep a profile's result apart
    uv run python -m eval.e2e_eval --limit 10 --workers 4           # a smoke run
"""

import argparse
import dataclasses
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.chat.pipeline import run_chat
from app.db import SessionLocal
from app.llm import LLMProvider
from app.models import Chunk, Message
from app.retrieval.embedder import get_embedder
from eval.common import (
    USAGE,
    Question,
    RetrievalParams,
    add_retrieval_args,
    instrument,
    judge_provider,
    load_questions,
    patched_retrieval,
    provider_for,
    report_keys,
    require_reranker,
    retrieval_params,
    write_json,
)

JUDGE_WORKERS = 4  # judge calls; the chat runs use --workers (1 for honest latency)


# --- running the chat -----------------------------------------------------------------------


def run_question(q: Question, provider: LLMProvider) -> dict[str, Any]:
    history = [Message(role=t["role"], content=t["content"]) for t in q.history]
    stages: dict[str, float] = {}
    answer: dict[str, Any] = {}
    start = last = time.perf_counter()
    current = "start"
    with SessionLocal() as db:
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
        retrieved = _chunks(db, answer.get("meta", {}).get("retrieved_chunks", []))
    stages["total"] = round(time.perf_counter() - start, 2)
    return {
        "answer": answer,
        "stages": stages,
        "ids": ids,
        "retrieved": [(c.report_id, c.page) for c in retrieved],
    }


def _chunks(db: Session, chunk_ids: list[str]) -> list[Chunk]:
    by_id = {c.id: c for c in db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


# --- grading --------------------------------------------------------------------------------


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


def judge(judge: LLMProvider, q: Question, answer: dict[str, Any]) -> Judgement:
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
    return judge.complete_json(system=JUDGE_SYSTEM, user=user, schema=Judgement, max_tokens=800)


def deterministic(q: Question, run: dict[str, Any]) -> dict[str, Any]:
    """Checks that need no model: the answer's status, its citations' verification, whether
    a trusted citation is on a gold page, and whether retrieval reached one at all (a wrong
    answer with a gold page retrieved is a generation failure, without one a retrieval
    failure)."""
    answer, ids = run["answer"], run["ids"]
    citations = answer.get("citations", [])
    gold_pages = {(ids.get(e["report"]), e["page"]) for e in q.evidence}
    trusted = [c for c in citations if c["verification"] in ("verified", "fuzzy")]
    return {
        "status": answer.get("status"),
        "abstained": answer.get("status") == "not_found",
        "skipped_llm": bool(answer.get("meta", {}).get("skipped_llm")),
        "citations": len(citations),
        "verified": sum(c["verification"] == "verified" for c in citations),
        "fuzzy": sum(c["verification"] == "fuzzy" for c in citations),
        "failed": sum(c["verification"] == "failed" for c in citations),
        "cited_gold_page": (
            any((c.get("report_id"), c.get("page")) in gold_pages for c in trusted)
            if q.answerable
            else None
        ),
        "retrieved_gold_page": (
            any(loc in gold_pages for loc in run["retrieved"]) if q.answerable else None
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]

    def share(items: list[dict], pred: Callable[[dict], Any]) -> float | None:
        return round(sum(1 for r in items if pred(r)) / len(items), 3) if items else None

    def correct(r: dict) -> bool:
        return r["judge"]["correctness"] == "correct"

    def correct_or_partial(r: dict) -> bool:
        return r["judge"]["correctness"] in ("correct", "partial")

    citations = sum(r["citations"] for r in rows)
    totals = sorted(r["stages"]["total"] for r in rows)
    summary = {
        "n": len(rows),
        "answer_correct": share(answerable, correct),
        "answer_correct_or_partial": share(answerable, correct_or_partial),
        "answer_incorrect": share(answerable, lambda r: r["judge"]["correctness"] == "incorrect"),
        "false_abstention": share(answerable, lambda r: r["abstained"]),
        "cited_gold_page": share(answerable, lambda r: r["cited_gold_page"]),
        "retrieved_gold_page": share(answerable, lambda r: r["retrieved_gold_page"]),
        "unanswerable_correct": share(unanswerable, correct),
        "hallucinated_on_unanswerable": share(
            unanswerable, lambda r: r["judge"]["correctness"] == "incorrect"
        ),
        "grounded": share(rows, lambda r: r["judge"]["grounded"]),
        "citations_per_answer": round(citations / len(rows), 2),
        "citation_verified": round(sum(r["verified"] for r in rows) / max(citations, 1), 3),
        "citation_fuzzy": round(sum(r["fuzzy"] for r in rows) / max(citations, 1), 3),
        "citation_failed": round(sum(r["failed"] for r in rows) / max(citations, 1), 3),
        "latency_p50_s": round(statistics.median(totals), 2),
        "latency_p90_s": round(totals[int(0.9 * (len(totals) - 1))], 2),
        "stage_p50_s": {
            stage: round(statistics.median(r["stages"].get(stage, 0) for r in rows), 2)
            for stage in ("analyzing", "retrieving", "generating")
        },
    }
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == t]
        by_type[t] = {
            "n": len(group),
            "correct": share(group, correct),
            "correct_or_partial": share(group, correct_or_partial),
            "retrieved_gold_page": share(
                [r for r in group if r["retrieved_gold_page"] is not None],
                lambda r: r["retrieved_gold_page"],
            ),
        }
    return {"summary": summary, "by_type": by_type}


# --- the run --------------------------------------------------------------------------------


def evaluate(
    params: RetrievalParams, model: str | None, limit: int | None, workers: int
) -> dict[str, Any]:
    questions = load_questions(limit)
    provider = provider_for(model)
    instrument(provider)
    USAGE.calls.clear()
    with patched_retrieval(params), ThreadPoolExecutor(workers) as pool:
        runs = list(pool.map(lambda q: run_question(q, provider), questions))
    usage = USAGE.summary()  # the judge is not instrumented and not counted
    judge_llm = judge_provider()
    with ThreadPoolExecutor(JUDGE_WORKERS) as pool:
        judgements = list(
            pool.map(
                lambda pair: judge(judge_llm, pair[0], pair[1]["answer"]),
                zip(questions, runs, strict=True),
            )
        )
    rows = [
        {
            "id": q.id,
            "type": q.type,
            "modality": q.modality,
            "question": q.question,
            "gold": q.answer,
            "answer": run["answer"].get("answer"),
            "query": run["answer"].get("meta", {}).get("standalone_question"),
            "best_score": run["answer"].get("meta", {}).get("best_score"),
            **deterministic(q, run),
            "judge": verdict.model_dump(),
            "cited": [
                {k: c.get(k) for k in ("verification", "company", "fiscal_year", "page", "quote")}
                for c in run["answer"].get("citations", [])
            ],
            "stages": run["stages"],
        }
        for q, run, verdict in zip(questions, runs, judgements, strict=True)
    ]
    return {
        "params": {**dataclasses.asdict(params), "model": provider.model_name()},
        "usage": usage,
        **aggregate(rows),
        "rows": rows,
    }


def print_summary(result: dict[str, Any]) -> None:
    s = result["summary"]
    print(f"\n== answers / {result['params']}")
    for key, value in s.items():
        print(f"{key:30} {value}")
    print(
        f"{'by type':30} " + "  ".join(f"{t} {m['correct']}" for t, m in result["by_type"].items())
    )
    u = result["usage"]
    print(
        f"{u['calls']} LLM calls, {u['input_tokens']} in / {u['output_tokens']} out tokens, "
        f"${u['cost_usd']:.3f} (${u['cost_usd'] / max(s['n'], 1):.4f} per question)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_retrieval_args(parser)
    parser.add_argument(
        "--model", help="answer model instead of the configured provider's main model"
    )
    parser.add_argument("--limit", type=int, help="first N questions only (a smoke run)")
    parser.add_argument("--workers", type=int, default=1, help="questions run in parallel")
    parser.add_argument("--tag", help="result file suffix (default: the overridden parameters)")
    args = parser.parse_args()
    params = retrieval_params(args)
    if params.rerank:
        require_reranker().score("warm-up", ["warm-up"])  # load the models outside the timings
    get_embedder().embed_query("warm-up")
    result = evaluate(params, args.model, args.limit, args.workers)
    print_summary(result)
    # an experiment or a partial run never overwrites the result of the app's settings
    tag = args.tag or "_".join(filter(None, [params.tag(), args.model]))
    suffix = f"_{tag}" if tag else ""
    if args.limit:
        suffix += f"_first{args.limit}"
    print(write_json(f"e2e{suffix}.json", result))


if __name__ == "__main__":
    main()
