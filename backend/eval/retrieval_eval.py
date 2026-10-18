"""Retrieval evaluation on the gold questions: what the chat retrieves, scored against the
gold evidence pages.

Every question goes through the chat's own retrieval (`_retrieve_all`: the rewrite's
sub-queries, hybrid search, RRF, reranker, score floor and top-k per sub-query, merged), so
the result is what the app retrieves today and how long it takes. The query rewrites (small
model) are cached, so runs with different parameters search the same queries; everything
else runs live.

    uv run python -m eval.retrieval_eval                            # the app's settings
    uv run python -m eval.retrieval_eval --candidates 40 --top-k 8  # an experiment
    uv run python -m eval.retrieval_eval --search vector --no-rerank
    uv run python -m eval.retrieval_eval --max-subqueries 0         # no query decomposition
"""

import argparse
import dataclasses
import hashlib
import json
import statistics
import time
from typing import Any

import app.chat.pipeline as pipeline
from app.chat import prompts
from app.chat.schemas import Query
from app.db import SessionLocal
from app.llm import get_provider
from app.models import Chunk, Message
from app.retrieval.embedder import get_embedder
from app.retrieval.search import Hit
from app.textutil import normalize
from eval.common import (
    CACHE_DIR,
    Question,
    RetrievalParams,
    add_retrieval_args,
    instrument,
    load_questions,
    patched_retrieval,
    report_keys,
    require_reranker,
    retrieval_params,
    write_json,
)


def rewrites(questions: list[Question]) -> dict[str, Query]:
    """The rewrite of every question, as the chat produces it. Cached under a key derived
    from the rewrite prompt and schema, so a prompt change never replays stale rewrites."""
    signature = hashlib.sha1(
        (prompts.QUERY_REWRITE + json.dumps(Query.model_json_schema(), sort_keys=True)).encode()
    ).hexdigest()[:10]
    path = CACHE_DIR / f"rewrites_{signature}.json"
    cache: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    provider = get_provider()
    instrument(provider)
    out = {}
    for q in questions:
        key = f"{q.question}|{json.dumps(q.history)}"
        if key not in cache:
            history = [Message(role=t["role"], content=t["content"]) for t in q.history]
            cache[key] = pipeline._rewrite_query(provider, q.question, history).model_dump()
        out[q.id] = Query(**cache[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return out


# --- metrics --------------------------------------------------------------------------------


def score_hits(q: Question, hits: list[Hit], ids: dict[str, str]) -> dict[str, Any]:
    """Per-question metrics against the gold evidence pages. `facts` is the lenient check
    that every key fact appears somewhere in the retrieved text of the right reports (a
    figure is often restated on several pages)."""
    gold_reports = {ids[key] for key in q.reports if key in ids}
    gold_pages = {(ids[e["report"]], e["page"]) for e in q.evidence if e["report"] in ids}
    located = [(hit.chunk.report_id, hit.chunk.page) for hit in hits]
    first = next((rank for rank, loc in enumerate(located, 1) if loc in gold_pages), None)
    context = " ".join(
        normalize(hit.chunk.content) for hit in hits if hit.chunk.report_id in gold_reports
    )
    return {
        "page_hit": first is not None,
        "mrr": 1 / first if first else 0.0,
        "facts": all(normalize(f) in context for f in q.key_facts) if q.key_facts else None,
        # share of the retrieved chunks from the right company and year
        "report_precision": (
            sum(loc[0] in gold_reports for loc in located) / len(located) if located else 0.0
        ),
        # share of the gold reports with at least one chunk: a comparison needs both sides
        "report_recall": (
            len(gold_reports & {loc[0] for loc in located}) / len(gold_reports)
            if gold_reports
            else None
        ),
        "best_score": hits[0].score if hits else 0.0,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(values), 3) if values else None

    return {
        "n": len(rows),
        "page_hit": mean("page_hit"),
        "mrr": mean("mrr"),
        "facts": mean("facts"),
        "report_precision": mean("report_precision"),
        "report_recall": mean("report_recall"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Over the answerable questions: in all, per question type and per modality (text or
    table); then what the unanswerable ones retrieved, for the abstention floor."""
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    seconds = sorted(r["seconds"] for r in rows)
    summary = aggregate(answerable)
    for group in ("type", "modality"):
        summary[f"by_{group}"] = {
            value: aggregate([r for r in answerable if r[group] == value])
            for value in sorted({r[group] for r in answerable})
        }
    summary |= {
        "subqueries_mean": round(statistics.mean(r["subqueries"] for r in rows), 2),
        "chunks_mean": round(statistics.mean(r["chunks"] for r in rows), 1),
        "seconds_p50": round(statistics.median(seconds), 2),
        "seconds_p90": round(seconds[int(0.9 * (len(seconds) - 1))], 2),
        "nothing_retrieved": [r["id"] for r in rows if r["chunks"] == 0],
        # the floor must stay below every answerable question's best score...
        "answerable_min_best_score": round(
            min((r["best_score"] for r in answerable if r["chunks"]), default=0.0), 3
        ),
        # ...and catches an unanswerable question only when nothing scores above it
        "unanswerable_nothing_retrieved": (
            round(sum(r["chunks"] == 0 for r in unanswerable) / len(unanswerable), 3)
            if unanswerable
            else None
        ),
    }
    return summary


# --- the run --------------------------------------------------------------------------------


def evaluate(params: RetrievalParams, limit: int | None = None) -> dict[str, Any]:
    questions = load_questions(limit)
    queries = rewrites(questions)
    get_embedder()  # load the models before timing; the app loads them at startup too
    if params.rerank:
        require_reranker()
    rows = []
    with SessionLocal() as db, patched_retrieval(params):
        ids = report_keys(db)
        chunks = db.query(Chunk).count()
        for q in questions:
            query = queries[q.id]
            start = time.perf_counter()
            hits = pipeline._retrieve_all(db, query, None)
            seconds = round(time.perf_counter() - start, 2)
            metrics = score_hits(q, hits, ids)
            print(
                f"{q.id}: {len(query.subqueries)} searches, {len(hits)} chunks, {seconds}s, "
                f"gold page {'hit' if metrics['page_hit'] else 'missed'}",
                flush=True,
            )
            rows.append(
                {
                    "id": q.id,
                    "type": q.type,
                    "modality": q.modality,
                    "query": query.standalone_question,
                    "searches": [sub.search_text for sub in query.subqueries],
                    "subqueries": len(query.subqueries),
                    "chunks": len(hits),
                    "seconds": seconds,
                    **metrics,
                    "hits": [
                        f"{hit.report.company} {hit.report.fiscal_year} p{hit.chunk.page} "
                        f"{hit.score:.3f}"
                        for hit in hits
                    ],
                }
            )
    return {
        "params": dataclasses.asdict(params),
        "chunks": chunks,
        "summary": summarize(rows),
        "questions": rows,
    }


def print_summary(result: dict[str, Any]) -> None:
    s = result["summary"]
    print(f"\n== retrieval / {result['params']}")
    print(
        f"{'questions':24} {'n':>3} {'hit@k':>6} {'mrr':>6} {'facts':>6} {'rprec':>6} {'rrec':>6}"
    )

    def line(name: str, m: dict[str, Any]) -> None:
        print(
            f"{name:24} {m['n']:3} {m['page_hit'] or 0:6.3f} {m['mrr'] or 0:6.3f} "
            f"{m['facts'] or 0:6.3f} {m['report_precision'] or 0:6.3f} {m['report_recall'] or 0:6.3f}"
        )

    line("all answerable", s)
    for group in ("by_type", "by_modality"):
        for name, m in s[group].items():
            line(f"  {name}", m)
    print(
        f"{s['subqueries_mean']} subqueries and {s['chunks_mean']} chunks per question; "
        f"{s['seconds_p50']} s p50, {s['seconds_p90']} s p90"
    )
    print(
        f"nothing retrieved for {s['nothing_retrieved'] or 'no question'}; lowest best score "
        f"on an answerable question {s['answerable_min_best_score']}; unanswerable questions "
        f"with nothing retrieved {s['unanswerable_nothing_retrieved']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_retrieval_args(parser)
    parser.add_argument("--limit", type=int, help="first N questions only (a smoke run)")
    parser.add_argument("--tag", help="result file suffix (default: the overridden parameters)")
    args = parser.parse_args()
    params = retrieval_params(args)
    result = evaluate(params, args.limit)
    print_summary(result)
    # an experiment or a partial run never overwrites the result of the app's settings
    suffix = "_" + (args.tag or params.tag()) if args.tag or params.tag() else ""
    if args.limit:
        suffix += f"_first{args.limit}"
    print(write_json(f"retrieval{suffix}.json", result))


if __name__ == "__main__":
    main()
