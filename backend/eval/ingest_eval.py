"""Ingest the eval corpus into the eval database, timing each stage, and score ingestion.

Runs the app's own pipeline stages in the app's order (parse, detect metadata, store pages,
chunk, embed, extract), so the timings and the index are what the app would produce. A
report is ingested once; its timings and LLM usage are kept from `results/ingestion.json`
when the corpus is scored again.

    uv run python -m eval.ingest_eval            # ingest what is missing, then score
    uv run python -m eval.ingest_eval --score    # score only
    uv run python -m eval.ingest_eval --tag openai   # results in ingestion_openai.json
"""

import argparse
import asyncio
import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select

from app.db import SessionLocal
from app.extraction.extractor import extract_report
from app.ingestion import pipeline
from app.llm import get_provider
from app.models import Chunk, Datapoint, Report, ReportStatus
from app.retrieval.embedder import get_embedder
from eval.common import (
    EMBED_USAGE,
    RESULTS_DIR,
    USAGE,
    CorpusDoc,
    instrument,
    instrument_embedder,
    load_corpus,
    timer,
    write_json,
)


async def ingest(doc: CorpusDoc) -> dict[str, Any]:
    """Run the app's stages on one report with a timer around each. A stage that raises
    marks the report failed, as the app does, and the run records the stage and error; the
    stages before it keep their timings."""
    timings: dict[str, float] = {}
    digest = hashlib.sha256(doc.path.read_bytes()).hexdigest()
    async with SessionLocal() as db:
        existing = await db.execute(select(Report.id).where(Report.file_hash == digest))
        if existing.first():
            print(f"{doc.key}: already ingested")
            return {}
        report = Report(company="", fiscal_year=0, filename=doc.path.name, file_hash=digest)
        db.add(report)
        await db.commit()
        state: dict[str, Any] = {}

        async def parse() -> None:
            state["parsed"] = await pipeline._parse(db, report, doc.path)

        async def detect_metadata() -> None:
            await pipeline._detect_metadata(db, report, state["parsed"])

        async def chunk_and_store() -> None:
            await pipeline._set_progress(db, report, ReportStatus.CHUNKING)
            pipeline._store_pages(db, report, state["parsed"])
            state["chunks"] = pipeline._chunk(db, report, state["parsed"])

        async def embed() -> None:
            await pipeline._set_progress(db, report, ReportStatus.EMBEDDING)
            await pipeline._embed(db, report, state["chunks"])

        async def extract() -> None:
            await pipeline._set_progress(db, report, ReportStatus.EXTRACTING)
            await extract_report(db, report, get_provider())

        stages: list[Callable[[], Awaitable[None]]] = [
            parse,
            detect_metadata,
            chunk_and_store,
            embed,
            extract,
        ]
        mark, embed_mark = USAGE.mark(), EMBED_USAGE.mark()
        run: dict[str, Any] = {}
        for stage in stages:
            try:
                with timer(timings, stage.__name__):
                    await stage()
            except Exception as exc:
                await db.rollback()
                await pipeline._mark_failed(db, report.id, exc)
                run["failed"] = {
                    "stage": stage.__name__,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
                break
        else:
            await pipeline._set_progress(db, report, ReportStatus.READY, fraction=1.0)
        timings["total"] = round(sum(timings.values()), 2)
        run |= {
            "timings": timings,
            "llm": USAGE.summary(mark),
            "embedding": EMBED_USAGE.summary(embed_mark),
        }
        if "parsed" in state:
            run["pages"] = state["parsed"].page_count
    outcome = (
        f"failed at {run['failed']['stage']}: {run['failed']['error']}"
        if "failed" in run
        else "ready"
    )
    print(
        f"{doc.key}: {run.get('pages')} pages, {len(state.get('chunks', []))} chunks, {outcome}, {timings}"
    )
    return run


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    words = [c.word_count for c in chunks]
    return {
        "chunks": len(chunks),
        "table_chunks": sum(c.kind == "table" for c in chunks),
        "words_p50": int(statistics.median(words)) if words else 0,
        "words_max": max(words, default=0),
        "duplicate_chunks": sum(n - 1 for n in Counter(c.content for c in chunks).values()),
    }


async def score(ingest_runs: dict[str, dict]) -> dict[str, Any]:
    """Per report: metadata detection against the corpus, chunk statistics, datapoints
    written, and the stage timings and LLM usage of its ingestion run."""
    out: dict[str, Any] = {"reports": {}}
    async with SessionLocal() as db:
        stored = (await db.execute(select(Report))).scalars()
        by_name = {r.filename: r for r in stored}
        for doc in load_corpus():
            report = by_name.get(doc.path.name)
            if report is None:
                continue
            chunks = list(
                (await db.execute(select(Chunk).where(Chunk.report_id == report.id))).scalars()
            )
            datapoints = (
                await db.execute(select(Datapoint).where(Datapoint.report_id == report.id))
            ).scalars()
            out["reports"][doc.key] = {
                "metadata": {
                    "detected": [report.company, report.fiscal_year],
                    "company_ok": report.company.lower() == doc.company.lower(),
                    "year_ok": report.fiscal_year == doc.fiscal_year,
                },
                "chunks": chunk_stats(chunks),
                "datapoints": dict(Counter(d.key for d in datapoints)),
                **ingest_runs.get(doc.key, {}),
            }
    reports = list(out["reports"].values())
    out["summary"] = {
        "reports": len(reports),
        "failed": [key for key, r in out["reports"].items() if "failed" in r],
        "metadata_accuracy": round(
            sum(r["metadata"]["company_ok"] and r["metadata"]["year_ok"] for r in reports)
            / max(len(reports), 1),
            3,
        ),
        "chunks": sum(r["chunks"]["chunks"] for r in reports),
    }
    timed = [r for r in reports if "timings" in r]  # reports ingested by this script
    if timed:
        total = sum(r["timings"]["total"] for r in timed)
        llm_cost = sum(r["llm"]["cost_usd"] for r in timed)
        embedding_cost = sum(r.get("embedding", {}).get("cost_usd", 0.0) for r in timed)
        out["summary"] |= {
            "timed_reports": len(timed),
            "total_seconds": round(total, 1),
            "stage_share": {
                stage: round(sum(r["timings"].get(stage, 0) for r in timed) / total, 3)
                for stage in ("parse", "detect_metadata", "chunk_and_store", "embed", "extract")
            },
            "llm_cost_usd": round(llm_cost, 4),
            "embedding_tokens": sum(r.get("embedding", {}).get("tokens", 0) for r in timed),
            "embedding_cost_usd": round(embedding_cost, 4),
            "cost_usd": round(llm_cost + embedding_cost, 4),
        }
    return out


def _previous_runs(name: str) -> dict[str, dict]:
    """Timings and usage of earlier runs: a report is ingested once, scored many times."""
    path = RESULTS_DIR / name
    if not path.exists():
        return {}
    reports = json.loads(path.read_text())["reports"]
    return {
        key: {f: run[f] for f in ("pages", "timings", "llm", "embedding", "failed") if f in run}
        for key, run in reports.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score", action="store_true", help="score only, do not ingest")
    parser.add_argument("--tag", help="result file suffix: ingestion_<tag>.json")
    args = parser.parse_args()
    asyncio.run(_main(args.score, args.tag))


async def _main(score_only: bool, tag: str | None) -> None:
    name = f"ingestion_{tag}.json" if tag else "ingestion.json"
    instrument(get_provider())
    instrument_embedder(get_embedder())
    runs = _previous_runs(name)
    if not score_only:
        for doc in load_corpus():
            run = await ingest(doc)
            if run:
                runs[doc.key] = run
                write_json(name, await score(runs))  # a crash later loses nothing
    result = await score(runs)
    print(result["summary"])
    print(write_json(name, result))


if __name__ == "__main__":
    main()
