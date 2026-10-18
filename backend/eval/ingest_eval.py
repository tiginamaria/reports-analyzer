"""Ingest the eval corpus into the eval database, timing each stage, and score ingestion.

Runs the app's own pipeline stages in the app's order (parse, detect metadata, store pages,
chunk, embed, extract) so the timings and the index are what the app would produce. Parsed
documents are cached for the chunking experiments in `variants.py`.

    uv run python -m eval.ingest_eval            # ingest what is missing, then score
    uv run python -m eval.ingest_eval --score    # score only
"""

import argparse
import hashlib
import json
import pickle
import statistics
from collections import Counter
from pathlib import Path

from app.db import SessionLocal
from app.extraction.extractor import extract_report
from app.ingestion import pipeline
from app.ingestion.parser import ParsedDocument
from app.llm import get_provider
from app.models import Chunk, Datapoint, Report, ReportStatus
from app.retrieval.embedder import get_embedder
from app.textutil import word_count
from eval.common import (
    CACHE_DIR,
    RESULTS_DIR,
    USAGE,
    CorpusDoc,
    instrument,
    load_corpus,
    timer,
    write_json,
)

EMPTY_PAGE_CHARS = 200  # a page with less text than this is a cover, divider or image-only page
BOILERPLATE_PAGE_SHARE = 0.2  # a line repeated on this share of pages is a running header


def parsed_cache_path(key: str) -> Path:
    return CACHE_DIR / "parsed" / f"{key}.pkl"


def ingest(doc: CorpusDoc) -> dict:
    timings: dict[str, float] = {}
    digest = hashlib.sha256(doc.path.read_bytes()).hexdigest()
    with SessionLocal() as db:
        if db.query(Report).filter(Report.file_hash == digest).first():
            print(f"{doc.key}: already ingested")
            return {}
        report = Report(company="", fiscal_year=0, filename=doc.path.name, file_hash=digest)
        db.add(report)
        db.commit()
        mark = USAGE.mark()
        with timer(timings, "parse"):
            parsed = pipeline._parse(db, report, doc.path)
        with timer(timings, "detect_metadata"):
            pipeline._detect_metadata(report, parsed)
        with timer(timings, "chunk_and_store"):
            pipeline._set_progress(db, report, ReportStatus.CHUNKING)
            pipeline._store_pages(db, report, parsed)
            chunks = pipeline._chunk(db, report, parsed)
        with timer(timings, "embed"):
            pipeline._set_progress(db, report, ReportStatus.EMBEDDING)
            pipeline._embed(db, report, chunks)
        with timer(timings, "extract"):
            pipeline._set_progress(db, report, ReportStatus.EXTRACTING)
            extract_report(db, report, get_provider())
        pipeline._set_progress(db, report, ReportStatus.READY, fraction=1.0)
        timings["total"] = round(sum(timings.values()), 2)
        llm = USAGE.summary(mark)
    path = parsed_cache_path(doc.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(parsed))
    print(f"{doc.key}: {parsed.page_count} pages, {len(chunks)} chunks, {timings}")
    return {"timings": timings, "llm": llm}


def load_parsed(key: str) -> ParsedDocument:
    return pickle.loads(parsed_cache_path(key).read_bytes())


def parse_stats(parsed: ParsedDocument) -> dict:
    """Parse quality signals that need no labels."""
    chars = [len(page) for page in parsed.pages]
    page_words = sum(word_count(page) for page in parsed.pages)
    body_words = sum(word_count(body) for body in parsed.bodies)
    table_words = sum(word_count(table.text) for table in parsed.tables)

    line_pages: Counter[str] = Counter()
    for page in parsed.pages:
        lines = {line.strip() for line in page.splitlines() if len(line.strip()) > 3}
        line_pages.update(lines)
    threshold = BOILERPLATE_PAGE_SHARE * parsed.page_count
    boilerplate = sorted((line for line, n in line_pages.items() if n >= threshold), key=len)
    return {
        "pages": parsed.page_count,
        "empty_pages": sum(c < EMPTY_PAGE_CHARS for c in chars),
        "median_chars_per_page": int(statistics.median(chars)),
        "tables": len(parsed.tables),
        "pages_with_tables": len({t.page for t in parsed.tables}),
        # body + tables vs the full page: < 1 means text lost with table regions, > 1 means
        # text indexed twice (overlapping tables)
        "indexed_word_ratio": round((body_words + table_words) / max(page_words, 1), 3),
        "boilerplate_lines": boilerplate[:15],
        "boilerplate_line_count": len(boilerplate),
    }


def chunk_stats(chunks: list[Chunk]) -> dict:
    words = sorted(c.word_count for c in chunks)
    texts = Counter(c.content for c in chunks)
    tables = [c.word_count for c in chunks if c.kind == "table"]

    def pct(p: float) -> int:
        return words[min(len(words) - 1, int(p * len(words)))]

    return {
        "chunks": len(chunks),
        "table_chunks": len(tables),
        "words_p10_p50_p90_max": [pct(0.1), pct(0.5), pct(0.9), words[-1]],
        "share_under_50_words": round(sum(w < 50 for w in words) / len(words), 3),
        "table_words_p50_max": [int(statistics.median(tables)), max(tables)] if tables else None,
        "duplicate_chunks": sum(n - 1 for n in texts.values() if n > 1),
    }


def score(ingest_runs: dict[str, dict]) -> dict:
    corpus = load_corpus()
    out: dict = {"reports": {}}
    with SessionLocal() as db:
        by_name = {r.filename: r for r in db.query(Report).all()}
        for doc in corpus:
            report = by_name.get(doc.path.name)
            if report is None:
                continue
            chunks = db.query(Chunk).filter(Chunk.report_id == report.id).all()
            datapoints = db.query(Datapoint).filter(Datapoint.report_id == report.id).all()
            out["reports"][doc.key] = {
                "metadata": {
                    "detected": [report.company, report.fiscal_year],
                    "company_ok": report.company.lower() == doc.company.lower(),
                    "year_ok": report.fiscal_year == doc.fiscal_year,
                },
                "parse": parse_stats(load_parsed(doc.key)),
                "chunks": chunk_stats(chunks),
                "datapoints": dict(Counter(d.key for d in datapoints)),
                **ingest_runs.get(doc.key, {}),
            }
    reports = out["reports"].values()
    out["metadata_accuracy"] = round(
        sum(r["metadata"]["company_ok"] and r["metadata"]["year_ok"] for r in reports)
        / max(len(out["reports"]), 1),
        3,
    )
    return out


def _previous_runs() -> dict[str, dict]:
    """Timings and LLM usage of earlier runs: a report is ingested once, scored many times."""
    path = RESULTS_DIR / "ingestion.json"
    if not path.exists():
        return {}
    reports = json.loads(path.read_text())["reports"]
    return {
        key: {field: run[field] for field in ("timings", "llm") if field in run}
        for key, run in reports.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true", help="score only, do not ingest")
    args = parser.parse_args()
    instrument(get_provider())
    get_embedder()
    runs: dict[str, dict] = {}
    if not args.score:
        for doc in load_corpus():
            run = ingest(doc)
            if run:
                runs[doc.key] = run
    result = score({**_previous_runs(), **runs})
    print(write_json("ingestion.json", result))


if __name__ == "__main__":
    main()
