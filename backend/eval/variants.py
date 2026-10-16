"""Build alternative indexes for the chunking experiments, one database per variant.

Reports and pages are copied from the base eval database (so report ids, metadata and the
verification source stay identical), then the cached parse is re-chunked with the variant's
parameters and re-embedded. No LLM calls.

    uv run python -m eval.variants words200 words600 noheader
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy import text as sql

from app.db import SessionLocal
from app.ingestion.chunker import chunk_document, contextual_header
from app.ingestion.pipeline import _chunk_id_prefix
from app.models import Chunk, Page, Report
from app.retrieval.embedder import get_embedder
from eval import EVAL_DATABASE_URL
from eval.common import BACKEND_DIR, load_corpus, write_json
from eval.ingest_eval import chunk_stats, load_parsed
from eval.retrieval_eval import session_for, variant_url

EMBED_BATCH = 32


@dataclass(frozen=True)
class Variant:
    max_words: int = 400
    overlap_words: int = 60
    header: bool = True


VARIANTS = {
    "words200": Variant(200, 30),
    "words600": Variant(600, 90),
    "noheader": Variant(header=False),
}


def create_database(name: str) -> None:
    admin = create_engine(
        EVAL_DATABASE_URL.rsplit("/", 1)[0] + "/postgres", isolation_level="AUTOCOMMIT"
    )
    with admin.connect() as conn:
        conn.execute(sql(f'DROP DATABASE IF EXISTS "reports_eval_{name}"'))
        conn.execute(sql(f'CREATE DATABASE "reports_eval_{name}"'))
    env = {**os.environ, "DATABASE_URL": variant_url(name)}
    subprocess.run(["alembic", "upgrade", "head"], cwd=BACKEND_DIR, env=env, check=True)


def build(name: str, variant: Variant) -> dict:
    create_database(name)
    keys = {doc.path.name: doc.key for doc in load_corpus()}
    embedder = get_embedder()
    stats: dict = {"variant": variant.__dict__, "reports": {}}
    start = time.perf_counter()
    with SessionLocal() as base, session_for(name) as target:
        for report in base.query(Report).all():
            pages = base.query(Page).filter(Page.report_id == report.id).all()
            target.merge(
                Report(**{c.key: getattr(report, c.key) for c in Report.__table__.columns})
            )
            target.add_all(
                Page(report_id=p.report_id, page_no=p.page_no, text=p.text) for p in pages
            )
            drafts = chunk_document(
                load_parsed(keys[report.filename]), variant.max_words, variant.overlap_words
            )
            prefix = _chunk_id_prefix(report)
            chunks = [
                Chunk(
                    id=f"{prefix}-p{d.page}-c{n}",
                    report_id=report.id,
                    page=d.page,
                    kind=d.kind,
                    content=d.content,
                    word_count=d.word_count,
                )
                for n, d in enumerate(drafts)
            ]
            for i in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[i : i + EMBED_BATCH]
                texts = [
                    f"{contextual_header(report.company, report.fiscal_year, c.page)}\n{c.content}"
                    if variant.header
                    else c.content
                    for c in batch
                ]
                for chunk, vector in zip(batch, embedder.embed_documents(texts), strict=True):
                    chunk.embedding = vector
            target.add_all(chunks)
            target.commit()
            stats["reports"][keys[report.filename]] = chunk_stats(chunks)
            print(f"{name}: {report.company} {report.fiscal_year}: {len(chunks)} chunks")
    stats["build_seconds"] = round(time.perf_counter() - start, 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=sorted(VARIANTS))
    args = parser.parse_args()
    for name in args.names:
        print(write_json(f"variant_{name}.json", build(name, VARIANTS[name])))


if __name__ == "__main__":
    main()
