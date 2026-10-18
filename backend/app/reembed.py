"""Rebuild the vectors of indexed reports with the configured embedding model.

    uv run python -m app.reembed            # reports embedded with another model
    uv run python -m app.reembed --all      # every indexed report
    uv run python -m app.reembed --dry-run  # only list what would be re-embedded

Only reports whose `embedding_model` equals the configured signature are searched, so after
switching `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL` the existing reports are invisible to
chat and extraction until this has run. Each report goes back to `embedding` while its
chunks are re-embedded in batches (progress in the UI as during ingestion) and is `ready`
again at the end; pages, chunks, datapoints and conversations are untouched. A report whose
re-embedding fails is `ready` but unsearchable (its vectors are part old, part new, so its
signature is cleared) until a later run completes it; it is reported at the end.
"""

import argparse
import logging
import sys
import time

from app.config import get_settings
from app.db import SessionLocal
from app.ingestion import cancellation
from app.ingestion.cancellation import IngestionCancelled
from app.ingestion.pipeline import _embed, _set_progress
from app.models import INDEXED_STATUSES, Chunk, Report, ReportStatus

log = logging.getLogger("reembed")


def reembed_report(report_id: str) -> int:
    """Re-embed one report's chunks; returns how many. On failure the report is `ready`
    again but without an embedding signature: some chunks already hold new vectors, so it
    must not be searched until a later run finishes it."""
    with SessionLocal() as db:
        report = db.get(Report, report_id)
        if report is None:
            raise IngestionCancelled(report_id)
        cancellation.begin(report_id)
        chunks = db.query(Chunk).filter(Chunk.report_id == report.id).order_by(Chunk.id).all()
        try:
            _set_progress(db, report, ReportStatus.EMBEDDING, total=len(chunks))
            _embed(db, report, chunks)
            _set_progress(db, report, ReportStatus.READY, fraction=1.0)
        except IngestionCancelled:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            report.embedding_model = None
            _set_progress(db, report, ReportStatus.READY, fraction=1.0)
            raise
        finally:
            cancellation.clear(report_id)
        return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="also reports already on this model")
    parser.add_argument("--dry-run", action="store_true", help="list the reports, change nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    signature = get_settings().embedding_signature
    with SessionLocal() as db:
        query = db.query(Report).filter(Report.status.in_(INDEXED_STATUSES))
        if not args.all:
            query = query.filter(
                (Report.embedding_model != signature) | (Report.embedding_model.is_(None))
            )
        targets = [(r.id, f"{r.company} {r.fiscal_year}", r.embedding_model) for r in query.all()]

    if not targets:
        print(f"Every indexed report already has {signature} vectors.")
        return 0
    print(f"Re-embedding {len(targets)} report(s) with {signature}:")
    for _, label, current in targets:
        print(f"  {label} (currently {current or 'none'})")
    if args.dry_run:
        return 0

    failures = []
    for report_id, label, _ in targets:
        started = time.perf_counter()
        try:
            count = reembed_report(report_id)
        except IngestionCancelled:
            print(f"  {label}: deleted meanwhile, skipped")
        except Exception as exc:
            failures.append(label)
            log.error("%s failed: %s", label, exc)
        else:
            print(f"  {label}: {count} chunks in {time.perf_counter() - started:.0f} s")
    if failures:
        print(
            f"Failed and not searchable until re-embedded: {', '.join(failures)}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
