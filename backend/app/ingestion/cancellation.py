"""Cancellation of background ingestion.

Deleting a report removes its rows and PDF at once, but its ingestion keeps running as a
background task with the report's text in memory. The task checks here between inference
batches and before every write: an in-process flag set by the delete endpoint, and a fresh
query for the report row (the flag is lost on a restart, and a deletion made through
another process or directly in the database has no flag at all).

The flag functions are synchronous and thread-safe, so blocking code running in a worker
thread (a model batch) can test `check_flag`; the database check is `check`, which is
async and is what the pipeline calls between its steps.
"""

import threading

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Report


class IngestionCancelled(Exception):
    """The report was deleted while it was being ingested; the worker stops quietly."""


_lock = threading.Lock()
_running: set[str] = set()  # reports a worker is busy with in this process
_cancelled: set[str] = set()


def begin(report_id: str) -> None:
    """A worker starts on the report; `clear` when it exits, whatever happened."""
    with _lock:
        _running.add(report_id)


def request_cancel(report_id: str) -> None:
    """Mark a report's running ingestion as cancelled. A no-op when no worker is busy with
    it (a worker that starts later finds no row and stops anyway), so the set of flags
    never outgrows the set of running jobs."""
    with _lock:
        if report_id in _running:
            _cancelled.add(report_id)


def is_cancelled(report_id: str) -> bool:
    with _lock:
        return report_id in _cancelled


def is_running(report_id: str) -> bool:
    with _lock:
        return report_id in _running


def clear(report_id: str) -> None:
    with _lock:
        _running.discard(report_id)
        _cancelled.discard(report_id)


def check_flag(report_id: str) -> None:
    """Raise `IngestionCancelled` when the delete endpoint has flagged the report. Needs no
    database, so blocking code in a worker thread can call it."""
    if is_cancelled(report_id):
        raise IngestionCancelled(report_id)


async def report_exists(db: AsyncSession, report_id: str) -> bool:
    """A fresh query, never the session's identity map, so a deletion committed by another
    session is seen."""
    result = await db.execute(select(Report.id).where(Report.id == report_id))
    return result.first() is not None


async def check(db: AsyncSession, report_id: str) -> None:
    """Raise `IngestionCancelled` when the report was deleted. Called between batches and
    before every persistence step of the pipeline."""
    if is_cancelled(report_id) or not await report_exists(db, report_id):
        raise IngestionCancelled(report_id)
