"""Fakes shared by the unit tests of ingestion and chat: a report row, a session that counts
commits and snapshots the report at each one, and an embedder that never loads a model."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from app.models import ReportStatus


class FakeReport:
    def __init__(self, report_id: str = "r1", status: str = ReportStatus.EMBEDDING) -> None:
        self.id = report_id
        self.company = "Test Co"
        self.fiscal_year = 2025
        self.filename = "test.pdf"
        self.file_hash = "0" * 64
        self.status = status
        self.progress = 0
        self.done = 0
        self.total = 0
        self.activity: str | None = None
        self.started_at = None
        self.embedding_model: str | None = None
        self.error: str | None = None


class FakeSession:
    """Counts commits and rollbacks; `get` serves `reports` by id; each commit snapshots the
    report's progress fields, so a test can see what the UI would have polled."""

    def __init__(self, *reports: FakeReport) -> None:
        self.reports = {r.id: r for r in reports}
        self.commits = 0
        self.rollbacks = 0
        self.snapshots: list[dict[str, Any]] = []

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get(self, model: type, report_id: str) -> FakeReport | None:
        return self.reports.get(report_id)

    def commit(self) -> None:
        self.commits += 1
        for report in self.reports.values():
            self.snapshots.append(
                {
                    "status": report.status,
                    "progress": report.progress,
                    "done": report.done,
                    "total": report.total,
                    "activity": report.activity,
                }
            )

    def rollback(self) -> None:
        self.rollbacks += 1


def chunks(n: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(id=f"c{i}", page=1, content=f"chunk {i}", embedding=None) for i in range(n)
    ]


class FakeEmbedder:
    """Returns a one-float vector per text; `after_call(number)` runs after each call, so a
    test can cancel the report mid-way as a delete request would."""

    signature = "fake/model"

    def __init__(self, slice_size: int = 10, after_call: Callable[[int], None] | None = None):
        self.slice_size = slice_size
        self.calls: list[list[str]] = []
        self.after_call = after_call

    def embed_documents(self, texts: list[str], check: Callable[[], None] | None = None):
        if check is not None:
            check()
        self.calls.append(list(texts))
        if self.after_call is not None:
            self.after_call(len(self.calls))
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]
