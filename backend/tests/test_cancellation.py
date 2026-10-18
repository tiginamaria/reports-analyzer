"""Deleting a report cancels its ingestion: the worker stops before its next inference
batch or write, whether it is embedding, waiting for the model, parsing or extracting; its
final state is accurate; and progress advances batch by batch while it runs."""

import threading
from pathlib import Path

import numpy as np
import pytest

import app.extraction.extractor as extractor
import app.ingestion.pipeline as pipeline
from app.api import reports as reports_api
from app.ingestion import cancellation
from app.ingestion.cancellation import IngestionCancelled
from app.models import ReportStatus
from app.retrieval.embedder import EmbeddingError, LocalEmbedder, model_lock
from tests.fakes import FakeEmbedder, FakeReport, FakeSession, chunks


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """A worker is busy with report r1; no row exists, existence is whatever the test says."""
    monkeypatch.setattr(cancellation, "report_exists", lambda db, report_id: True)
    monkeypatch.setattr(pipeline, "models_to_load", lambda: [])
    cancellation.begin("r1")
    yield
    cancellation.clear("r1")


def embed_with(monkeypatch, embedder: FakeEmbedder, n_chunks: int):
    monkeypatch.setattr(pipeline, "get_embedder", lambda: embedder)
    report, db = FakeReport(), FakeSession(FakeReport())
    db.reports[report.id] = report
    return report, db, chunks(n_chunks)


# --- deletion during active embedding ---------------------------------------------------


def test_delete_during_embedding_stops_before_the_next_batch_and_before_persisting(monkeypatch):
    # the delete request arrives while the second batch of ten is being embedded
    embedder = FakeEmbedder(
        slice_size=10, after_call=lambda n: cancellation.request_cancel("r1") if n == 2 else None
    )
    report, db, items = embed_with(monkeypatch, embedder, 50)
    with pytest.raises(IngestionCancelled):
        pipeline._embed(db, report, items)
    assert len(embedder.calls) == 2, "no batch starts after the cancellation is observed"
    assert db.commits == 1, "batch 1 was persisted; batch 2 was not (checked before the write)"
    assert items[0].embedding is not None and items[25].embedding is None
    assert report.embedding_model is None, "a cancelled report never becomes searchable"


def test_a_row_deleted_outside_the_process_is_noticed_by_the_fresh_database_check(monkeypatch):
    exists = {"value": True}
    monkeypatch.setattr(cancellation, "report_exists", lambda db, report_id: exists["value"])
    embedder = FakeEmbedder(slice_size=10, after_call=lambda n: exists.update(value=False))
    report, db, items = embed_with(monkeypatch, embedder, 30)
    with pytest.raises(IngestionCancelled):
        pipeline._embed(db, report, items)
    assert len(embedder.calls) == 1
    assert db.commits == 0


# --- deletion while waiting for model access ----------------------------------------------


def test_delete_while_waiting_for_the_model_lock_never_runs_the_batch():
    class Model:
        encodes = 0

        def encode(self, texts, **kwargs):
            Model.encodes += 1
            return np.zeros((len(texts), 1))

    embedder = LocalEmbedder.__new__(LocalEmbedder)
    embedder.model = Model()
    embedder.batch_size = 4
    embedder.slice_size = 4
    embedder.signature = "local/test"
    cancelled = threading.Event()

    def check():
        if cancelled.is_set():
            raise IngestionCancelled("r1")

    outcome: dict[str, BaseException | str] = {}

    def job():
        try:
            embedder.embed_documents(["a", "b", "c", "d"], check=check)
            outcome["result"] = "finished"
        except BaseException as exc:
            outcome["result"] = exc

    model_lock.acquire(interactive=True)  # a chat query holds the model
    try:
        worker = threading.Thread(target=job)
        worker.start()
        worker.join(0.1)
        assert worker.is_alive(), "the job waits for the model"
        cancelled.set()  # the report is deleted while the job waits
    finally:
        model_lock.release()
    worker.join(5)
    assert isinstance(outcome["result"], IngestionCancelled)
    assert Model.encodes == 0
    assert not model_lock.locked()


def test_local_embedder_holds_the_lock_per_batch_not_per_call():
    holds = []

    class Model:
        def encode(self, texts, **kwargs):
            holds.append(len(texts))
            assert model_lock.locked()
            return np.zeros((len(texts), 1))

    embedder = LocalEmbedder.__new__(LocalEmbedder)
    embedder.model = Model()
    embedder.batch_size = 3
    embedder.slice_size = 3
    embedder.signature = "local/test"
    vectors = embedder.embed_documents([str(i) for i in range(8)])
    assert holds == [3, 3, 2] and len(vectors) == 8
    assert not model_lock.locked()


# --- the other stages -----------------------------------------------------------------------


def test_parsing_checks_cancellation_after_every_page(monkeypatch):
    pages_parsed = []

    def fake_parse_pdf(path, on_progress):
        for page in range(1, 4):
            pages_parsed.append(page)
            on_progress(page, 3)
        raise AssertionError("parsing should have stopped")

    monkeypatch.setattr(pipeline, "parse_pdf", fake_parse_pdf)
    report, db = FakeReport(status=ReportStatus.PARSING), FakeSession()
    monkeypatch.setattr(cancellation, "report_exists", lambda db, report_id: len(pages_parsed) < 2)
    with pytest.raises(IngestionCancelled):
        pipeline._parse(db, report, Path("x.pdf"))
    assert pages_parsed == [1, 2]


def test_extraction_stops_between_entries_and_records_no_outcome(monkeypatch):
    retrieved = []

    def candidate_hits(db, report, spec):
        retrieved.append(spec.key)
        cancellation.request_cancel("r1")  # deleted during the first retrieval
        return []

    class Query:
        def filter(self, *a, **k):
            return self

        def delete(self, **k):
            return 0

    class DB(FakeSession):
        def query(self, *a):
            return Query()

    failures = []
    monkeypatch.setattr(extractor, "_candidate_hits", candidate_hits)
    monkeypatch.setattr(extractor, "start_extraction", lambda db, report, keys: dict.fromkeys(keys))
    monkeypatch.setattr(extractor, "record_failure", lambda *a, **k: failures.append(a))
    monkeypatch.setattr(extractor, "_ask_model", lambda *a: pytest.fail("no model call starts"))
    report, db = FakeReport(status=ReportStatus.EXTRACTING), DB()
    with pytest.raises(IngestionCancelled):
        extractor.extract_report(
            db, report, provider=None, check=lambda: cancellation.check(db, "r1")
        )
    assert len(retrieved) == 1
    assert failures == [], "a deleted report gets no outcome rows"
    assert db.rollbacks == 1


# --- the delete endpoint and the worker's final state ---------------------------------------


def test_delete_endpoint_requests_cancellation_before_removing_the_row(monkeypatch, tmp_path):
    report = FakeReport()
    seen = {}

    class DB(FakeSession):
        def delete(self, obj):
            seen["cancelled_before_delete"] = cancellation.is_cancelled(obj.id)

    monkeypatch.setattr(reports_api, "_stored_report_path", lambda digest: tmp_path / "x.pdf")
    reports_api.delete_report("r1", DB(report))
    assert seen["cancelled_before_delete"] is True


def test_deleting_an_idle_report_leaves_no_flag_behind(monkeypatch, tmp_path):
    monkeypatch.setattr(reports_api, "_stored_report_path", lambda digest: tmp_path / "x.pdf")
    idle = FakeReport(report_id="idle")

    class DB(FakeSession):
        def delete(self, obj):
            pass

    reports_api.delete_report("idle", DB(idle))
    assert not cancellation.is_cancelled("idle") and not cancellation.is_running("idle")


def test_a_cancelled_ingestion_writes_nothing_and_clears_its_flag(monkeypatch):
    report = FakeReport(status=ReportStatus.EMBEDDING)
    db = FakeSession(report)
    monkeypatch.setattr(pipeline, "SessionLocal", lambda: db)

    def stages(db, report, path):
        cancellation.request_cancel(report.id)
        raise IngestionCancelled(report.id)

    monkeypatch.setattr(pipeline, "_run_stages", stages)
    pipeline.ingest_report("r1", Path("x.pdf"))
    assert db.commits == 0 and db.rollbacks == 1
    assert report.status == ReportStatus.EMBEDDING and report.error is None
    assert not cancellation.is_cancelled("r1") and not cancellation.is_running("r1")


def test_a_failed_ingestion_names_the_stage_the_cause_and_the_retry_action(monkeypatch):
    report = FakeReport(status=ReportStatus.PENDING)
    db = FakeSession(report)
    monkeypatch.setattr(pipeline, "SessionLocal", lambda: db)

    def stages(db, report, path):
        report.status = ReportStatus.EMBEDDING
        raise EmbeddingError("Embedding provider rate limit hit after retries")

    monkeypatch.setattr(pipeline, "_run_stages", stages)
    pipeline.ingest_report("r1", Path("x.pdf"))
    assert report.status == ReportStatus.FAILED
    assert report.error == (
        "Embedding failed: Embedding provider rate limit hit after retries. "
        "Delete the report and upload it again."
    )
    assert report.activity is None and report.progress == 0


def test_a_failure_after_deletion_writes_nothing(monkeypatch):
    db = FakeSession(FakeReport())
    monkeypatch.setattr(pipeline, "SessionLocal", lambda: db)

    def stages(db, report, path):
        del db.reports[report.id]  # the row is gone when the worker tries to record the error
        raise RuntimeError("UPDATE matched 0 rows")

    monkeypatch.setattr(pipeline, "_run_stages", stages)
    pipeline.ingest_report("r1", Path("x.pdf"))
    assert db.commits == 0


def test_unexpected_errors_keep_their_type_in_the_message():
    message = pipeline.failure_message(ValueError("bad page"), ReportStatus.PARSING)
    assert message == "Parsing failed: ValueError: bad page. Delete the report and upload it again."


# --- progress during a long stage -----------------------------------------------------------


def test_progress_advances_after_every_embedding_batch(monkeypatch):
    report, db, items = embed_with(monkeypatch, FakeEmbedder(slice_size=10), 100)
    pipeline._embed(db, report, items)
    assert [s["done"] for s in db.snapshots] == list(range(10, 101, 10))
    assert all(s["total"] == 100 for s in db.snapshots)
    percents = [s["progress"] for s in db.snapshots]
    assert percents == sorted(percents) and percents[0] > 35 and percents[-1] == 80
    assert db.snapshots[0]["activity"].startswith("Embedded chunks 1-10 of 100 in ")
    assert db.snapshots[-1]["activity"].startswith("Embedded chunks 91-100 of 100 in ")
    assert report.embedding_model == "fake/model"


def test_loading_a_local_model_is_reported_as_activity(monkeypatch):
    monkeypatch.setattr(pipeline, "models_to_load", lambda: ["BAAI/bge-m3"])
    report, db, items = embed_with(monkeypatch, FakeEmbedder(slice_size=10), 10)
    pipeline._embed(db, report, items)
    assert db.snapshots[0]["activity"] == "Loading BAAI/bge-m3"
    assert db.snapshots[0]["status"] == ReportStatus.EMBEDDING
