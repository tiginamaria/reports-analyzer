"""What the chat does before any model runs: small talk is answered at once; with nothing
to search the answer says what is missing (nothing uploaded, still processing, failed,
other embedding model) without calling the LLM or loading a model; local models that still
have to load are a stage of their own."""

import pytest

import app.chat.pipeline as pipeline
from app.chat.pipeline import SMALL_TALK, Library
from app.chat.schemas import Query
from app.config import get_settings


class NoCalls:
    """Stands in for the LLM provider and the model loaders: any use fails the test."""

    def __getattr__(self, name):
        pytest.fail(f"{name} must not be used")

    def __call__(self, *args, **kwargs):
        pytest.fail("must not be called")


@pytest.fixture(autouse=True)
def nothing_runs(monkeypatch):
    for name in ("get_embedder", "get_reranker", "models_to_load", "retrieve", "_rewrite_query"):
        monkeypatch.setattr(pipeline, name, NoCalls())


def ask(monkeypatch, library: Library, question: str = "How many FTEs did ASML have?", ids=None):
    monkeypatch.setattr(pipeline, "_library", lambda db, report_ids: library)
    events = list(pipeline.run_chat(None, NoCalls(), question, [], ids))
    assert [e.type for e in events] == ["answer"], "no stage runs, only the answer"
    return events[0].data


@pytest.mark.parametrize(
    ("library", "reason", "text"),
    [
        (Library(), "no_reports", "No report has been uploaded yet. Upload an annual report"),
        (Library(filtered=True), "no_reports", "The selected reports no longer exist"),
        (
            Library(processing=["ABN AMRO 2025 (embedding, 59%)"]),
            "reports_processing",
            "1 report still being processed: ABN AMRO 2025 (embedding, 59%). Ask again",
        ),
        (Library(failed=["Shell 2025"]), "reports_failed", "Ingestion failed for Shell 2025"),
        (
            Library(stale=["ASML 2025"]),
            "reports_need_reembedding",
            "1 report (ASML 2025) was indexed with a different embedding model",
        ),
        # processing wins over failed and stale: something is about to become searchable
        (
            Library(processing=["ASML 2025 (parsing, 10%)"], failed=["Shell 2025"]),
            "reports_processing",
            "1 report still being processed",
        ),
    ],
)
def test_nothing_to_search_is_answered_without_llm_or_models(monkeypatch, library, reason, text):
    answer = ask(monkeypatch, library)
    assert answer["status"] == "not_found"
    assert answer["answer"].startswith(text)
    assert answer["meta"]["reason"] == reason
    assert answer["meta"]["skipped_llm"] is True and answer["meta"]["skipped_retrieval"] is True
    assert answer["citations"] == []


@pytest.mark.parametrize(
    "greeting", ["hi", "Hello!", "hey there", "Good morning", "thanks a lot", "Thank you."]
)
def test_small_talk_gets_a_reply_without_llm_or_retrieval(monkeypatch, greeting):
    answer = ask(monkeypatch, Library(searchable=["ASML 2025", "Shell 2025"]), greeting)
    assert answer["status"] == "answered"
    assert answer["meta"]["reason"] == "greeting"
    assert "ASML 2025, Shell 2025" in answer["answer"]


def test_a_greeting_with_an_empty_library_points_to_the_upload(monkeypatch):
    answer = ask(monkeypatch, Library(), "hi")
    assert (
        answer["answer"]
        == "Hello! Upload an annual report on the Reports page, then ask me about it."
    )


@pytest.mark.parametrize(
    "question",
    [
        "hi, how many FTEs did ASML have?",
        "What is a good morning routine at Shell?",
        "Hello world revenue",
        "OK Computer sales",
    ],
)
def test_real_questions_are_not_small_talk(question):
    assert SMALL_TALK.match(question) is None


# --- model loading as its own stage ------------------------------------------------------


def searchable(monkeypatch, pending: list[str]):
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline, "_library", lambda db, report_ids: Library(searchable=["ASML 2025"])
    )
    monkeypatch.setattr(pipeline, "models_to_load", lambda: pending)
    monkeypatch.setattr(pipeline, "get_embedder", lambda: calls.append("embedder"))
    monkeypatch.setattr(pipeline, "get_reranker", lambda: calls.append("reranker"))
    monkeypatch.setattr(
        pipeline,
        "_rewrite_query",
        lambda provider, q, history: (
            calls.append("rewrite") or Query(standalone_question=q, subqueries=[])
        ),
    )
    monkeypatch.setattr(pipeline, "retrieve", lambda db, q, ids: calls.append("retrieve") or [])
    return calls


def test_models_still_to_load_are_a_stage_before_the_rewrite(monkeypatch):
    calls = searchable(monkeypatch, ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"])
    events = list(pipeline.run_chat(None, None, "FTEs?", [], None))
    assert [e.type for e in events] == ["stage", "stage", "stage", "answer"]
    assert events[0].data == {
        "name": "loading_models",
        "models": ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"],
    }
    assert events[1].data == {"name": "analyzing"}
    assert calls == ["embedder", "reranker", "rewrite", "retrieve"]
    assert events[-1].data["meta"]["reason"] == "no_evidence"
    assert events[-1].data["answer"].startswith("The ingested reports do not appear to cover this.")


def test_loaded_models_add_no_stage(monkeypatch):
    calls = searchable(monkeypatch, [])
    events = list(pipeline.run_chat(None, None, "FTEs?", [], None))
    assert [e.data.get("name") for e in events[:-1]] == ["analyzing", "retrieving"]
    assert calls == ["rewrite", "retrieve"]


def test_no_evidence_mentions_reports_still_processing(monkeypatch):
    searchable(monkeypatch, [])
    monkeypatch.setattr(
        pipeline,
        "_library",
        lambda db, ids: Library(
            searchable=["ASML 2025"], processing=["Shell 2025 (embedding, 40%)"]
        ),
    )
    answer = list(pipeline.run_chat(None, None, "FTEs?", [], None))[-1].data
    assert answer["answer"].endswith("1 report still being processed: Shell 2025 (embedding, 40%).")


# --- the library query ------------------------------------------------------------------


def test_library_sorts_reports_by_what_can_be_searched():
    signature = get_settings().embedding_signature

    class Rows:
        def __init__(self, rows):
            self.rows = rows
            self.filtered = None

        def filter(self, clause):
            self.filtered = clause
            return self

        def all(self):
            return self.rows

    rows = Rows(
        [
            ("ASML", 2025, "ready", 100, signature),
            ("Shell", 2025, "extracting", 90, signature),
            ("ABN AMRO", 2025, "embedding", 59, None),
            ("CM.com", 2024, "ready", 100, "local/BAAI/bge-m3"),
            ("Heineken", 2025, "failed", 0, None),
            ("", 0, "parsing", 3, None),
        ]
    )

    class DB:
        def query(self, *columns):
            return rows

    library = pipeline._library(DB(), None)
    assert library.searchable == ["ASML 2025", "Shell 2025"]
    assert library.processing == ["ABN AMRO 2025 (embedding, 59%)", "a report (parsing, 3%)"]
    assert library.stale == ["CM.com 2024"]
    assert library.failed == ["Heineken 2025"]
    assert library.filtered is False and rows.filtered is None

    assert pipeline._library(DB(), ["x"]).filtered is True
    assert rows.filtered is not None
