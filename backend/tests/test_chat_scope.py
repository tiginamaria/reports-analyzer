"""Sub-query retrieval: each sub-query is searched by a text that names its company and year
and keeps its own chunks; the results are merged, and the number of searches is capped."""

from types import SimpleNamespace

import pytest

import app.chat.pipeline as pipeline
from app.chat.pipeline import MAX_SUBQUERIES, _retrieve_all
from app.chat.schemas import Query, SubQuery


def analysis(*questions) -> Query:
    queries = [SubQuery(question=q, company=None, fiscal_year=None) for q in questions]
    return Query(standalone_question="whole?", subqueries=queries)


# --- search_text: the company and year are always in what is searched ----------------------


@pytest.mark.parametrize(
    ("question", "company", "year", "expected"),
    [
        # names both: searched as written, the natural question retrieval was tuned on
        (
            "How many FTEs did ASML have in 2025?",
            "ASML",
            2025,
            "How many FTEs did ASML have in 2025?",
        ),
        ("How many FTEs in 2025?", "ASML", 2025, "How many FTEs in 2025? ASML"),
        (
            "How many FTEs did ASML have?",
            "ASML",
            2025,
            "How many FTEs did ASML have? annual report 2025",
        ),
        ("How many FTEs?", "ASML", 2025, "How many FTEs? ASML annual report 2025"),
        ("How many FTEs did asml have?", "ASML", None, "How many FTEs did asml have?"),  # case
        (
            "Which companies have a net zero target?",
            None,
            None,
            "Which companies have a net zero target?",
        ),
    ],
)
def test_search_text_appends_only_what_the_question_lacks(question, company, year, expected):
    assert SubQuery(company=company, fiscal_year=year, question=question).search_text == expected


# --- retrieval over sub-queries ------------------------------------------------------------


def hit(chunk_id, score):
    return SimpleNamespace(chunk=SimpleNamespace(id=chunk_id), score=score)


def recording(calls: list[str]):
    """A `retrieve` stand-in that records what is searched and finds nothing."""

    def retrieve(db, question, report_ids):
        calls.append(question)
        return []

    return retrieve


def test_each_sub_query_is_retrieved_and_merged(monkeypatch):
    calls = []

    def fake_retrieve(db, question, report_ids):
        calls.append((question, report_ids))
        if question == "shell?":
            return [hit("shell", 0.99), hit("both", 0.5)]
        return [hit("asml", 0.2), hit("both", 0.9)]

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    hits = _retrieve_all(None, analysis("shell?", "asml?"), ["r1"])
    assert [h.chunk.id for h in hits] == ["shell", "both", "asml"]  # duplicate kept once
    assert calls == [("shell?", ["r1"]), ("asml?", ["r1"])]  # the user's filter on each


def test_without_sub_queries_the_standalone_question_is_searched(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "retrieve", recording(calls))
    _retrieve_all(None, analysis(), None)
    assert calls == ["whole?"]


def test_no_hits_abstains_without_the_model_and_says_why(monkeypatch):
    monkeypatch.setattr(pipeline, "_rewrite_query", lambda provider, q, history: analysis("q?"))
    monkeypatch.setattr(pipeline, "retrieve", lambda db, question, report_ids: [])
    monkeypatch.setattr(pipeline, "models_to_load", lambda: [])
    monkeypatch.setattr(
        pipeline, "_library", lambda db, report_ids: pipeline.Library(searchable=["ASML 2025"])
    )
    events = list(pipeline.run_chat(None, None, "q?", [], None))
    answer = events[-1]
    assert answer.type == "answer"
    assert answer.data["status"] == "not_found"
    assert answer.data["answer"].startswith("The ingested reports do not appear to cover this.")
    assert answer.data["meta"]["skipped_llm"] is True
    assert answer.data["meta"]["reason"] == "no_evidence"


def test_sub_queries_are_searched_by_their_search_text(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "retrieve", recording(calls))
    sub = SubQuery(company="ASML", fiscal_year=2025, question="How many FTEs?")
    _retrieve_all(None, Query(standalone_question="whole?", subqueries=[sub]), None)
    assert calls == ["How many FTEs? ASML annual report 2025"]


def test_only_the_first_max_subqueries_are_searched(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "retrieve", recording(calls))
    questions = [f"q{i}?" for i in range(MAX_SUBQUERIES + 5)]
    _retrieve_all(None, analysis(*questions), None)
    assert calls == questions[:MAX_SUBQUERIES]


def test_every_sub_query_keeps_its_chunks_and_the_merge_is_sorted(monkeypatch):
    # a sub-query with low scores is not crowded out by one with high scores
    scores = {"high?": [0.99, 0.98, 0.97], "low?": [0.6, 0.5, 0.4]}
    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda db, question, report_ids: [
            hit(f"{question}-{i}", s) for i, s in enumerate(scores[question])
        ],
    )
    hits = _retrieve_all(None, analysis("high?", "low?"), None)
    assert len(hits) == 6
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert [h.chunk.id for h in hits][-3:] == ["low?-0", "low?-1", "low?-2"]
