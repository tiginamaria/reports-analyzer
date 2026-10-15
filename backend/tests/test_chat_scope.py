"""Scope rules for chat retrieval: which indexed reports a question is answered from."""

from types import SimpleNamespace

from app.chat.pipeline import _resolve_scope
from app.chat.schemas import QueryAnalysis

SHELL_24 = SimpleNamespace(id="a", company="Shell", fiscal_year=2024)
SHELL_25 = SimpleNamespace(id="b", company="Shell plc", fiscal_year=2025)
ASML_25 = SimpleNamespace(id="c", company="ASML", fiscal_year=2025)
ALL = [SHELL_24, SHELL_25, ASML_25]


def analysis(companies=(), year=None) -> QueryAnalysis:
    return QueryAnalysis(standalone_question="q", companies=list(companies), fiscal_year=year)


def test_no_or_unmatched_hints_search_every_indexed_report():
    assert _resolve_scope(analysis(), ALL, None) == ALL
    assert _resolve_scope(analysis(["Heineken"], 1999), ALL, None) == ALL


def test_company_and_year_narrow_the_scope():
    assert _resolve_scope(analysis(["shell"]), ALL, None) == [SHELL_24, SHELL_25]
    assert _resolve_scope(analysis(["Shell"], 2025), ALL, None) == [SHELL_25]
    assert _resolve_scope(analysis([], 2025), ALL, None) == [SHELL_25, ASML_25]


def test_explicit_filter_wins_over_query_hints():
    assert _resolve_scope(analysis(["Shell"]), ALL, ["c"]) == [ASML_25]


def test_explicit_filter_matching_nothing_is_empty_not_everything():
    # An unknown id, or a report still ingesting, must not widen the search to all reports.
    assert _resolve_scope(analysis(), ALL, ["missing"]) == []
    assert _resolve_scope(analysis(), [], ["a"]) == []
