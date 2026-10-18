"""Query rewrite against the configured small model (skipped when no API key is set).

The rewrite decides what retrieval searches: one subquery per company, fiscal year and topic.
These tests check the cases that go wrong in practice: a company and year named only in a
question many turns back, companies and years mixed across subqueries, topics merged into or
split beyond what was asked, questions about all companies or all years, and how many
subqueries one question can produce.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from app.chat.pipeline import _rewrite_query, _rewrite_turns
from app.chat.schemas import Query

pytestmark = pytest.mark.integration

COMPANIES = ("abnamro", "asml", "cmcom", "heineken", "shell")


@pytest.fixture(scope="module")
def provider():
    from app.config import get_settings
    from app.llm import get_provider

    settings = get_settings()
    key = (
        settings.anthropic_api_key
        if settings.llm_provider == "anthropic"
        else settings.openai_api_key
    )
    if not key:
        pytest.skip(f"No API key configured for LLM provider {settings.llm_provider!r}")
    return get_provider()


def msg(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


def conversation(*pairs: tuple[str, str]) -> list[SimpleNamespace]:
    return [m for q, a in pairs for m in (msg("user", q), msg("assistant", a))]


def company_key(name: str | None) -> str | None:
    """'ABN AMRO Bank N.V.' -> 'abnamro'; None for no or an unknown company."""
    if not name:
        return None
    flat = re.sub(r"[^a-z]", "", name.lower())
    return next((key for key in COMPANIES if key in flat), flat)


def mentions(text: str) -> set[str]:
    """The known companies named in a text."""
    flat = re.sub(r"[^a-z]", "", text.lower())
    return {key for key in COMPANIES if key in flat}


def targets(query: Query) -> list[tuple[str | None, int | None]]:
    return [(company_key(s.company), s.fiscal_year) for s in query.subqueries]


def assert_no_mixing(query: Query) -> None:
    """Each subquery's question is about its own company and year only."""
    for sub in query.subqueries:
        named = mentions(sub.question)
        own = company_key(sub.company)
        assert named <= {own}, f"{sub.question!r} names other companies than {sub.company!r}"
        years = {int(y) for y in re.findall(r"\b20\d\d\b", sub.question)}
        if sub.fiscal_year is not None:
            assert years <= {sub.fiscal_year}, f"{sub.question!r} names years other than its own"


# --- company and year named only in an old question ----------------------------------------

# Four exchanges: only the first question names the company and year. Its answer is outside
# the answers the rewrite sees; the question itself is not, because it gets every question.
TERSE = conversation(
    ("How many FTEs did ASML have at the end of 2024?", "44,027 FTEs at year-end 2024."),
    ("And the net sales?", "EUR 28.3 billion."),
    ("Net income?", "EUR 7.6 billion."),
    ("And the dividend?", "EUR 6.40 per share."),
)
EXPLICIT = conversation(
    ("How many FTEs did ASML have at the end of 2024?", "ASML had 44,027 FTEs at year-end 2024."),
    ("And the net sales?", "ASML's net sales were EUR 28.3 billion in 2024."),
    ("Net income?", "ASML's net income was EUR 7.6 billion in 2024."),
    ("And the dividend?", "ASML paid a dividend of EUR 6.40 per share for 2024."),
)


def test_the_rewrite_sees_the_company_only_in_the_first_question():
    # precondition for the two tests below
    turns = _rewrite_turns(TERSE)
    assert "asml" in mentions(turns[0].content) and "2024" in turns[0].content
    rest = " ".join(t.content for t in turns[1:])
    assert "asml" not in mentions(rest) and "2024" not in rest


def test_context_named_only_in_an_old_question_is_resolved(provider):
    query = _rewrite_query(provider, "What about their CO2 emissions?", TERSE)
    assert targets(query) == [("asml", 2024)]
    assert "asml" in mentions(query.standalone_question)


def test_context_repeated_in_answers_is_resolved(provider):
    query = _rewrite_query(provider, "What about their CO2 emissions?", EXPLICIT)
    assert targets(query) == [("asml", 2024)]
    assert "asml" in mentions(query.standalone_question)


def test_follow_up_resolves_company_and_year_from_the_last_turns(provider):
    history = conversation(
        ("What is ABN AMRO's CET1 ratio in 2025?", "ABN AMRO's CET1 ratio was 14.5% in 2025."),
    )
    query = _rewrite_query(provider, "And how many employees did they have?", history)
    assert targets(query) == [("abnamro", 2025)]


# --- companies and years are not mixed -----------------------------------------------------


def test_each_company_keeps_its_own_year(provider):
    query = _rewrite_query(
        provider, "Compare ASML's net sales in 2025 with ABN AMRO's net profit in 2024.", []
    )
    assert sorted(targets(query)) == [("abnamro", 2024), ("asml", 2025)]
    assert_no_mixing(query)


def test_follow_up_switching_company_and_year_drops_the_old_ones(provider):
    history = conversation(
        ("How many FTEs did ASML have in 2025?", "ASML had 44,209 FTEs at the end of 2025."),
    )
    query = _rewrite_query(provider, "And ABN AMRO in 2024?", history)
    assert targets(query) == [("abnamro", 2024)]
    assert_no_mixing(query)


def test_comparison_follow_up_carries_the_year_to_the_new_company(provider):
    history = conversation(
        ("How many FTEs did ASML have in 2025?", "ASML had 44,209 FTEs at the end of 2025."),
    )
    query = _rewrite_query(provider, "How does that compare to CM.com?", history)
    assert sorted(targets(query)) == [("asml", 2025), ("cmcom", 2025)]
    assert_no_mixing(query)


# --- one report, one subquery per topic ----------------------------------------------------


def test_each_metric_of_one_report_gets_its_own_subquery(provider):
    question = (
        "What were ASML's net sales, net income, number of employees, Scope 1 and 2 emissions "
        "and dividend per share in 2025?"
    )
    query = _rewrite_query(provider, question, [])
    assert targets(query) == [("asml", 2025)] * 5, targets(query)
    assert len({s.question for s in query.subqueries}) == 5, [s.question for s in query.subqueries]


# The same three asks joined in different ways: one subquery per ask, each keeping the
# company and the year, and never more subqueries than asks.
ABN_AMRO_2025_ASKS = [
    "How many FTEs did ABN AMRO have in 2025 and what are its climate goals and what was its "
    "CET1 ratio?",
    "How many FTEs did ABN AMRO have in 2025, what are its climate goals, and what was its "
    "CET1 ratio?",
    "Tell me about ABN AMRO in 2025: how many FTEs did it have, what are its climate goals and "
    "what was its CET1 ratio?",
    "What were ABN AMRO's FTEs and CET1 ratio in 2025, and what are its climate goals?",
    "Tell me about ABN AMRO in 2025. How many FTEs did it have? What are its climate goals? "
    "And what was its CET1 ratio?",
]


@pytest.mark.parametrize("question", ABN_AMRO_2025_ASKS)
def test_several_asks_about_one_report_give_one_subquery_each(provider, question):
    query = _rewrite_query(provider, question, [])
    assert targets(query) == [("abnamro", 2025)] * 3, targets(query)


@pytest.mark.parametrize("question", ABN_AMRO_2025_ASKS)
def test_split_subqueries_search_with_the_company_and_year(provider, question):
    # The model tends to leave the year out of "What are ABN AMRO's climate goals?"; the
    # searched text must still carry it, which `search_text` guarantees from the fields.
    query = _rewrite_query(provider, question, [])
    for sub in query.subqueries:
        assert "abnamro" in mentions(sub.search_text), sub.search_text
        assert "2025" in sub.search_text, sub.search_text


def test_follow_ups_about_the_same_report_inherit_its_company_and_year(provider):
    history = conversation(
        ("What were ASML's net sales in 2025?", "ASML's net sales were EUR 32.7 billion in 2025."),
    )
    query = _rewrite_query(provider, "And its net income, R&D costs and headcount?", history)
    assert targets(query) == [("asml", 2025)] * 3, targets(query)
    assert_no_mixing(query)


# --- all companies, all years --------------------------------------------------------------


def test_question_about_all_companies_names_none(provider):
    query = _rewrite_query(provider, "Which companies have a net zero target?", [])
    assert len(query.subqueries) <= 1, targets(query)
    assert all(company is None for company, _ in targets(query)), targets(query)


def test_every_named_company_gets_one_subquery(provider):
    query = _rewrite_query(
        provider,
        "Compare the 2025 headcount of ABN AMRO, ASML, CM.com, Heineken and Shell.",
        [],
    )
    assert sorted(targets(query)) == [(c, 2025) for c in COMPANIES]
    assert_no_mixing(query)


def test_question_about_all_years_names_none(provider):
    query = _rewrite_query(provider, "How has ASML's workforce developed over the years?", [])
    assert targets(query) == [("asml", None)]


def test_target_year_is_not_a_fiscal_year(provider):
    query = _rewrite_query(provider, "What are Shell's emission targets for 2030 and 2050?", [])
    assert set(targets(query)) == {("shell", None)}, targets(query)
    assert len(query.subqueries) <= 2, targets(query)  # one topic, or one per target at most


def test_two_years_give_two_subqueries(provider):
    query = _rewrite_query(provider, "How did ASML's net sales change from 2024 to 2025?", [])
    assert sorted(targets(query)) == [("asml", 2024), ("asml", 2025)]
    assert_no_mixing(query)


# --- how many subqueries one question can give ---------------------------------------------


def test_companies_times_years_give_the_full_grid(provider):
    query = _rewrite_query(
        provider, "Compare the FTEs of ABN AMRO, ASML and CM.com in 2024 and 2025.", []
    )
    expected = [(c, y) for c in ("abnamro", "asml", "cmcom") for y in (2024, 2025)]
    assert sorted(targets(query)) == expected
    assert_no_mixing(query)


def test_a_very_wide_question_still_gives_a_query(provider):
    # 5 companies x 11 years. The prompt asks for at most MAX_SUBQUERIES and `_retrieve_all`
    # cuts to that anyway; whatever the model returns, the chat must get a usable query.
    question = (
        "List the FTEs of ABN AMRO, ASML, CM.com, Heineken and Shell for every year "
        "from 2015 to 2025."
    )
    query = _rewrite_query(provider, question, [])
    assert query.standalone_question
    assert_no_mixing(query)
    print(f"\n{len(query.subqueries)} subqueries: {targets(query)}")
