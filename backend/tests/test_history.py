"""Which earlier messages are sent to the models with a new question: the main model gets
the whole conversation, the rewrite every question plus the last few answers."""

from types import SimpleNamespace

from app.chat.pipeline import (
    REWRITE_ANSWER_MESSAGES,
    REWRITE_QUESTIONS,
    _conversation_turns,
    _rewrite_turns,
)
from tests.helpers import msg


def exchanges(n: int) -> list[SimpleNamespace]:
    return [msg(r, f"{r}{i}") for i in range(n) for r in ("user", "assistant")]


# --- the main model: the whole conversation ------------------------------------------------


def test_the_whole_conversation_is_kept_in_order():
    turns = _conversation_turns(exchanges(10))
    assert [t.content for t in turns] == [
        f"{r}{i}" for i in range(10) for r in ("user", "assistant")
    ]


def test_the_conversation_starts_with_a_question():
    # an answer without its question is dropped, so the turns begin with a user message
    history = [msg("assistant", "a0"), msg("user", "q1"), msg("assistant", "a1")]
    assert [t.content for t in _conversation_turns(history)] == ["q1", "a1"]
    assert _conversation_turns([msg("assistant", "a")]) == []


def test_unanswered_question_is_kept():
    # an earlier turn that errored has no assistant message
    history = [msg("user", "q1"), msg("user", "q2"), msg("assistant", "a2")]
    assert [t.content for t in _conversation_turns(history)] == ["q1", "q2", "a2"]


# --- the rewrite: every question, answers only in the recent window -------------------------


def test_rewrite_sees_all_questions_but_only_recent_answers():
    turns = _rewrite_turns(exchanges(10))
    questions = [t.content for t in turns if t.role == "user"]
    answers = [t.content for t in turns if t.role == "assistant"]
    assert questions == [f"user{i}" for i in range(10)]
    assert answers == [f"assistant{i}" for i in range(10 - REWRITE_ANSWER_MESSAGES // 2, 10)]


def test_rewrite_keeps_the_order_of_the_conversation():
    turns = _rewrite_turns(exchanges(4))
    assert [t.content for t in turns] == [
        "user0",
        "user1",
        "assistant1",
        "user2",
        "assistant2",
        "user3",
        "assistant3",
    ]


def test_rewrite_caps_the_number_of_questions():
    history = exchanges(REWRITE_QUESTIONS + 5)
    questions = [t.content for t in _rewrite_turns(history) if t.role == "user"]
    assert len(questions) == REWRITE_QUESTIONS
    assert questions[0] == "user5"  # the oldest ones go


def test_short_history_goes_to_the_rewrite_whole():
    history = [msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2")]
    assert [t.content for t in _rewrite_turns(history)] == ["q1", "a1", "q2"]
    assert _rewrite_turns([]) == []
