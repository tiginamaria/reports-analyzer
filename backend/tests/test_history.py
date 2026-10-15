"""Token-budgeted selection of the conversation turns sent to the model."""

from types import SimpleNamespace

from app.chat.pipeline import estimate_tokens, select_history


def msg(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


def test_keeps_whole_recent_turns_within_budget():
    history = [
        msg("user", "q1 " * 100),
        msg("assistant", "a1 " * 100),
        msg("user", "q2"),
        msg("assistant", "a2"),
    ]
    budget = estimate_tokens("q2") + estimate_tokens("a2") + 5
    turns = select_history(history, budget)
    assert [(t.role, t.content) for t in turns] == [("user", "q2"), ("assistant", "a2")]


def test_unanswered_question_is_its_own_turn():
    # an earlier turn that errored has no assistant message; it is kept or dropped as a unit
    history = [msg("user", "q1"), msg("user", "q2"), msg("assistant", "a2")]
    assert [t.content for t in select_history(history, 10_000)] == ["q1", "q2", "a2"]
