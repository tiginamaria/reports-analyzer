"""The provider-agnostic `complete_json` (repair retry, history) and the Anthropic adapter."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm.anthropic_provider import AnthropicProvider, _messages
from app.llm.base import ChatTurn, LLMError, LLMProvider, json_schema_for


class Out(BaseModel):
    name: str
    count: int


class FlakyProvider(LLMProvider):
    """Replays canned replies and records the turns it was sent."""

    name = "flaky"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def model_name(self, small: bool = False) -> str:
        return "flaky"

    def _raw_json(self, *, system, turns, schema, small, max_tokens) -> str:
        self.turns = turns
        self.prompts.append(turns[-1].content)
        return self.replies.pop(0)


# --- complete_json ---------------------------------------------------------------------------


def test_repair_retry_sends_validation_error_back_and_accepts_fenced_json():
    p = FlakyProvider(['{"name": "x"}', '```json\n{"name": "x", "count": 2}\n```'])
    out = p.complete_json(system="s", user="u", schema=Out)
    assert out.count == 2
    assert "not valid for the schema" in p.prompts[1]


def test_two_failures_raise():
    p = FlakyProvider(["nope", "still nope"])
    with pytest.raises(LLMError):
        p.complete_json(system="s", user="u", schema=Out)


def test_history_precedes_the_user_turn_and_survives_the_repair_retry():
    history = [ChatTurn("user", "q1"), ChatTurn("assistant", "a1")]
    p = FlakyProvider(["nope", '{"name": "x", "count": 1}'])
    p.complete_json(system="s", user="q2", schema=Out, history=history)
    assert p.turns[:2] == history
    assert p.turns[-1].role == "user" and p.turns[-1].content.startswith("q2")


def test_schema_is_strict():
    schema = json_schema_for(Out)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name", "count"]


# --- the Anthropic adapter -------------------------------------------------------------------


def test_anthropic_messages_put_cache_breakpoint_on_last_history_turn():
    turns = [ChatTurn("user", "q1"), ChatTurn("assistant", "a1"), ChatTurn("user", "q2")]
    msgs = _messages(turns)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [("cache_control" in m["content"][0]) for m in msgs] == [False, True, False]
    assert "cache_control" not in _messages(turns[-1:])[0]["content"][0]


def test_anthropic_reports_a_cut_off_reply_instead_of_returning_broken_json(monkeypatch):
    provider = AnthropicProvider(Settings(anthropic_api_key="test"))
    usage = SimpleNamespace(
        input_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=5
    )
    response = SimpleNamespace(
        stop_reason="max_tokens",
        usage=usage,
        model="m",
        content=[SimpleNamespace(type="text", text='{"name": "x", "cou')],
    )

    @contextmanager
    def fake_stream(**kwargs):
        yield SimpleNamespace(get_final_message=lambda: response)

    monkeypatch.setattr(provider._client.messages, "stream", fake_stream)
    with pytest.raises(LLMError, match="cut off at 5 output tokens"):
        provider._raw_json(
            system="s", turns=[ChatTurn("user", "u")], schema=Out, small=False, max_tokens=5
        )
