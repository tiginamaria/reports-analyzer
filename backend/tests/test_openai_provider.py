"""The OpenAI adapter with reasoning models: their thinking counts against the completion
limit, so the call gets headroom and a reasoning effort, and running out of tokens is
reported as such rather than as an empty message."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.llm.base import LLMError
from app.llm.openai_provider import REASONING_HEADROOM, OpenAIProvider


class Out(BaseModel):
    word: str


class FakeCompletions:
    def __init__(self, content: str | None = '{"word": "x"}', finish_reason: str = "stop") -> None:
        self.calls: list[dict] = []
        self.content = content
        self.finish_reason = finish_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)]
        )


def provider(
    model: str, effort: str | None = "low", **fake_kwargs
) -> tuple[OpenAIProvider, FakeCompletions]:
    fake = FakeCompletions(**fake_kwargs)
    p = OpenAIProvider.__new__(OpenAIProvider)
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    p._model = model
    p._small = model
    p._reasoning_effort = effort
    return p, fake


def test_reasoning_models_get_headroom_and_an_effort():
    p, fake = provider("gpt-5")
    assert p.complete_json(system="s", user="u", schema=Out, max_tokens=4000).word == "x"
    call = fake.calls[0]
    assert call["max_completion_tokens"] == 4000 + REASONING_HEADROOM
    assert call["reasoning_effort"] == "low"


def test_non_reasoning_models_get_the_plain_budget():
    p, fake = provider("gpt-4.1")
    p.complete_json(system="s", user="u", schema=Out, max_tokens=4000)
    call = fake.calls[0]
    assert call["max_completion_tokens"] == 4000
    assert "reasoning_effort" not in call


def test_no_effort_setting_sends_none():
    p, fake = provider("gpt-5-mini", effort=None)
    p.complete_json(system="s", user="u", schema=Out, max_tokens=100)
    assert "reasoning_effort" not in fake.calls[0]


def test_running_out_of_tokens_while_reasoning_is_a_clear_error():
    p, _ = provider("gpt-5", content=None, finish_reason="length")
    with pytest.raises(LLMError, match="spent the whole budget reasoning"):
        p.complete_json(system="s", user="u", schema=Out, max_tokens=4000)


def test_a_truncated_reply_is_not_passed_on_as_json():
    p, _ = provider("gpt-5", content='{"word": "x', finish_reason="length")
    with pytest.raises(LLMError, match="hit its output limit"):
        p.complete_json(system="s", user="u", schema=Out, max_tokens=4000)
