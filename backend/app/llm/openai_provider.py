"""OpenAI adapter."""

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from app.config import Settings
from app.llm.base import ChatTurn, LLMError, LLMProvider, json_schema_for

# Model families whose hidden reasoning tokens count against `max_completion_tokens` and
# that accept `reasoning_effort`.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
# Output tokens granted on top of the caller's budget for a reasoning model's thinking, so
# the budget the pipeline asks for is what is left for the visible answer.
REASONING_HEADROOM = 16_000


def is_reasoning_model(name: str) -> bool:
    return name.startswith(REASONING_MODEL_PREFIXES)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._client = openai.OpenAI(api_key=settings.openai_api_key, timeout=180.0)
        self._model = settings.openai_model
        self._small = settings.openai_small_model
        self._reasoning_effort = settings.openai_reasoning_effort

    def model_name(self, small: bool = False) -> str:
        return self._small if small else self._model

    def _raw_json(
        self,
        *,
        system: str,
        turns: list[ChatTurn],
        schema: type[BaseModel],
        small: bool,
        max_tokens: int,
    ) -> str:
        model = self.model_name(small)
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=_messages(system, turns),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema_for(schema),
                        "strict": True,
                    },
                },
                **self._output_limits(model, max_tokens),
            )
        except openai.AuthenticationError as exc:
            raise LLMError(f"{self.name} authentication failed: check the API key") from exc
        except openai.RateLimitError as exc:
            raise LLMError(f"{self.name} rate limit hit, try again shortly") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"{self.name} API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"Could not reach {self.name}: {exc}") from exc
        choice = response.choices[0]
        content = choice.message.content
        if choice.finish_reason == "length":
            raise LLMError(
                f"{self.name} model {model} hit its output limit before finishing the reply"
                + (" (it spent the whole budget reasoning)" if not content else "")
                + "; lower OPENAI_REASONING_EFFORT or raise the budget"
            )
        if not content:
            raise LLMError(f"{self.name} returned an empty message")
        return content

    def _output_limits(self, model: str, max_tokens: int) -> dict:
        """`max_completion_tokens` and, for a reasoning model, `reasoning_effort` plus
        headroom: these models count their hidden reasoning against the completion limit,
        so a 4,000-token answer budget can be used up before a single visible token."""
        if not is_reasoning_model(model):
            return {"max_completion_tokens": max_tokens}
        limits: dict = {"max_completion_tokens": max_tokens + REASONING_HEADROOM}
        if self._reasoning_effort:
            limits["reasoning_effort"] = self._reasoning_effort
        return limits


def _messages(system: str, turns: list[ChatTurn]) -> list[ChatCompletionMessageParam]:
    """OpenAI caches shared prompt prefixes automatically; keeping the system prompt and
    earlier turns byte-identical across requests is all that is needed."""
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
    for turn in turns:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.content})
        else:
            messages.append({"role": "assistant", "content": turn.content})
    return messages
