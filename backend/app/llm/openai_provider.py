"""OpenAI adapter."""

from __future__ import annotations

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from app.config import Settings
from app.llm.base import ChatTurn, LLMError, LLMProvider, json_schema_for


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._client = openai.OpenAI(api_key=settings.openai_api_key, timeout=180.0)
        self._model = settings.openai_model
        self._small = settings.openai_small_model

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
        try:
            response = self._client.chat.completions.create(
                model=self.model_name(small),
                messages=_messages(system, turns),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema_for(schema),
                        "strict": True,
                    },
                },
                max_completion_tokens=max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise LLMError(f"{self.name} authentication failed: check the API key") from exc
        except openai.RateLimitError as exc:
            raise LLMError(f"{self.name} rate limit hit, try again shortly") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"{self.name} API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"Could not reach {self.name}: {exc}") from exc
        content = response.choices[0].message.content
        if not content:
            raise LLMError(f"{self.name} returned an empty message")
        return content


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
