"""Provider-agnostic LLM interface.

Every provider must return JSON that validates against a Pydantic model. Providers use
their native structured-output mode; `complete_json` validates and retries once with the
validation error appended ("repair retry").

A call is a system prompt plus a list of turns: optional earlier conversation turns and the
current user message last. Providers send the turns as a real multi-turn conversation so the
stable prefix (system prompt and earlier turns) can be served from the provider's prompt cache.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatTurn:
    role: Role
    content: str


class LLMError(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def model_name(self, small: bool = False) -> str: ...

    @abstractmethod
    def _raw_json(
        self,
        *,
        system: str,
        turns: list[ChatTurn],
        schema: type[BaseModel],
        small: bool,
        max_tokens: int,
    ) -> str:
        """Return the model's JSON text for the given schema. `turns` ends with the user
        message to answer; everything before it is earlier conversation."""

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        small: bool = False,
        max_tokens: int = 4096,
        history: Sequence[ChatTurn] = (),
    ) -> T:
        last_error: Exception | None = None
        prompt = user
        for attempt in range(2):
            raw = self._raw_json(
                system=system,
                turns=[*history, ChatTurn("user", prompt)],
                schema=schema,
                small=small,
                max_tokens=max_tokens,
            )
            try:
                return schema.model_validate_json(_strip_fences(raw))
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                log.warning("Invalid JSON from %s (attempt %d): %s", self.name, attempt + 1, exc)
                prompt = (
                    f"{user}\n\nYour previous reply was not valid for the schema:\n{exc}\n"
                    "Reply again with JSON that matches the schema exactly."
                )
        raise LLMError(f"{self.name} returned invalid JSON twice: {last_error}")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def json_schema_for(schema: type[BaseModel]) -> dict:
    """Strict JSON schema: every property required, no additional properties."""
    raw = schema.model_json_schema()

    def tighten(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                tighten(v)
        elif isinstance(node, list):
            for v in node:
                tighten(v)

    tighten(raw)
    return raw
