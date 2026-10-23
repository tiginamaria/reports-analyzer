"""Anthropic adapter."""

import logging
import time
from collections.abc import Callable

import anthropic
from anthropic.types import CacheControlEphemeralParam, Message, MessageParam, TextBlockParam
from pydantic import BaseModel

from app.config import Settings
from app.llm.base import ChatTurn, LLMError, LLMProvider, json_schema_for

log = logging.getLogger(__name__)
CACHE = CacheControlEphemeralParam(type="ephemeral")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        headers = (
            {"anthropic-workspace-id": settings.anthropic_workspace_id}
            if settings.anthropic_workspace_id
            else None
        )
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key, timeout=180.0, default_headers=headers
        )
        self._model = settings.anthropic_model
        self._small = settings.anthropic_small_model
        # Called with every complete response and the seconds it took; the eval harness
        # records token usage and cost through it.
        self.on_response: Callable[[Message, float], None] | None = None

    def model_name(self, small: bool = False) -> str:
        return self._small if small else self._model

    async def _raw_json(
        self,
        *,
        system: str,
        turns: list[ChatTurn],
        schema: type[BaseModel],
        small: bool,
        max_tokens: int,
    ) -> str:
        start = time.perf_counter()
        try:
            # streamed, so a long reply (a goal list can run to tens of thousands of tokens)
            # is not cut by the HTTP timeout; the caller still gets one complete message
            async with self._client.messages.stream(
                model=self.model_name(small),
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": CACHE}],
                messages=_messages(turns),
                output_config={
                    "format": {"type": "json_schema", "schema": json_schema_for(schema)}
                },
            ) as stream:
                response = await stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic authentication failed: check ANTHROPIC_API_KEY") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit hit, try again shortly") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc
        if self.on_response:
            self.on_response(response, time.perf_counter() - start)
        usage = response.usage
        log.info(
            "%s %s: input=%d cache_read=%d cache_write=%d output=%d",
            self.name,
            response.model,
            usage.input_tokens,
            usage.cache_read_input_tokens or 0,
            usage.cache_creation_input_tokens or 0,
            usage.output_tokens,
        )
        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request")
        if response.stop_reason == "max_tokens":
            # a cut-off reply is invalid JSON and would fail twice through the repair retry
            raise LLMError(
                f"The reply was cut off at {max_tokens} output tokens; raise max_tokens or "
                "ask for less"
            )
        for block in response.content:
            if block.type == "text":
                return block.text
        raise LLMError("Anthropic returned no text block")


def _messages(turns: list[ChatTurn]) -> list[MessageParam]:
    """Earlier turns are replayed byte for byte on every request of a conversation, so a
    cache breakpoint on the last of them lets the next request read the whole prefix from
    cache and pay full price only for the final user message (chunks and question)."""
    messages: list[MessageParam] = []
    for index, turn in enumerate(turns):
        block = TextBlockParam(type="text", text=turn.content)
        if index == len(turns) - 2:
            block["cache_control"] = CACHE
        messages.append(MessageParam(role=turn.role, content=[block]))
    return messages
