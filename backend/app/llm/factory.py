from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLMProvider


@lru_cache
def get_provider() -> LLMProvider:
    """The configured provider, built once. SDKs are imported lazily so that only the
    configured one has to be installed and initialised."""
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if settings.llm_provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
