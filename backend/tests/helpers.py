"""Helpers shared by the test modules."""

from types import SimpleNamespace

import pytest

from app.config import Settings


def msg(role: str, content: str) -> SimpleNamespace:
    """A conversation message as the pipeline reads it: only `role` and `content` matter."""
    return SimpleNamespace(role=role, content=content)


def require_api_key(settings: Settings) -> None:
    """Skip the test when the configured LLM provider has no API key."""
    if settings.llm_provider == "anthropic":
        key = settings.anthropic_api_key
    else:
        key = settings.openai_api_key
    if not key:
        pytest.skip(f"No API key configured for LLM provider {settings.llm_provider!r}")
