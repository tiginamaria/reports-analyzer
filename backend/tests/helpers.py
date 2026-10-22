"""Helpers shared by the test modules."""

from types import SimpleNamespace

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import NullPool

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


def as_async(fn):
    """Wrap a plain function so it can stand in for an async one the code awaits."""

    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


async def collect(events):
    """Every event of an async generator, as `list()` did for the synchronous pipeline."""
    return [event async for event in events]


def sync_engine() -> Engine:
    """A synchronous engine on the app's database URL, for test plumbing (availability
    checks, fixture updates) that must not touch the app's async pool from another loop."""
    return create_engine(Settings().database_url, poolclass=NullPool)


def require_postgres(probe: str = "select 1") -> None:
    """Skip the module when the database (or the migration `probe` needs) is unavailable."""
    try:
        with sync_engine().connect() as conn:
            conn.execute(text(probe))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres not available: {exc}")
