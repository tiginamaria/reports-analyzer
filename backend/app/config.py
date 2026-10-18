"""Application settings, read from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # The repo-root .env (shared with docker compose) and, if present, backend/.env on top.
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql+psycopg://reports:reports@localhost:5432/reports"
    reports_dir: Path = Path("../data/store")

    # LLM provider
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_workspace_id: str | None = None  # required for user-scoped keys (sk-ant-usr-...)
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_small_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5"
    openai_small_model: str = "gpt-5-mini"

    # Local models
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"
    embedding_batch_size: int = 32
    model_device: str | None = None  # "cpu", "mps", "cuda" or None for auto


@lru_cache
def get_settings() -> Settings:
    return Settings()
