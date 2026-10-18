"""Application settings, read from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# One flag chooses the whole retrieval stack: what embeds the chunks and queries and whether
# a reranker runs. `openai` needs only an API key; `local` runs bge-m3 and the bge reranker
# in-process (the `local-models` dependency group) and needs no embedding key.
RETRIEVAL_PROFILES: dict[str, dict[str, str]] = {
    "openai": {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "reranker_provider": "none",
    },
    "local": {
        "embedding_provider": "local",
        "embedding_model": "BAAI/bge-m3",
        "reranker_provider": "local",
    },
}


class Settings(BaseSettings):
    # The repo-root .env (shared with docker compose) and, if present, backend/.env on top.
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql+psycopg://reports:reports@localhost:5432/reports"
    reports_dir: Path = Path("../data/store")

    # LLM provider (chat, extraction, metadata detection). OpenAI by default, so one key
    # covers the LLM and the hosted embeddings; Anthropic is a drop-in alternative.
    llm_provider: Literal["anthropic", "openai"] = "openai"
    #
    anthropic_api_key: str | None = None
    anthropic_workspace_id: str | None = None  # required for user-scoped keys (sk-ant-usr-...)
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_small_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5"
    openai_small_model: str = "gpt-5-mini"
    # Reasoning models (gpt-5 family, o-series) think before they answer; the effort sent
    # with every call. Grounded answers over given chunks need little of it, and more only
    # adds latency and eats into the output budget. None sends nothing.
    openai_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "low"

    # Retrieval stack, independent of the LLM provider: `openai` (hosted embeddings, no
    # reranker) or `local` (bge-m3 and the bge reranker in-process). It fills the three
    # settings below; any of them set explicitly wins over the profile.
    retrieval_models: Literal["openai", "local"] = "openai"
    # `openai` calls a hosted embedding endpoint (any OpenAI-compatible one through
    # `embedding_base_url`); `local` runs a sentence-transformers model in-process.
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_api_key: str | None = None  # falls back to openai_api_key
    embedding_base_url: str | None = None
    embedding_dimensions: int | None = None  # None: the model's native size
    embedding_batch_size: int = 64  # texts per request (hosted) or per model call (local)
    embedding_concurrency: int = 4  # hosted requests in flight at once
    embedding_timeout_seconds: float = 60.0  # per hosted request
    embedding_max_retries: int = 3  # per hosted request, on timeouts, 429 and 5xx

    # Reranking: `none` keeps the fused search order (no local model, no score floor);
    # `local` runs a cross-encoder in-process.
    reranker_provider: Literal["none", "local"] = "none"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Device of the local models: "cpu", "mps", "cuda" or None for auto
    model_device: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_retrieval_profile(cls, data: Any) -> Any:
        """Fill the embedding and reranker settings from `retrieval_models` unless they were
        set explicitly (an empty value counts as unset, so `EMBEDDING_PROVIDER=` in a .env
        does not override the profile)."""
        if not isinstance(data, dict):
            return data
        name = str(data.get("retrieval_models") or "openai").lower()
        profile = RETRIEVAL_PROFILES.get(name)
        if profile is None:
            return data  # the field validator reports the bad value
        return (
            data
            | {"retrieval_models": name}
            | {k: v for k, v in profile.items() if not data.get(k)}
        )

    @property
    def embedding_signature(self) -> str:
        """Names the vectors a report was embedded with (`openai/text-embedding-3-large`,
        `local/BAAI/bge-m3`). Stored per report; only reports whose signature matches the
        configured one are searched, since vectors of different models do not compare."""
        signature = f"{self.embedding_provider}/{self.embedding_model}"
        if self.embedding_provider == "openai" and self.embedding_dimensions:
            signature += f"@{self.embedding_dimensions}"
        return signature

    @property
    def reranker_name(self) -> str | None:
        return self.reranker_model if self.reranker_provider == "local" else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
