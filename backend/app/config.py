"""Application settings, read from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Literal

from pydantic import field_validator
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

    # Parsing: a detected table is kept only if it has at least this many rows and numeric
    # cells; layout grids and navigation bars fail the numeric test.
    table_min_rows: int = 2
    table_min_numeric_cells: int = 3

    # Chunking: fixed windows of words with overlap (roughly 0.75 words per token); chunks
    # shorter than the minimum carry no meaning and are dropped.
    chunk_max_words: int = 400
    chunk_overlap_words: int = 60
    chunk_min_words: int = 5

    # Retrieval
    retrieval_candidates: int = 40
    rerank_top_k: int = 8
    rerank_min_score: float = 0.003  # sigmoid of reranker logit; below this we abstain

    # Chat history sent to the model: the most recent whole turns that fit this budget
    # (estimated tokens). Older turns are dropped; the retrieved chunks of earlier turns are
    # never replayed, only the question and the answer text.
    history_max_tokens: int = 3000

    # Verification
    fuzzy_quote_threshold: float = 0.92

    # System prompts
    prompt_detect_metadata: str = """
        You identify annual reports. Given the first pages of a PDF, return the reporting
        company's short common name (e.g. 'Shell', 'ABN AMRO', 'ASML', 'Heineken', 'CM.com'),
        the fiscal year the report covers, and the report title.
    """
    prompt_extract: str = """
        You extract datapoints from annual report excerpts.
        Rules:
        - Use only the provided chunks. Never use outside knowledge.
        - Every item needs a `quote` copied verbatim (character for character) from a single chunk,
          and the `chunk_id` of that chunk. Do not paraphrase, do not merge sentences.
        - If the information is not present, return found=false and an empty list.
        - Keep values exactly as written in the quote, including thousands separators.
    """
    prompt_query_analysis: str = """
        You prepare user questions about company annual reports for retrieval.
        Rewrite the latest question so it stands alone (resolve "it", "they", "that year" using the
        conversation). Identify the companies mentioned and the fiscal year if one is stated. Do not
        answer the question.
    """
    prompt_answer: str = """
        You answer questions about company annual reports using only the chunks provided.
        The conversation so far is given as earlier turns; the chunks for the latest question are in
        the last message. Rules:
        - Base every statement on the chunks in the last message. Earlier turns only tell you what
          the user is referring to; never cite or reuse figures from them. Never use outside
          knowledge or guess.
        - Every figure must carry its unit and period (e.g. "EUR 1.2 billion in 2025").
        - For each claim add a citation with the chunk_id and a quote copied verbatim (character for
          character) from that chunk. Do not paraphrase quotes. Prefer the sentence or table row that
          contains the number.
        - If the chunks do not contain the answer, set status to "not_found" and say plainly that the
          report does not state it. Do not estimate. If the report mentions a related but different
          figure (e.g. climate mitigation spending instead of adaptation spending), you may say so
          with a citation and set status to "partial".
        - Write the answer as short plain prose in the user's language, without Markdown markup.
    """

    @field_validator(
        "prompt_detect_metadata", "prompt_extract", "prompt_query_analysis", "prompt_answer"
    )
    @classmethod
    def _dedent_prompt(cls, value: str) -> str:
        """Prompts are written indented inside the class; strip that before use."""
        return dedent(value).strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
