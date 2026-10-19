"""Shared helpers for the evaluation scripts: corpus and gold files, result files, LLM usage
accounting, and the retrieval parameters an experiment can override."""

import argparse
import dataclasses
import functools
import json
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

import yaml
from anthropic.types import Message
from openai.types.chat import ChatCompletion
from sqlalchemy.orm import Session

import app.chat.pipeline as pipeline
import app.retrieval.search as search
from app.config import get_settings
from app.llm import LLMProvider, get_provider
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.openai_provider import OpenAIProvider
from app.models import Report
from app.retrieval.embedder import Embedder, OpenAIEmbedder, Reranker, get_reranker

EVAL_DIR = Path(__file__).parent
BACKEND_DIR = EVAL_DIR.parent
GOLD_DIR = EVAL_DIR / "gold"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"

# The judge is a different and stronger model than the generator, to limit self-preference.
JUDGE_MODEL = "claude-opus-5-5"

# USD per million tokens (input, output), for the cost estimate only. A cached input token
# costs a tenth on both providers; Anthropic charges 1.25x to write the cache, OpenAI nothing.
PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-5-5": (4.0, 20.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
}
DEFAULT_PRICE = (3.0, 15.0)
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25
# USD per million tokens of the hosted embedding models; a local model costs nothing per token.
EMBEDDING_PRICES = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}


# --- corpus and gold files ------------------------------------------------------------------


@dataclass
class CorpusDoc:
    key: str
    company: str
    fiscal_year: int
    path: Path
    source: str | None = None


def load_corpus() -> list[CorpusDoc]:
    entries = yaml.safe_load((EVAL_DIR / "corpus.yaml").read_text())
    return [
        CorpusDoc(**{**entry, "path": (BACKEND_DIR / entry["path"]).resolve()}) for entry in entries
    ]


def load_yaml(name: str) -> Any:
    return yaml.safe_load((GOLD_DIR / name).read_text())


@dataclass
class Question:
    id: str
    question: str
    type: str
    modality: str
    reports: list[str]
    key_facts: list[str]
    evidence: list[dict]
    history: list[dict] = field(default_factory=list)
    answer: str = ""
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_questions(limit: int | None = None) -> list[Question]:
    fields = Question.__dataclass_fields__
    questions = [
        Question(**{k: v for k, v in item.items() if k in fields})
        for item in load_yaml("qa.yaml")["qa"]
    ]
    return questions[:limit]


def report_keys(db: Session) -> dict[str, str]:
    """Corpus key -> report id in the eval database, matched on the stored file name."""
    by_name = {Path(doc.path).name: doc.key for doc in load_corpus()}
    return {by_name[r.filename]: r.id for r in db.query(Report).all() if r.filename in by_name}


def write_json(name: str, data: Any) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    return path


# --- LLM usage ------------------------------------------------------------------------------


def price(model: str) -> tuple[float, float]:
    """The price of `model`; a dated snapshot (`gpt-5-2025-08-07`) is priced as its family."""
    matches = [name for name in PRICES if model == name or model.startswith(name + "-")]
    return PRICES[max(matches, key=len)] if matches else DEFAULT_PRICE


@dataclass
class Usage:
    """Token usage of every LLM call made while a hook is installed, per model."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        model: str,
        *,
        uncached: int,
        cache_read: int,
        cache_write: int,
        output: int,
        seconds: float,
    ) -> None:
        """One call: the input tokens billed at full price, the cached and cache-writing
        input tokens apart, the output tokens and the seconds the call took."""
        self.calls.append(
            {
                "model": model,
                "input": uncached,
                "cache_read": cache_read,
                "cache_write": cache_write,
                "output": output,
                "seconds": seconds,
            }
        )

    def mark(self) -> int:
        return len(self.calls)

    def summary(self, since: int = 0) -> dict[str, Any]:
        calls = self.calls[since:]
        cost = 0.0
        for c in calls:
            price_in, price_out = price(c["model"])
            cost += (
                c["input"] * price_in
                + c["cache_read"] * price_in * CACHE_READ_FACTOR
                + c["cache_write"] * price_in * CACHE_WRITE_FACTOR
                + c["output"] * price_out
            ) / 1e6
        return {
            "calls": len(calls),
            "input_tokens": sum(c["input"] + c["cache_read"] + c["cache_write"] for c in calls),
            "output_tokens": sum(c["output"] for c in calls),
            "llm_seconds": round(sum(c["seconds"] for c in calls), 2),
            "cost_usd": round(cost, 4),
        }


USAGE = Usage()


def instrument(provider: LLMProvider) -> None:
    """Record the usage of every call `provider` makes into `USAGE`, through the adapter's
    response hook. Anthropic reports cached and cache-writing input apart from the rest;
    OpenAI counts cached tokens inside `prompt_tokens` and a reasoning model's hidden
    reasoning inside `completion_tokens`, billed as output."""
    if isinstance(provider, AnthropicProvider):

        def on_anthropic(response: Message, seconds: float) -> None:
            u = response.usage
            USAGE.record(
                response.model,
                uncached=u.input_tokens,
                cache_read=u.cache_read_input_tokens or 0,
                cache_write=u.cache_creation_input_tokens or 0,
                output=u.output_tokens,
                seconds=seconds,
            )

        provider.on_response = on_anthropic
    elif isinstance(provider, OpenAIProvider):

        def on_openai(response: ChatCompletion, seconds: float) -> None:
            u = response.usage
            if u is None:
                return
            cached = (u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0) or 0
            USAGE.record(
                response.model,
                uncached=u.prompt_tokens - cached,
                cache_read=cached,
                cache_write=0,
                output=u.completion_tokens,
                seconds=seconds,
            )

        provider.on_response = on_openai


@dataclass
class EmbeddingUsage:
    """Tokens of every hosted embedding request made while the hook is installed."""

    model: str = ""
    requests: list[tuple[int, float]] = field(default_factory=list)  # (tokens, seconds)

    def mark(self) -> int:
        return len(self.requests)

    def summary(self, since: int = 0) -> dict[str, Any]:
        requests = self.requests[since:]
        tokens = sum(t for t, _ in requests)
        return {
            "requests": len(requests),
            "tokens": tokens,
            "seconds": round(sum(s for _, s in requests), 2),
            "cost_usd": round(tokens * EMBEDDING_PRICES.get(self.model, 0.0) / 1e6, 4),
        }


EMBED_USAGE = EmbeddingUsage()


def instrument_embedder(embedder: Embedder) -> None:
    """Record the tokens of every request a hosted embedder makes into `EMBED_USAGE`; a
    local model has no per-token cost and stays unrecorded."""
    if isinstance(embedder, OpenAIEmbedder):
        EMBED_USAGE.model = get_settings().embedding_model
        embedder.on_response = lambda tokens, seconds: EMBED_USAGE.requests.append(
            (tokens, seconds)
        )


def provider_for(model: str | None = None) -> LLMProvider:
    """The configured provider (`LLM_PROVIDER`), or the same provider with `model` as its
    main model instead of the configured one."""
    if model is None:
        return get_provider()
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(settings.model_copy(update={"anthropic_model": model}))
    return OpenAIProvider(settings.model_copy(update={"openai_model": model}))


def judge_provider() -> AnthropicProvider:
    """The judge is `claude-opus-5-5` whatever the generator, a different and stronger
    model; it needs `ANTHROPIC_API_KEY` even when the app answers with OpenAI."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit(
            f"The judge is {JUDGE_MODEL}: set ANTHROPIC_API_KEY (and ANTHROPIC_WORKSPACE_ID "
            "for a user-scoped key)"
        )
    return AnthropicProvider(settings.model_copy(update={"anthropic_model": JUDGE_MODEL}))


def require_reranker() -> Reranker:
    """A run that reranks needs the reranker loaded; refuse to run without one."""
    reranker = get_reranker()
    if reranker is None:
        raise SystemExit(
            "No reranker is configured: set RETRIEVAL_MODELS=local (or RERANKER_PROVIDER=local), "
            "or pass --no-rerank to evaluate the fused order"
        )
    return reranker


@contextmanager
def timer(store: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round(time.perf_counter() - start, 2)


# --- retrieval parameters -------------------------------------------------------------------


def _reranker_configured() -> bool:
    return get_settings().reranker_provider == "local"


@dataclass(frozen=True)
class RetrievalParams:
    """The retrieval settings an experiment can change. The defaults are the app's constants
    and the configured retrieval profile, so a run without overrides evaluates the shipped
    pipeline: with `RETRIEVAL_MODELS=openai` nothing reranks, with `local` the cross-encoder
    and its floor run."""

    candidates: int = search.RETRIEVAL_CANDIDATES  # chunks per search, kept for reranking
    top_k: int = search.RERANK_TOP_K  # chunks kept per sub-query after reranking
    min_score: float = search.RERANK_MIN_SCORE  # reranker floor; below it nothing is kept
    rrf_k: int = search.RRF_K
    search: str = "hybrid"  # "vector" or "fulltext" runs that search alone
    # without the reranker the RRF order is kept and no floor applies
    rerank: bool = field(default_factory=_reranker_configured)
    max_subqueries: int = pipeline.MAX_SUBQUERIES  # 0 searches the standalone question only

    def tag(self) -> str:
        """The fields that differ from the app's settings, for a result file name:
        `candidates40_top_k8`, `vector_no_rerank`; empty when nothing differs."""
        default = RetrievalParams()
        parts = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value == getattr(default, f.name):
                continue
            if isinstance(value, bool):
                parts.append(f.name if value else f"no_{f.name}")
            elif f.name == "search":
                parts.append(value)
            else:
                parts.append(f"{f.name}{value}")
        return "_".join(parts)


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    d = RetrievalParams()
    group = parser.add_argument_group("retrieval overrides (defaults: the app's settings)")
    group.add_argument("--candidates", type=int, default=d.candidates, help=f"({d.candidates})")
    group.add_argument("--top-k", type=int, default=d.top_k, help=f"({d.top_k})")
    group.add_argument("--min-score", type=float, default=d.min_score, help=f"({d.min_score})")
    group.add_argument("--rrf-k", type=int, default=d.rrf_k, help=f"({d.rrf_k})")
    group.add_argument("--search", choices=("hybrid", "vector", "fulltext"), default=d.search)
    group.add_argument(
        "--no-rerank",
        action="store_true",
        help="keep the fused order (what the openai retrieval profile does anyway)",
    )
    group.add_argument(
        "--max-subqueries",
        type=int,
        default=d.max_subqueries,
        help=f"0 searches the standalone question only ({d.max_subqueries})",
    )


def retrieval_params(args: argparse.Namespace) -> RetrievalParams:
    return RetrievalParams(
        candidates=args.candidates,
        top_k=args.top_k,
        min_score=args.min_score,
        rrf_k=args.rrf_k,
        search=args.search,
        rerank=RetrievalParams().rerank and not args.no_rerank,
        max_subqueries=args.max_subqueries,
    )


@contextmanager
def patched_retrieval(params: RetrievalParams) -> Iterator[None]:
    """Make the chat pipeline retrieve with `params`. The pipeline calls `retrieve` and
    `rrf_merge` with their module defaults, so it is handed versions with these values
    bound; a dropped search method returns no ranking; without a reranker `retrieve` keeps
    the fused order and applies no floor."""

    def nothing(*args: Any, **kwargs: Any) -> list[str]:
        return []

    patches: list[Any] = [
        mock.patch.object(
            pipeline,
            "retrieve",
            functools.partial(
                search.retrieve,
                candidates=params.candidates,
                top_k=params.top_k,
                min_score=params.min_score,
            ),
        ),
        mock.patch.object(
            pipeline,
            "_retrieve_all",
            functools.partial(pipeline._retrieve_all, max_subqueries=params.max_subqueries),
        ),
        mock.patch.object(search, "rrf_merge", functools.partial(search.rrf_merge, k=params.rrf_k)),
    ]
    if params.search == "vector":
        patches.append(mock.patch.object(search, "fulltext_search", nothing))
    if params.search == "fulltext":
        patches.append(mock.patch.object(search, "vector_search", nothing))
    if not params.rerank:
        patches.append(mock.patch.object(search, "get_reranker", lambda: None))
    with ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield
