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
from sqlalchemy.orm import Session

import app.chat.pipeline as pipeline
import app.retrieval.search as search
from app.config import get_settings
from app.llm import LLMProvider
from app.llm.anthropic_provider import AnthropicProvider
from app.models import Report
from app.retrieval.embedder import Reranker, get_reranker

EVAL_DIR = Path(__file__).parent
BACKEND_DIR = EVAL_DIR.parent
GOLD_DIR = EVAL_DIR / "gold"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"

# The judge is a different and stronger model than the generator, to limit self-preference.
JUDGE_MODEL = "claude-opus-5-5"

# USD per million tokens (input, output), for the cost estimate only.
PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-5-5": (4.0, 20.0),
}
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


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


@dataclass
class Usage:
    """Token usage of every LLM call made while it is installed, per model."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, model: str, usage: Any, seconds: float) -> None:
        self.calls.append(
            {
                "model": model,
                "input": usage.input_tokens,
                "cache_read": usage.cache_read_input_tokens or 0,
                "cache_write": usage.cache_creation_input_tokens or 0,
                "output": usage.output_tokens,
                "seconds": seconds,
            }
        )

    def mark(self) -> int:
        return len(self.calls)

    def summary(self, since: int = 0) -> dict[str, Any]:
        calls = self.calls[since:]
        cost = 0.0
        for c in calls:
            price_in, price_out = PRICES.get(c["model"], (3.0, 15.0))
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
    """Record the usage of every call the provider makes into `USAGE`. Only the Anthropic
    adapter reports usage; with another provider the cost figures stay empty."""
    if isinstance(provider, AnthropicProvider):
        provider.on_response = lambda response, seconds: USAGE.record(
            response.model, response.usage, seconds
        )


def anthropic_provider(model: str | None = None) -> AnthropicProvider:
    """The Anthropic adapter with the configured main model, or `model` instead."""
    settings = get_settings()
    if model:
        settings = settings.model_copy(update={"anthropic_model": model})
    return AnthropicProvider(settings)


def judge_provider() -> AnthropicProvider:
    return anthropic_provider(JUDGE_MODEL)


def require_reranker() -> Reranker:
    """The evals measure the shipped pipeline, which reranks; refuse to run without one."""
    reranker = get_reranker()
    if reranker is None:
        raise SystemExit("The evaluation needs the reranker: set RERANKER_MODEL")
    return reranker


@contextmanager
def timer(store: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round(time.perf_counter() - start, 2)


# --- retrieval parameters -------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalParams:
    """The retrieval settings an experiment can change. The defaults are the app's constants,
    so a run without overrides evaluates the shipped pipeline."""

    candidates: int = search.RETRIEVAL_CANDIDATES  # chunks per search, kept for reranking
    top_k: int = search.RERANK_TOP_K  # chunks kept per sub-query after reranking
    min_score: float = search.RERANK_MIN_SCORE  # reranker floor; below it nothing is kept
    rrf_k: int = search.RRF_K
    search: str = "hybrid"  # "vector" or "fulltext" runs that search alone
    rerank: bool = True  # without the reranker the RRF order is kept and no floor applies
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
    group.add_argument("--no-rerank", action="store_true", help="keep the fused order")
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
        rerank=not args.no_rerank,
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
