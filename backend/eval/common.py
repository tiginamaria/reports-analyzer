"""Shared helpers for the evaluation scripts: corpus, gold files, results, LLM usage."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EVAL_DIR = Path(__file__).parent
BACKEND_DIR = EVAL_DIR.parent
GOLD_DIR = EVAL_DIR / "gold"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"

# USD per million tokens (input, output), for the cost estimate only.
PRICES = {
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-5-5": (5.0, 25.0),
}
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


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


def write_json(name: str, data: Any) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    return path


def report_keys(db) -> dict[str, str]:  # noqa: ANN001 - Session
    """Corpus key -> report id in the eval database, matched on the stored file name."""
    from app.models import Report

    by_name = {Path(doc.path).name: doc.key for doc in load_corpus()}
    return {by_name[r.filename]: r.id for r in db.query(Report).all() if r.filename in by_name}


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


def instrument(provider) -> None:  # noqa: ANN001 - AnthropicProvider
    """Record the usage of every Anthropic call the provider makes into `USAGE`."""
    messages = provider._client.messages
    original = messages.create
    if getattr(original, "_eval_wrapped", False):
        return

    def create(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        response = original(*args, **kwargs)
        USAGE.record(response.model, response.usage, time.perf_counter() - start)
        return response

    create._eval_wrapped = True  # type: ignore[attr-defined]
    messages.create = create


@contextmanager
def timer(store: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round(time.perf_counter() - start, 2)
