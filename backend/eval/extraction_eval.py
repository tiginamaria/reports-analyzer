"""Evaluation of ingestion-time extraction (FTE and sustainability goals) against gold labels.

`stored` scores the datapoints the ingestion run wrote. Experiments re-run extraction with a
modified registry entry inside a transaction that is rolled back, so the stored datapoints
stay as ingestion produced them.

FTE is scored per report: exact value, an acceptable alternative (average FTE, headcount),
or wrong. Goals are matched to the gold list by an LLM judge (labels and targets are
paraphrased, so string matching undercounts); recall is reported over all gold goals and over
the headline (`core`) ones. Candidate-page recall tells retrieval misses from model misses.

    uv run python -m eval.extraction_eval stored
    uv run python -m eval.extraction_eval goals_k48 goals_queries
"""

from __future__ import annotations

import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel

from app.config import get_settings
from app.db import SessionLocal
from app.extraction.extractor import (
    DatapointSpec,
    _candidate_hits,
    _extract_datapoints,
    load_registry,
)
from app.llm import LLMProvider, get_provider
from app.llm.anthropic_provider import AnthropicProvider
from app.models import Datapoint, Report
from eval.common import USAGE, instrument, load_corpus, load_yaml, write_json
from eval.e2e_eval import JUDGE_MODEL

NUMERIC_TOLERANCE = 0.5

EXTRA_GOAL_QUERIES = [
    "our targets and ambitions table KPI target year progress",
    "we aim to achieve by 2030 commitment",
    "net zero by 2040 2050 transition plan interim targets",
]

# The extractor caps output at 6,000 tokens; goal lists from 48 chunks overflow it (JSON is
# ~1.25 characters per token) and the truncated reply fails twice. `max_tokens` raises the
# cap for an experiment so recall is not confounded by truncation.
BIG_OUTPUT = 16000

EXPERIMENTS = {
    "stored": None,
    "goals_k48": {"sustainability_goal": {"top_k": 48}},
    "goals_queries": {"sustainability_goal": {"queries_extra": EXTRA_GOAL_QUERIES}},
    "goals_k48_queries": {
        "sustainability_goal": {"top_k": 48, "queries_extra": EXTRA_GOAL_QUERIES}
    },
    "goals_k48_16k": {"sustainability_goal": {"top_k": 48, "max_tokens": BIG_OUTPUT}},
    "goals_k48_queries_16k": {
        "sustainability_goal": {
            "top_k": 48,
            "queries_extra": EXTRA_GOAL_QUERIES,
            "max_tokens": BIG_OUTPUT,
        }
    },
}


class OutputBudget:
    """The provider with a raised floor on max_tokens; everything else is delegated."""

    def __init__(self, provider: LLMProvider, max_tokens: int) -> None:
        self._provider = provider
        self._max_tokens = max_tokens

    def complete_json(self, **kwargs: Any) -> Any:
        kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), self._max_tokens)
        return self._provider.complete_json(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


def as_dict(dp: Datapoint) -> dict[str, Any]:
    return {
        "key": dp.key,
        "label": dp.label,
        "value": dp.value,
        "numeric_value": dp.numeric_value,
        "unit": dp.unit,
        "period": dp.period,
        "quote": dp.quote,
        "page": dp.page,
        "verification": dp.verification,
    }


def modified_spec(spec: DatapointSpec, change: dict[str, Any]) -> DatapointSpec:
    change = dict(change)
    extra = change.pop("queries_extra", [])
    change.pop("max_tokens", None)
    return dataclasses.replace(spec, queries=[*spec.queries, *extra], **change)


def extracted(db, report: Report, experiment: str) -> tuple[list[dict], dict[str, set[int]]]:  # noqa: ANN001
    """Datapoints for the report (stored or freshly extracted) and the candidate pages each
    registry entry passed to the model."""
    changes = EXPERIMENTS[experiment] or {}
    specs = [
        modified_spec(spec, changes[spec.key]) if spec.key in changes else spec
        for spec in load_registry()
    ]
    pages = {
        spec.key: {hit.chunk.page for hit in _candidate_hits(db, report, spec)} for spec in specs
    }
    if EXPERIMENTS[experiment] is None:
        rows = db.query(Datapoint).filter(Datapoint.report_id == report.id).all()
        return [as_dict(dp) for dp in rows], pages
    provider = get_provider()
    instrument(provider)
    out = []
    for spec in specs:
        if spec.key in changes:
            budget = changes[spec.key].get("max_tokens")
            llm = OutputBudget(provider, budget) if budget else provider
            out.extend(as_dict(dp) for dp in _extract_datapoints(db, report, spec, llm))
        else:
            rows = db.query(Datapoint).filter(
                Datapoint.report_id == report.id, Datapoint.key == spec.key
            )
            out.extend(as_dict(dp) for dp in rows)
    db.rollback()
    return out, pages


def score_fte(gold: dict, items: list[dict], candidate_pages: set[int]) -> dict[str, Any]:
    item = next((d for d in items if d["key"] == "fte"), None)
    result = {
        "gold": gold["value"],
        "gold_page": gold["page"],
        "candidate_has_gold_page": gold["page"] in candidate_pages,
    }
    if item is None:
        return {**result, "outcome": "missing"}

    def same(value: float | None, target: float) -> bool:
        return value is not None and abs(value - target) <= NUMERIC_TOLERANCE

    if same(item["numeric_value"], gold["numeric"]):
        outcome = "exact"
    elif any(same(item["numeric_value"], alt["numeric"]) for alt in gold.get("alternatives", [])):
        outcome = "alternative"
    else:
        outcome = "wrong"
    return {
        **result,
        "outcome": outcome,
        "extracted": item["value"],
        "unit": item["unit"],
        "period": item["period"],
        "page": item["page"],
        "verification": item["verification"],
    }


class GoalMatch(BaseModel):
    gold_index: int
    extracted_index: int | None


class ExtractedVerdict(BaseModel):
    extracted_index: int
    is_goal: bool  # a forward-looking, concrete commitment (not a past result or vague aim)


class GoalJudgement(BaseModel):
    matches: list[GoalMatch]
    extracted: list[ExtractedVerdict]


GOAL_JUDGE = """You compare sustainability goals extracted from an annual report with an
analyst's gold list for the same report.
- matches: for every gold goal (by index) give the index of the extracted item that states the
  same goal (same subject and target; target year may be written differently), or null.
  An extracted item may match at most one gold goal.
- extracted: for every extracted item, is_goal = true if it is a forward-looking, concrete
  commitment or target (quantified, or a clearly defined outcome with a date), false if it
  is a past achievement, a current status, a policy statement without a target, or a
  duplicate of an earlier extracted item."""


def score_goals(judge: LLMProvider, gold: list[dict], items: list[dict], pages: set[int]) -> dict:
    goals = [d for d in items if d["key"] == "sustainability_goal"]
    if not goals:
        return {"gold": len(gold), "extracted": 0, "recall": 0.0, "recall_core": 0.0}
    gold_text = "\n".join(
        f"{i}. {g['label']}: {g['target']} by {g['year']}" for i, g in enumerate(gold)
    )
    extracted_text = "\n".join(
        f"{i}. {d['label']}: {d['value']} {d['unit'] or ''} ({d['period']}) | quote: {d['quote']}"
        for i, d in enumerate(goals)
    )
    verdict = judge.complete_json(
        system=GOAL_JUDGE,
        user=f"Gold goals:\n{gold_text}\n\nExtracted items:\n{extracted_text}",
        schema=GoalJudgement,
        max_tokens=4000,
    )
    matched = {m.gold_index for m in verdict.matches if m.extracted_index is not None}
    core = [i for i, g in enumerate(gold) if g.get("core")]
    real = {v.extracted_index for v in verdict.extracted if v.is_goal}
    trusted = {i for i, d in enumerate(goals) if d["verification"] in ("verified", "fuzzy")}
    return {
        "gold": len(gold),
        "gold_core": len(core),
        "extracted": len(goals),
        "recall": round(len(matched) / len(gold), 3),
        "recall_core": round(len(matched & set(core)) / len(core), 3) if core else None,
        "precision": round(len(real & trusted) / len(goals), 3),
        "is_goal_rate": round(len(real) / len(goals), 3),
        "verified_rate": round(len(trusted) / len(goals), 3),
        "candidate_page_recall": round(sum(g["page"] in pages for g in gold) / len(gold), 3),
        "missed": [gold[i]["label"] for i in range(len(gold)) if i not in matched],
        "not_goals": [goals[i]["label"] for i in range(len(goals)) if i not in real],
    }


def evaluate(experiment: str) -> dict[str, Any]:
    fte_gold = {g["report"]: g for g in load_yaml("extraction.yaml")["fte"]}
    goal_gold = {
        g["report"]: g["goals"] for g in load_yaml("extraction.yaml")["sustainability_goals"]
    }
    judge = AnthropicProvider(get_settings().model_copy(update={"anthropic_model": JUDGE_MODEL}))
    keys = {doc.path.name: doc.key for doc in load_corpus()}

    def one(report_id: str) -> tuple[str, dict]:
        with SessionLocal() as db:
            report = db.get(Report, report_id)
            key = keys[report.filename]
            items, pages = extracted(db, report, experiment)
        return key, {
            "fte": score_fte(fte_gold[key], items, pages["fte"]),
            "goals": score_goals(judge, goal_gold[key], items, pages["sustainability_goal"]),
        }

    with SessionLocal() as db:
        report_ids = [r.id for r in db.query(Report).all() if r.filename in keys]
    mark = USAGE.mark()
    with ThreadPoolExecutor(3) as pool:
        per_report = dict(pool.map(one, report_ids))
    reports = per_report.values()
    n = len(per_report)
    summary = {
        "fte_exact": round(sum(r["fte"]["outcome"] == "exact" for r in reports) / n, 3),
        "fte_exact_or_alternative": round(
            sum(r["fte"]["outcome"] in ("exact", "alternative") for r in reports) / n, 3
        ),
        "fte_verified": round(
            sum(r["fte"].get("verification") == "verified" for r in reports) / n, 3
        ),
        "goals_recall": round(sum(r["goals"]["recall"] for r in reports) / n, 3),
        "goals_recall_core": round(sum(r["goals"].get("recall_core") or 0 for r in reports) / n, 3),
        "goals_precision": round(sum(r["goals"].get("precision", 0) for r in reports) / n, 3),
        "goals_candidate_page_recall": round(
            sum(r["goals"].get("candidate_page_recall", 0) for r in reports) / n, 3
        ),
        "goals_extracted": sum(r["goals"]["extracted"] for r in reports),
    }
    return {
        "experiment": experiment,
        "summary": summary,
        "usage": USAGE.summary(mark),
        "reports": dict(sorted(per_report.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", choices=sorted(EXPERIMENTS))
    args = parser.parse_args()
    for name in args.experiments:
        result = evaluate(name)
        print(name, result["summary"])
        print(write_json(f"extraction_{name}.json", result))


if __name__ == "__main__":
    main()
