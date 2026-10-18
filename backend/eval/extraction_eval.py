"""Extraction evaluation: run the registry on every eval report as ingestion does, score each
datapoint against the gold labels, and account for what the stage costs.

For every report and registry entry: how many candidate chunks went to the model and whether
a gold page was among them, the call's tokens, seconds and cost, and the outcome. A
single-value datapoint is `exact`, an acceptable `alternative` (another basis the report
also states), `wrong` or `missing`; where the report states none (CM.com pays no dividend)
it is `none` when nothing was extracted and `spurious` otherwise. Sustainability goals are
matched to the gold list by an LLM judge: recall over all gold goals and over the headline
(`core`) ones, precision, and candidate-page recall. Reports run one after another, so the
tokens and cost of each call are attributed to the entry that made it.

    uv run python -m eval.extraction_eval              # run the registry as it is
    uv run python -m eval.extraction_eval --stored     # score what ingestion wrote, no model calls

To measure a change to an entry, change it in registry.yaml and run again.

Nothing is written to the database: each report's extraction is rolled back.
"""

import argparse
import re
import statistics
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

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
from app.retrieval.embedder import get_embedder, get_reranker
from eval.common import USAGE, instrument, load_corpus, load_yaml, write_json
from eval.e2e_eval import JUDGE_MODEL

# Two numbers are the same figure when they agree to 0.01% or to half a cent, after the unit's
# scale word (thousand, million, billion) is applied to each. An absolute tolerance alone
# would call 1.24 and 1.54 per share the same dividend.
RELATIVE_TOLERANCE = 1e-4
ABSOLUTE_TOLERANCE = 0.005
SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mln": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
}
ACCEPTABLE = {"exact", "alternative", "none"}  # outcomes that count as a correct extraction

# --- running one registry entry -------------------------------------------------------------


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


def run_entry(
    db: Session, report: Report, spec: DatapointSpec, provider: LLMProvider
) -> tuple[list[dict], dict[str, Any]]:
    """One registry entry on one report, as ingestion runs it: candidate retrieval, the
    model call, quote verification. Returns the datapoints and the run's accounting."""
    start = time.perf_counter()
    hits = _candidate_hits(db, report, spec)
    mark = USAGE.mark()
    items = [as_dict(dp) for dp in _extract_datapoints(db, report, spec, provider, hits)]
    usage = USAGE.summary(mark)
    seconds = round(time.perf_counter() - start, 1)
    return items, {
        "chunks": len(hits),
        "candidate_pages": sorted({hit.chunk.page for hit in hits}),
        "calls": usage["calls"],  # 2 means the repair retry ran
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": usage["cost_usd"],
        "seconds": seconds,
        "llm_seconds": usage["llm_seconds"],
        "retrieval_seconds": round(seconds - usage["llm_seconds"], 1),
    }


def stored_entry(
    db: Session, report: Report, spec: DatapointSpec
) -> tuple[list[dict], dict[str, Any]]:
    """What ingestion wrote for one entry, with today's candidates for the page check."""
    hits = _candidate_hits(db, report, spec)
    rows = db.query(Datapoint).filter(Datapoint.report_id == report.id, Datapoint.key == spec.key)
    return [as_dict(dp) for dp in rows], {
        "chunks": len(hits),
        "candidate_pages": sorted({hit.chunk.page for hit in hits}),
    }


# --- scoring --------------------------------------------------------------------------------


def scaled(value: float | None, unit: str | None) -> float | None:
    """A figure in absolute terms: 8,716 with unit "EUR million" is 8.716e9. Lets "2.3
    billion" and "2,252 million" be compared, whichever way the model wrote it."""
    if value is None:
        return None
    words = (unit or "").lower().replace("€", " ").replace("/", " ").split()
    return value * next((SCALES[w] for w in words if w in SCALES), 1.0)


def same_figure(value: float | None, target: float | None) -> bool:
    if value is None or target is None:
        return False
    return abs(value - target) <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * abs(target))


def score_single(
    key: str, gold: dict, items: list[dict], candidate_pages: list[int], fiscal_year: int
) -> dict:
    """One single-value datapoint against its gold label (see the module docstring). Besides
    the outcome: was the gold page among the candidates, does the quote come from it, does
    the period name the fiscal year, does the unit carry the gold's scale word."""
    item = next((d for d in items if d["key"] == key), None)
    result = {
        "gold": gold["value"],
        "gold_page": gold["page"],
        "gold_page_in_candidates": gold["page"] in candidate_pages,
    }
    if gold["numeric"] is None:
        outcome = "none" if item is None else "spurious"
    elif item is None:
        outcome = "missing"
    else:
        value = scaled(item["numeric_value"], item["unit"])
        # the figure as written matches, or it matches once both units' scales are applied
        # (a missing scale word is reported separately as `unit_has_scale`)
        if same_figure(item["numeric_value"], gold["numeric"]) or same_figure(
            value, scaled(gold["numeric"], gold.get("unit"))
        ):
            outcome = "exact"
        elif any(
            same_figure(value, scaled(alt["numeric"], alt.get("unit")))
            or same_figure(item["numeric_value"], alt["numeric"])
            for alt in gold.get("alternatives", [])
        ):
            outcome = "alternative"
        else:
            outcome = "wrong"
    if item is None:
        return {**result, "outcome": outcome}
    gold_scale = next((w for w in (gold.get("unit") or "").lower().split() if w in SCALES), None)
    return {
        **result,
        "outcome": outcome,
        "cited_gold_page": item["page"] == gold["page"],  # the quote comes from the gold page
        "period_has_year": str(fiscal_year) in (item["period"] or ""),
        "unit_has_scale": gold_scale is None or gold_scale in (item["unit"] or "").lower(),
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


def dated_before(year: Any, fiscal_year: int) -> bool:
    """Whether a gold goal's target year is in or before the fiscal year: a result by the
    time the report is written, which the registry tells the model to skip."""
    years = re.findall(r"\b(20\d\d)\b", str(year or ""))
    return bool(years) and max(int(y) for y in years) <= fiscal_year


def score_goals(
    judge: LLMProvider,
    gold: list[dict],
    items: list[dict],
    candidate_pages: list[int],
    fiscal_year: int,
) -> dict:
    """Goals against the gold list, judged by an LLM. Gold goals dated in or before the
    fiscal year are left out, as the registry asks the model to leave them out."""
    excluded = [g for g in gold if dated_before(g["year"], fiscal_year)]
    gold = [g for g in gold if g not in excluded]
    goals = [d for d in items if d["key"] == "sustainability_goal"]
    page_recall = round(sum(g["page"] in candidate_pages for g in gold) / max(len(gold), 1), 3)
    extracted = [
        {"label": d["label"], "value": d["value"], "period": d["period"], "page": d["page"]}
        for d in goals
    ]
    if not goals:
        return {
            "gold": len(gold),
            "gold_excluded_dated": len(excluded),
            "extracted": 0,
            "recall": 0.0,
            "recall_core": 0.0,
            "precision": None,
            "candidate_page_recall": page_recall,
            "missed": [g["label"] for g in gold],
            "extracted_items": extracted,
        }
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
    # one extracted item can satisfy one gold goal: a second gold goal mapped to the same
    # item (the judge is asked not to, but may) does not count as found
    matched: set[int] = set()
    used: set[int] = set()
    for match in verdict.matches:
        if match.extracted_index is not None and match.extracted_index not in used:
            matched.add(match.gold_index)
            used.add(match.extracted_index)
    core = [i for i, g in enumerate(gold) if g.get("core")]
    real = {v.extracted_index for v in verdict.extracted if v.is_goal}
    trusted = {i for i, d in enumerate(goals) if d["verification"] in ("verified", "fuzzy")}
    return {
        "gold": len(gold),
        "gold_excluded_dated": len(excluded),
        "gold_core": len(core),
        "extracted": len(goals),
        "extracted_items": extracted,
        "recall": round(len(matched) / len(gold), 3),
        "recall_core": round(len(matched & set(core)) / len(core), 3) if core else None,
        "precision": round(len(real & trusted) / len(goals), 3),
        "verified_rate": round(len(trusted) / len(goals), 3),
        "candidate_page_recall": page_recall,
        "missed": [gold[i]["label"] for i in range(len(gold)) if i not in matched],
        "not_goals": [goals[i]["label"] for i in range(len(goals)) if i not in real],
    }


# --- the run --------------------------------------------------------------------------------


def evaluate(stored: bool) -> dict[str, Any]:
    specs = load_registry()
    single_keys = [spec.key for spec in specs if not spec.multi]
    gold = load_yaml("extraction.yaml")
    single_gold = {key: {g["report"]: g for g in gold[key]} for key in single_keys}
    goal_gold = {g["report"]: g["goals"] for g in gold["sustainability_goals"]}
    keys = {doc.path.name: doc.key for doc in load_corpus()}
    years = {doc.key: doc.fiscal_year for doc in load_corpus()}
    judge = AnthropicProvider(get_settings().model_copy(update={"anthropic_model": JUDGE_MODEL}))
    provider = get_provider()
    instrument(provider)
    get_embedder()
    get_reranker()

    per_report: dict[str, dict[str, Any]] = {}
    with SessionLocal() as db:
        reports = [
            r
            for r in db.query(Report).order_by(Report.company, Report.fiscal_year)
            if r.filename in keys
        ]
        for report in reports:
            key = keys[report.filename]
            runs: dict[str, dict[str, Any]] = {}
            for spec in specs:
                items, run = (
                    stored_entry(db, report, spec)
                    if stored
                    else run_entry(db, report, spec, provider)
                )
                runs[spec.key] = {**run, "items": items}
                if not stored:
                    print(
                        f"{report.company} {report.fiscal_year} {spec.key}: {len(items)} items "
                        f"from {run['chunks']} chunks, {run['input_tokens']} in / "
                        f"{run['output_tokens']} out tokens, ${run['cost_usd']:.3f}, "
                        f"{run['seconds']} s",
                        flush=True,
                    )
            db.rollback()  # the fresh extraction never replaces what ingestion wrote
            scored = {
                k: {
                    **score_single(
                        k,
                        single_gold[k][key],
                        runs[k]["items"],
                        runs[k]["candidate_pages"],
                        years[key],
                    ),
                    **{f: v for f, v in runs[k].items() if f != "items"},
                }
                for k in single_keys
            }
            goals_run = runs["sustainability_goal"]
            scored["goals"] = {
                **score_goals(
                    judge,
                    goal_gold[key],
                    goals_run["items"],
                    goals_run["candidate_pages"],
                    years[key],
                ),
                **{f: v for f, v in goals_run.items() if f != "items"},
            }
            per_report[key] = scored
    return {
        "run": "stored" if stored else "current",
        "summary": summarize(per_report, single_keys, stored),
        "reports": per_report,
    }


def summarize(per_report: dict[str, dict], single_keys: list[str], stored: bool) -> dict[str, Any]:
    reports = list(per_report.values())
    n = len(reports)

    def share(rows: list[dict], test: Callable[[dict], Any]) -> float:
        return round(sum(bool(test(r)) for r in rows) / len(rows), 3)

    def mean(rows: list[dict], field: str) -> float | None:
        values = [r[field] for r in rows if r.get(field) is not None]
        return round(statistics.mean(values), 1) if values else None

    def cost(rows: list[dict]) -> dict[str, Any]:
        if stored:
            return {"chunks_mean": mean(rows, "chunks")}
        return {
            "chunks_mean": mean(rows, "chunks"),
            "input_tokens_mean": mean(rows, "input_tokens"),
            "output_tokens_mean": mean(rows, "output_tokens"),
            "seconds_mean": mean(rows, "seconds"),
            "llm_seconds_mean": mean(rows, "llm_seconds"),
            "repair_retries": sum(r["calls"] > 1 for r in rows),  # replies that failed once
            "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 4),
        }

    datapoints = {}
    for key in single_keys:
        rows = [r[key] for r in reports]
        datapoints[key] = {
            "n": n,
            "exact": share(rows, lambda r: r["outcome"] == "exact"),
            "acceptable": share(rows, lambda r: r["outcome"] in ACCEPTABLE),
            "outcomes": {
                o: sum(r["outcome"] == o for r in rows)
                for o in ("exact", "alternative", "none", "wrong", "missing", "spurious")
            },
            "gold_page_in_candidates": share(rows, lambda r: r["gold_page_in_candidates"]),
            "cited_gold_page": share(rows, lambda r: r.get("cited_gold_page", False)),
            "period_has_year": share(rows, lambda r: r.get("period_has_year", False)),
            "unit_has_scale": share(rows, lambda r: r.get("unit_has_scale", False)),
            "verified": share(rows, lambda r: r.get("verification") == "verified"),
            **cost(rows),
        }
    goal_rows = [r["goals"] for r in reports]

    def mean_of(field: str) -> float:  # over the reports where the figure is defined
        values = [r[field] for r in goal_rows if r.get(field) is not None]
        return round(statistics.mean(values), 3) if values else 0.0

    goals = {
        "recall": mean_of("recall"),
        "recall_core": mean_of("recall_core"),
        "precision": mean_of("precision"),
        "candidate_page_recall": round(
            statistics.mean(r["candidate_page_recall"] for r in goal_rows), 3
        ),
        "extracted_total": sum(r["extracted"] for r in goal_rows),
        "gold_total": sum(r["gold"] for r in goal_rows),
        "gold_excluded_dated_total": sum(r["gold_excluded_dated"] for r in goal_rows),
        **cost(goal_rows),
    }
    totals: dict[str, Any] = {"reports": n}
    if not stored:
        all_rows = [r[k] for r in reports for k in [*single_keys, "goals"]]
        totals |= {
            "cost_usd_total": round(sum(r["cost_usd"] for r in all_rows), 4),
            "cost_usd_per_report": round(sum(r["cost_usd"] for r in all_rows) / n, 4),
            "input_tokens_per_report": round(sum(r["input_tokens"] for r in all_rows) / n),
            "output_tokens_per_report": round(sum(r["output_tokens"] for r in all_rows) / n),
            "seconds_per_report": round(sum(r["seconds"] for r in all_rows) / n, 1),
        }
    return {"datapoints": datapoints, "goals": goals, "totals": totals}


def print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(f"\n== extraction / {result['run']}")
    print(
        f"{'datapoint':20} {'exact':>6} {'accept':>6} {'gold pg':>7} {'cited':>6} {'year':>5} "
        f"{'scale':>5} {'chunks':>6} {'in tok':>7} {'out tok':>7} {'$ total':>8} {'s':>5}"
    )
    for key, m in summary["datapoints"].items():
        print(
            f"{key:20} {m['exact']:6.2f} {m['acceptable']:6.2f} {m['gold_page_in_candidates']:7.2f} "
            f"{m['cited_gold_page']:6.2f} {m['period_has_year']:5.2f} {m['unit_has_scale']:5.2f} "
            f"{m['chunks_mean'] or 0:6.1f} {m.get('input_tokens_mean') or 0:7.0f} "
            f"{m.get('output_tokens_mean') or 0:7.0f} {m.get('cost_usd_total') or 0:8.3f} "
            f"{m.get('seconds_mean') or 0:5.0f}"
        )
    print(
        "  gold pg: gold page among the candidates; cited: quote taken from it; year: period "
        "names the fiscal year; scale: unit carries the scale word"
    )
    g = summary["goals"]
    print(
        f"{'goals':20} recall {g['recall']:.2f} (core {g['recall_core']:.2f}) precision "
        f"{g['precision']:.2f} gold pages in candidates {g['candidate_page_recall']:.2f}; "
        f"{g['extracted_total']} extracted for {g['gold_total']} gold; {g['chunks_mean']} chunks, "
        f"{g.get('input_tokens_mean') or 0:.0f} in / {g.get('output_tokens_mean') or 0:.0f} out "
        f"tokens, ${g.get('cost_usd_total') or 0:.3f}"
    )
    t = summary["totals"]
    if "cost_usd_total" in t:
        print(
            f"per report: ${t['cost_usd_per_report']:.3f}, {t['input_tokens_per_report']} in / "
            f"{t['output_tokens_per_report']} out tokens, {t['seconds_per_report']} s; "
            f"{t['reports']} reports, ${t['cost_usd_total']:.3f} in all"
        )
    for key, report in result["reports"].items():
        wrong = [
            f"{k}={v['outcome']}"
            for k, v in report.items()
            if k != "goals" and v["outcome"] not in ACCEPTABLE
        ]
        if wrong or report["goals"]["missed"]:
            print(
                f"  {key}: {' '.join(wrong) or 'datapoints ok'}; goals missed: {len(report['goals']['missed'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stored", action="store_true", help="score what ingestion wrote")
    args = parser.parse_args()
    result = evaluate(args.stored)
    print_summary(result)
    print(write_json(f"extraction_{result['run']}.json", result))


if __name__ == "__main__":
    main()
