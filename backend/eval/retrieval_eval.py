"""Retrieval evaluation against the gold questions: the shipped pipeline, then a replay of
alternative configurations.

- `app` mode runs the retrieval code the chat uses, `_retrieve_all` with the real
  `retrieve`: one search per subquery of the rewrite, the app's candidates, fusion,
  reranker, score floor and per-subquery top-k, merged as the chat merges them. Nothing is
  cached, so it measures what the app retrieves today and how long it takes. Run it after
  any pipeline change.
- `all` and `oracle` modes replay the standalone question as one search through a sweep of
  configurations (stage ablation, candidates, top-k, RRF k), over all reports or over the
  gold reports only. The expensive parts are computed once per index variant and cached:
  the rewrite (small LLM), deep vector and full-text rankings, and cross-encoder scores per
  (query, chunk); after the first run a sweep costs no model calls. A scoped ranking is the
  global one filtered to the scope, exact as long as the depth covers the scope.

    uv run python -m eval.retrieval_eval                      # app mode and the sweeps
    uv run python -m eval.retrieval_eval --modes app          # the shipped pipeline only
    uv run python -m eval.retrieval_eval --variant words200   # an index built by variants.py
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy import text as sql
from sqlalchemy.orm import Session, sessionmaker

from app.chat import prompts
from app.chat.pipeline import _retrieve_all, _rewrite_query
from app.chat.schemas import Query
from app.ingestion.chunker import contextual_header
from app.llm import get_provider
from app.models import INDEXED_STATUSES, Message, Report
from app.retrieval.embedder import get_embedder, get_reranker
from app.retrieval.search import rrf_merge
from app.textutil import normalize
from eval import EVAL_DATABASE_URL
from eval.common import CACHE_DIR, instrument, load_yaml, report_keys, write_json

DEFAULT_MODES = "app,all,oracle"
DEPTH = 400  # chunks per cached global ranking; deep enough to cover a one-report scope

# tsquery constructions replayed as the full-text half of hybrid search
FTS_SQL = {
    "fts": "websearch_to_tsquery('english', :q)",  # every term must match
    "fts_or": "replace(plainto_tsquery('english', :q)::text, ' & ', ' | ')::tsquery",  # app
}

Hits = list[tuple[str, float]]  # (chunk id, score), best first


# --- gold questions -------------------------------------------------------------------------


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


def load_questions() -> list[Question]:
    fields = Question.__dataclass_fields__
    return [
        Question(**{k: v for k, v in item.items() if k in fields})
        for item in load_yaml("qa.yaml")["qa"]
    ]


# --- databases and caches -------------------------------------------------------------------


def variant_url(variant: str) -> str:
    if variant == "base":
        return EVAL_DATABASE_URL
    return EVAL_DATABASE_URL.rsplit("/", 1)[0] + f"/reports_eval_{variant}"


def session_for(variant: str) -> Session:
    return sessionmaker(bind=create_engine(variant_url(variant)))()


class JsonCache:
    def __init__(self, name: str) -> None:
        self.path = CACHE_DIR / name
        self.data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))


def analyses(questions: list[Question]) -> dict[str, Query]:
    """The rewrite of every question, as the chat produces it. Cached under a key derived
    from the rewrite prompt and schema, so a prompt change never replays stale rewrites."""
    signature = hashlib.sha1(
        (prompts.QUERY_REWRITE + json.dumps(Query.model_json_schema(), sort_keys=True)).encode()
    ).hexdigest()[:10]
    cache = JsonCache(f"analysis_{signature}.json")
    provider = get_provider()
    instrument(provider)
    rewrites = {}
    for q in questions:
        key = f"{q.question}|{json.dumps(q.history)}"
        if key not in cache.data:
            history = [Message(role=t["role"], content=t["content"]) for t in q.history]
            cache.data[key] = _rewrite_query(provider, q.question, history).model_dump()
        rewrites[q.id] = Query(**cache.data[key])
    cache.save()
    return rewrites


@dataclass
class ChunkInfo:
    report_id: str
    page: int
    content: str
    header: str
    label: str  # "ASML 2025 p12"


class Index:
    """One index variant: chunk metadata in memory, plus cached rankings and rerank scores
    for the replay modes."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.db = session_for(variant)
        rows = self.db.execute(
            sql(
                "SELECT c.id, c.report_id, c.page, c.content, r.company, r.fiscal_year "
                "FROM chunks c JOIN reports r ON r.id = c.report_id"
            )
        )
        self.chunks = {
            cid: ChunkInfo(
                report_id,
                page,
                content,
                contextual_header(company, year, page),
                f"{company} {year} p{page}",
            )
            for cid, report_id, page, content, company, year in rows
        }
        self.rankings = JsonCache(f"rankings_{variant}.json")
        self.scores = JsonCache(f"rerank_{variant}.json")
        self.timings: dict[str, list[float]] = defaultdict(list)

    def ranking(self, method: str, query: str) -> list[str]:
        """The best `DEPTH` chunk ids over the whole corpus for one search method."""
        key = f"{method}|{query}"
        if key not in self.rankings.data:
            start = time.perf_counter()
            if method == "vector":
                statement = (
                    "SELECT id FROM chunks WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> CAST(:q AS vector) LIMIT :limit"
                )
                params = {"q": str(get_embedder().embed_query(query)), "limit": DEPTH}
            else:
                statement = (
                    f"SELECT c.id FROM chunks c, (SELECT {FTS_SQL[method]} AS q) t "
                    "WHERE c.tsv @@ t.q ORDER BY ts_rank_cd(c.tsv, t.q) DESC LIMIT :limit"
                )
                params = {"q": query, "limit": DEPTH}
            self.rankings.data[key] = [row[0] for row in self.db.execute(sql(statement), params)]
            self.timings[method].append(time.perf_counter() - start)
        return self.rankings.data[key]

    def rerank(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """Cross-encoder probability per chunk, computing only the missing pairs, on the
        same passage the app scores (header, then content)."""
        cached = self.scores.data.setdefault(query, {})
        missing = [cid for cid in chunk_ids if cid not in cached]
        if missing:
            start = time.perf_counter()
            passages = [f"{self.chunks[cid].header}\n{self.chunks[cid].content}" for cid in missing]
            for cid, score in zip(missing, get_reranker().score(query, passages), strict=True):
                cached[cid] = score
            self.timings["rerank_per_pair"].append((time.perf_counter() - start) / len(missing))
        return {cid: cached[cid] for cid in chunk_ids}

    def save(self) -> None:
        self.rankings.save()
        self.scores.save()


# --- metrics --------------------------------------------------------------------------------


def score_hits(index: Index, q: Question, hits: Hits, ids: dict[str, str]) -> dict[str, Any]:
    """Per-question retrieval metrics against the gold evidence pages. `facts` is the lenient
    check that every key fact appears somewhere in the retrieved text of the right reports
    (a figure is often restated on several pages)."""
    gold_reports = {ids[key] for key in q.reports if key in ids}
    gold_pages = {(ids[e["report"]], e["page"]) for e in q.evidence if e["report"] in ids}
    located = [(index.chunks[cid].report_id, index.chunks[cid].page) for cid, _ in hits]
    first = next((rank for rank, loc in enumerate(located, 1) if loc in gold_pages), None)
    context = " ".join(
        normalize(index.chunks[cid].content)
        for cid, _ in hits
        if index.chunks[cid].report_id in gold_reports
    )
    return {
        "page_hit": first is not None,
        "mrr": 1 / first if first else 0.0,
        "page_recall": len(gold_pages & set(located)) / len(gold_pages) if gold_pages else 0.0,
        "facts": all(normalize(f) in context for f in q.key_facts) if q.key_facts else None,
        "report_precision": (
            sum(loc[0] in gold_reports for loc in located) / len(located) if located else 0.0
        ),
        # share of the gold reports with at least one chunk: a comparison needs both sides
        "report_recall": (
            len(gold_reports & {loc[0] for loc in located}) / len(gold_reports)
            if gold_reports
            else None
        ),
        "best_score": hits[0][1] if hits else 0.0,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(values), 3) if values else None

    return {
        "n": len(rows),
        "page_hit": mean("page_hit"),
        "mrr": mean("mrr"),
        "page_recall": mean("page_recall"),
        "facts": mean("facts"),
        "report_precision": mean("report_precision"),
        "report_recall": mean("report_recall"),
    }


def breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate over all rows, then per question type and per modality (text or table)."""
    entry = aggregate(rows)
    for group in ("type", "modality"):
        entry[f"by_{group}"] = {
            value: aggregate([r for r in rows if r[group] == value])
            for value in sorted({r[group] for r in rows})
        }
    return entry


def abstention_curve(answerable: list[float], unanswerable: list[float]) -> list[dict]:
    """For each threshold on the best rerank score: share of answerable questions wrongly
    abstained and share of unanswerable ones correctly abstained before the LLM call."""
    return [
        {
            "threshold": threshold,
            "answerable_abstained": round(
                sum(s < threshold for s in answerable) / max(len(answerable), 1), 3
            ),
            "unanswerable_abstained": round(
                sum(s < threshold for s in unanswerable) / max(len(unanswerable), 1), 3
            ),
        }
        for threshold in (0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5)
    ]


class Scorer:
    """Collects per-question metrics for one configuration and produces its result entry."""

    def __init__(self, index: Index, ids: dict[str, str]) -> None:
        self.index, self.ids = index, ids
        self.rows: list[dict[str, Any]] = []
        self.unanswerable_scores: list[float] = []

    def add(self, q: Question, hits: Hits) -> dict[str, Any]:
        metrics = score_hits(self.index, q, hits, self.ids)
        if q.answerable:
            self.rows.append({**metrics, "type": q.type, "modality": q.modality})
        else:
            self.unanswerable_scores.append(metrics["best_score"])
        return metrics

    def entry(self) -> dict[str, Any]:
        entry = breakdown(self.rows)
        entry["abstention"] = abstention_curve(
            [r["best_score"] for r in self.rows], self.unanswerable_scores
        )
        return entry


def labelled(index: Index, hits: Hits) -> list[str]:
    return [f"{index.chunks[cid].label} {score:.3f}" for cid, score in hits]


# --- the shipped pipeline -------------------------------------------------------------------


def evaluate_app(
    index: Index, questions: list[Question], rewrites: dict[str, Query], ids: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the chat's `_retrieve_all` on every question and score what it returns, with the
    time it takes. Prints a line per question, since every one runs the models."""
    scorer, per_question, seconds = Scorer(index, ids), {}, []
    get_embedder(), get_reranker()  # load before timing: the app loads them at startup too
    for q in questions:
        rewrite = rewrites[q.id]
        start = time.perf_counter()
        hits = [(hit.chunk.id, hit.score) for hit in _retrieve_all(index.db, rewrite, None)]
        seconds.append(time.perf_counter() - start)
        metrics = scorer.add(q, hits)
        print(
            f"{q.id}: {len(rewrite.subqueries)} searches, {len(hits)} chunks, "
            f"{seconds[-1]:.1f}s, gold page {'hit' if metrics['page_hit'] else 'missed'}",
            flush=True,
        )
        per_question[q.id] = {
            "query": rewrite.standalone_question,
            "searches": [sub.search_text for sub in rewrite.subqueries],
            "chunks": len(hits),
            "seconds": round(seconds[-1], 2),
            **metrics,
            "hits": labelled(index, hits),
        }
    entry = scorer.entry()
    entry["subqueries_mean"] = round(
        statistics.mean(len(rewrites[q.id].subqueries) for q in questions), 2
    )
    entry["chunks_mean"] = round(statistics.mean(r["chunks"] for r in per_question.values()), 1)
    entry["seconds_p50"] = round(statistics.median(seconds), 2)
    entry["seconds_p90"] = round(sorted(seconds)[int(0.9 * (len(seconds) - 1))], 2)
    entry["nothing_retrieved"] = [qid for qid, r in per_question.items() if r["chunks"] == 0]
    return entry, per_question


# --- replay of alternative configurations ---------------------------------------------------


@dataclass(frozen=True)
class Config:
    name: str
    methods: tuple[str, ...] = ("vector", "fts")  # rankings fused with RRF
    rerank: bool = True
    candidates: int = 40
    top_k: int = 8
    rrf_k: int = 60
    fts_weight: float = 1.0  # weight of the full-text ranking in the fusion
    per_report: bool = False  # balance the top_k across the reports in scope


BASELINE = Config("baseline")
HYBRID_OR = ("vector", "fts_or")


def configs() -> list[Config]:
    """Stage ablation, one-parameter sweeps around the baseline, then the tuned settings."""
    out = [
        Config("vector", methods=("vector",), rerank=False),
        Config("fts", methods=("fts",), rerank=False),
        Config("fts_or", methods=("fts_or",), rerank=False),
        Config("hybrid", rerank=False),
        Config("hybrid_or", methods=HYBRID_OR, rerank=False),
        Config("vector+rerank", methods=("vector",)),
        BASELINE,
        Config("hybrid_or+rerank", methods=HYBRID_OR),
    ]
    for candidates in (10, 20, 60, 80):
        out.append(Config(f"candidates={candidates}", candidates=candidates))
        out.append(Config(f"or,candidates={candidates}", methods=HYBRID_OR, candidates=candidates))
    for top_k in (3, 5, 12, 16):
        out.append(Config(f"top_k={top_k}", top_k=top_k))
        out.append(Config(f"or,top_k={top_k}", methods=HYBRID_OR, top_k=top_k))
    for top_k in (10, 12, 16, 18):  # the app's 60 candidates with more chunks kept
        out.append(
            Config(f"or,candidates=60,top_k={top_k}", methods=HYBRID_OR, candidates=60, top_k=top_k)
        )
    for rrf_k in (10, 30, 100):
        out.append(Config(f"rrf_k={rrf_k}", rrf_k=rrf_k))
    for weight in (0.5, 0.3):
        out.append(
            Config(
                f"or,candidates=60,fts_weight={weight}",
                methods=HYBRID_OR,
                candidates=60,
                fts_weight=weight,
            )
        )
    out += [
        Config("per_report", per_report=True),
        Config("tuned", candidates=60, top_k=12),
        Config("tuned+per_report", candidates=60, top_k=12, per_report=True),
        Config("tuned_or", methods=HYBRID_OR, candidates=60, top_k=12),
        Config(
            "tuned_or,fts_weight=0.5", methods=HYBRID_OR, candidates=60, top_k=12, fts_weight=0.5
        ),
        Config("tuned_vector_only", methods=("vector",), candidates=60, top_k=12),
    ]
    return out


def scope_for(mode: str, q: Question, indexed: list[Report], ids: dict[str, str]) -> set[str]:
    """Report ids a replay searches: every indexed report, or (`oracle`) the gold ones."""
    if mode == "oracle":
        return {ids[key] for key in q.reports}
    return {r.id for r in indexed}


def replay(index: Index, cfg: Config, query: str, scope: set[str]) -> Hits:
    """One configuration from the caches: scoped rankings, RRF, optional rerank, top_k.

    With `per_report`, a scope of several reports (comparisons across companies or years) is
    searched one report at a time and each report is guaranteed an equal share of the top_k
    slots; the rest go to the best remaining chunks."""
    if cfg.per_report and len(scope) > 1:
        single = dataclasses.replace(cfg, per_report=False)
        per = [replay(index, single, query, {report}) for report in scope]
        quota = max(1, cfg.top_k // len(scope))
        chosen = [hit for hits in per for hit in hits[:quota]]
        rest = sorted((hit for hits in per for hit in hits[quota:]), key=lambda p: -p[1])
        chosen += rest[: max(0, cfg.top_k - len(chosen))]
        return sorted(chosen, key=lambda p: -p[1])[: cfg.top_k]
    rankings = [
        [cid for cid in index.ranking(m, query) if index.chunks[cid].report_id in scope][
            : cfg.candidates
        ]
        for m in cfg.methods
    ]
    weights = tuple(cfg.fts_weight if m.startswith("fts") else 1.0 for m in cfg.methods)
    fused = rrf_merge(rankings, cfg.rrf_k, weights)[: cfg.candidates]
    if cfg.rerank:
        scores = index.rerank(query, [cid for cid, _ in fused])
        fused = sorted(((cid, scores[cid]) for cid, _ in fused), key=lambda p: -p[1])
    return fused[: cfg.top_k]


def evaluate_replay(
    index: Index,
    mode: str,
    questions: list[Question],
    rewrites: dict[str, Query],
    ids: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every configuration in one scope mode; per-question details for the baseline."""
    indexed = index.db.query(Report).filter(Report.status.in_(INDEXED_STATUSES)).all()
    table, per_question = {}, {}
    for cfg in configs():
        scorer = Scorer(index, ids)
        for q in questions:
            query = rewrites[q.id].standalone_question
            hits = replay(index, cfg, query, scope_for(mode, q, indexed, ids))
            metrics = scorer.add(q, hits)
            if cfg == BASELINE:
                per_question[q.id] = {"query": query, **metrics, "hits": labelled(index, hits)}
        table[cfg.name] = scorer.entry()
    index.save()
    return table, per_question


# --- run and report -------------------------------------------------------------------------


def evaluate(variant: str, modes: tuple[str, ...], limit: int | None = None) -> dict[str, Any]:
    questions = load_questions()[:limit]
    rewrites = analyses(questions)
    index = Index(variant)
    ids = report_keys(index.db)
    result: dict[str, Any] = {"variant": variant, "chunks": len(index.chunks), "modes": {}}
    for mode in modes:
        if mode == "app":
            entry, per_question = evaluate_app(index, questions, rewrites, ids)
            result["modes"]["app"] = {"current": entry}
        else:
            result["modes"][mode], per_question = evaluate_replay(
                index, mode, questions, rewrites, ids
            )
        result[f"questions_{mode}"] = per_question
    result["timings_ms"] = {
        k: round(1000 * statistics.median(v), 2) for k, v in index.timings.items() if v
    }
    return result


def print_table(result: dict[str, Any]) -> None:
    for mode, table in result["modes"].items():
        title = "shipped pipeline" if mode == "app" else f"replay, scope={mode}"
        print(f"\n== {result['variant']} / {title}")
        print(f"{'config':24} {'hit@k':>6} {'mrr':>6} {'facts':>6} {'rprec':>6} {'rrec':>6}")
        for name, m in table.items():
            print(
                f"{name:24} {m['page_hit']:6.3f} {m['mrr']:6.3f} {m['facts'] or 0:6.3f} "
                f"{m['report_precision']:6.3f} {m['report_recall'] or 0:6.3f}"
            )
        if mode == "app":
            m = table["current"]
            print(
                f"{m['subqueries_mean']} subqueries and {m['chunks_mean']} chunks per question; "
                f"{m['seconds_p50']} s p50, {m['seconds_p90']} s p90; "
                f"nothing retrieved for {m['nothing_retrieved'] or 'no question'}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="base")
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--limit", type=int, help="first N questions only (a smoke run)")
    args = parser.parse_args()
    modes = tuple(args.modes.split(","))
    result = evaluate(args.variant, modes, args.limit)
    print_table(result)
    # a partial run (some modes, or a few questions) never overwrites the full results file
    suffix = "" if args.modes == DEFAULT_MODES else "_" + "+".join(modes)
    if args.limit:
        suffix += f"_first{args.limit}"
    print(write_json(f"retrieval_{args.variant}{suffix}.json", result))


if __name__ == "__main__":
    main()
