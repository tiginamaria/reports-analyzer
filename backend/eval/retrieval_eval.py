"""Retrieval evaluation: stage ablation and parameter sweeps against the gold questions.

For every question the expensive parts are computed once and cached per index variant: the
query analysis (small LLM), deep vector and full-text rankings over the whole corpus, and
cross-encoder scores per (query, chunk). Every retrieval configuration is then replayed
offline from the caches, so a sweep over candidates, RRF k, top-k, rerank on/off and scope
costs no model calls. Scoped rankings are the global ones filtered to the scope: exact for
full-text (ts_rank_cd is per document) and for exact cosine search as long as the depth
covers the scope's candidates.

    uv run python -m eval.retrieval_eval                      # base index
    uv run python -m eval.retrieval_eval --variant words200   # an index built by variants.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy import text as sql
from sqlalchemy.orm import Session, sessionmaker

from app.chat.pipeline import _analyse, _resolve_scope
from app.chat.schemas import QueryAnalysis
from app.llm import ChatTurn, get_provider
from app.models import INDEXED_STATUSES, Report
from app.retrieval.embedder import get_embedder, get_reranker
from app.retrieval.search import rrf_merge
from app.textutil import normalize
from eval import EVAL_DATABASE_URL
from eval.common import CACHE_DIR, instrument, load_yaml, report_keys, write_json

DEPTH = 400  # chunks fetched per global ranking; deep enough to cover a one-report scope

FTS_SQL = {
    # the app's query: every term must match
    "fts": "websearch_to_tsquery('english', :q)",
    # any term may match, ranked by cover density
    "fts_or": "replace(plainto_tsquery('english', :q)::text, ' & ', ' | ')::tsquery",
}


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

    @property
    def turns(self) -> list[ChatTurn]:
        return [ChatTurn(turn["role"], turn["content"]) for turn in self.history]


def load_questions() -> list[Question]:
    fields = Question.__dataclass_fields__
    return [
        Question(**{k: v for k, v in item.items() if k in fields})
        for item in load_yaml("qa.yaml")["qa"]
    ]


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


def analyses(questions: list[Question]) -> dict[str, QueryAnalysis]:
    """Query analysis per question, as the chat pipeline would produce it (cached)."""
    cache = JsonCache("analysis.json")
    provider = get_provider()
    instrument(provider)
    for q in questions:
        key = f"{q.question}|{json.dumps(q.history)}"
        if key not in cache.data:
            cache.data[key] = _analyse(provider, q.question, q.turns).model_dump()
    cache.save()
    return {
        q.id: QueryAnalysis(**cache.data[f"{q.question}|{json.dumps(q.history)}"])
        for q in questions
    }


@dataclass
class ChunkInfo:
    report_id: str
    page: int
    kind: str
    content: str
    company: str
    year: int


class Index:
    """One index variant: chunk metadata in memory plus cached rankings and rerank scores."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.db = session_for(variant)
        rows = self.db.execute(
            sql(
                "SELECT c.id, c.report_id, c.page, c.kind, c.content, r.company, r.fiscal_year "
                "FROM chunks c JOIN reports r ON r.id = c.report_id"
            )
        )
        self.chunks = {row[0]: ChunkInfo(*row[1:]) for row in rows}
        self.rankings = JsonCache(f"rankings_{variant}.json")
        self.scores = JsonCache(f"rerank_{variant}.json")
        self.timings: dict[str, list[float]] = defaultdict(list)

    def ranking(self, method: str, query: str) -> list[str]:
        key = f"{method}|{query}"
        if key not in self.rankings.data:
            start = time.perf_counter()
            if method == "vector":
                vector = get_embedder().embed_query(query)
                rows = self.db.execute(
                    sql(
                        "SELECT id FROM chunks WHERE embedding IS NOT NULL "
                        "ORDER BY embedding <=> CAST(:vec AS vector) LIMIT :limit"
                    ),
                    {"vec": str(vector), "limit": DEPTH},
                )
            else:
                rows = self.db.execute(
                    sql(
                        f"SELECT c.id FROM chunks c, (SELECT {FTS_SQL[method]} AS q) t "
                        "WHERE c.tsv @@ t.q ORDER BY ts_rank_cd(c.tsv, t.q) DESC LIMIT :limit"
                    ),
                    {"q": query, "limit": DEPTH},
                )
            self.rankings.data[key] = [row[0] for row in rows]
            self.timings[method].append(time.perf_counter() - start)
        return self.rankings.data[key]

    def rerank(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """Cross-encoder probability per chunk, computing only the missing pairs. The passage
        is built exactly like `search._passage`."""
        cached = self.scores.data.setdefault(query, {})
        missing = [cid for cid in chunk_ids if cid not in cached]
        if missing:
            start = time.perf_counter()
            passages = [
                f"{self.chunks[cid].company} {self.chunks[cid].year}\n{self.chunks[cid].content}"
                for cid in missing
            ]
            for cid, score in zip(missing, get_reranker().score(query, passages), strict=True):
                cached[cid] = score
            self.timings["rerank_per_pair"].append((time.perf_counter() - start) / len(missing))
        return {cid: cached[cid] for cid in chunk_ids}

    def save(self) -> None:
        self.rankings.save()
        self.scores.save()


@dataclass(frozen=True)
class Config:
    name: str
    methods: tuple[str, ...] = ("vector", "fts")  # rankings fused with RRF
    rerank: bool = True
    candidates: int = 40
    top_k: int = 8
    rrf_k: int = 60
    per_report: bool = False  # balance the top_k across the reports in scope


BASELINE = Config("baseline")
TUNED = Config("tuned", candidates=60, top_k=12)


def configs() -> list[Config]:
    """Stage ablation, then one-parameter sweeps around the baseline."""
    out = [
        Config("vector", methods=("vector",), rerank=False),
        Config("fts", methods=("fts",), rerank=False),
        Config("fts_or", methods=("fts_or",), rerank=False),
        Config("hybrid", rerank=False),
        Config("hybrid_or", methods=("vector", "fts_or"), rerank=False),
        Config("vector+rerank", methods=("vector",)),
        BASELINE,
        Config("hybrid_or+rerank", methods=("vector", "fts_or")),
    ]
    for candidates in (10, 20, 60, 80):
        out.append(Config(f"candidates={candidates}", candidates=candidates))
        out.append(
            Config(
                f"or,candidates={candidates}", methods=("vector", "fts_or"), candidates=candidates
            )
        )
    for top_k in (3, 5, 12, 16):
        out.append(Config(f"top_k={top_k}", top_k=top_k))
        out.append(Config(f"or,top_k={top_k}", methods=("vector", "fts_or"), top_k=top_k))
    for rrf_k in (10, 30, 100):
        out.append(Config(f"rrf_k={rrf_k}", rrf_k=rrf_k))
    out += [
        Config("per_report", per_report=True),
        TUNED,
        Config("tuned+per_report", candidates=60, top_k=12, per_report=True),
        Config("tuned_or", methods=("vector", "fts_or"), candidates=60, top_k=12),
        Config("tuned_vector_only", methods=("vector",), candidates=60, top_k=12),
    ]
    return out


def retrieve(index: Index, cfg: Config, query: str, scope: set[str]) -> list[tuple[str, float]]:
    """Replay one configuration: scoped rankings, RRF, optional rerank, cut to top_k.

    With `per_report`, a scope of several reports (comparisons across companies or years) is
    searched one report at a time and each report is guaranteed an equal share of the top_k
    slots; the rest go to the best remaining chunks. Otherwise one report tends to take
    every slot and the other side of the comparison is missing."""
    if cfg.per_report and len(scope) > 1:
        single = dataclasses.replace(cfg, per_report=False)
        per = {report: retrieve(index, single, query, {report}) for report in scope}
        quota = max(1, cfg.top_k // len(scope))
        chosen = [hit for hits in per.values() for hit in hits[:quota]]
        rest = sorted((h for hits in per.values() for h in hits[quota:]), key=lambda p: -p[1])
        chosen += rest[: max(0, cfg.top_k - len(chosen))]
        return sorted(chosen, key=lambda p: -p[1])[: cfg.top_k]
    rankings = [
        [cid for cid in index.ranking(m, query) if index.chunks[cid].report_id in scope][
            : cfg.candidates
        ]
        for m in cfg.methods
    ]
    fused = rrf_merge(rankings, cfg.rrf_k)[: cfg.candidates]
    if cfg.rerank:
        scores = index.rerank(query, [cid for cid, _ in fused])
        fused = sorted(((cid, scores[cid]) for cid, _ in fused), key=lambda p: -p[1])
    return fused[: cfg.top_k]


def judge_hits(
    index: Index, q: Question, hits: list[tuple[str, float]], ids: dict[str, str]
) -> dict[str, Any]:
    """Per-question retrieval metrics. Gold pages come from the evidence; `facts` is the
    lenient check that every key fact appears somewhere in the retrieved text of the right
    reports (a figure is often restated on several pages)."""
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
        "best_score": hits[0][1] if hits else 0.0,
    }


def scope_for(
    mode: str, q: Question, analysis: QueryAnalysis, indexed: list[Report], ids: dict[str, str]
) -> set[str]:
    if mode == "oracle":
        return {ids[key] for key in q.reports}
    if mode == "pipeline":
        return {r.id for r in _resolve_scope(analysis, indexed, None)}
    return {r.id for r in indexed}


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
    }


def abstention_curve(answerable: list[float], unanswerable: list[float]) -> list[dict]:
    """For each threshold on the best rerank score: share of answerable questions wrongly
    abstained and share of unanswerable ones correctly abstained before the LLM call."""
    out = []
    for threshold in (0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5):
        out.append(
            {
                "threshold": threshold,
                "answerable_abstained": round(
                    sum(s < threshold for s in answerable) / max(len(answerable), 1), 3
                ),
                "unanswerable_abstained": round(
                    sum(s < threshold for s in unanswerable) / max(len(unanswerable), 1), 3
                ),
            }
        )
    return out


def evaluate(variant: str, modes: tuple[str, ...]) -> dict[str, Any]:
    questions = load_questions()
    analysis = analyses(questions)
    index = Index(variant)
    indexed = index.db.query(Report).filter(Report.status.in_(INDEXED_STATUSES)).all()
    ids = report_keys(index.db)
    labels = {r.id: f"{r.company} {r.fiscal_year}" for r in indexed}
    result: dict[str, Any] = {"variant": variant, "chunks": len(index.chunks), "modes": {}}
    per_question: dict[str, Any] = {}
    for mode in modes:
        table: dict[str, Any] = {}
        for cfg in configs():
            rows, unanswerable_scores = [], []
            for q in questions:
                query = analysis[q.id].standalone_question
                scope = scope_for(mode, q, analysis[q.id], indexed, ids)
                hits = retrieve(index, cfg, query, scope)
                metrics = judge_hits(index, q, hits, ids)
                if q.answerable:
                    rows.append({**metrics, "type": q.type, "modality": q.modality})
                else:
                    unanswerable_scores.append(metrics["best_score"])
                if cfg == BASELINE and mode == "pipeline":
                    per_question[q.id] = {
                        "query": query,
                        "scope": sorted(labels[r] for r in scope),
                        **metrics,
                        "hits": [
                            f"{index.chunks[c].company} {index.chunks[c].year} p{index.chunks[c].page}"
                            f" {s:.3f}"
                            for c, s in hits
                        ],
                    }
            entry = aggregate(rows)
            entry["by_type"] = {
                t: aggregate([r for r in rows if r["type"] == t])
                for t in sorted({r["type"] for r in rows})
            }
            entry["by_modality"] = {
                m: aggregate([r for r in rows if r["modality"] == m])
                for m in sorted({r["modality"] for r in rows})
            }
            if cfg.rerank:
                entry["abstention"] = abstention_curve(
                    [r["best_score"] for r in rows], unanswerable_scores
                )
            table[cfg.name] = entry
        index.save()
        result["modes"][mode] = table
    result["scope_accuracy"] = scope_accuracy(questions, analysis, indexed, ids)
    result["timings_ms"] = {
        k: round(1000 * statistics.median(v), 2) for k, v in index.timings.items() if v
    }
    result["questions"] = per_question
    return result


def scope_accuracy(
    questions: list[Question],
    analysis: dict[str, QueryAnalysis],
    indexed: list[Report],
    ids: dict[str, str],
) -> dict[str, Any]:
    """Does the pipeline's scope contain the gold reports, and how much else does it keep?"""
    contains, sizes, misses = 0, [], []
    answerable = [q for q in questions if q.answerable]
    for q in answerable:
        scope = scope_for("pipeline", q, analysis[q.id], indexed, ids)
        gold = {ids[key] for key in q.reports}
        if gold <= scope:
            contains += 1
        else:
            misses.append(q.id)
        sizes.append(len(scope))
    return {
        "contains_gold": round(contains / len(answerable), 3),
        "mean_scope_size": round(statistics.mean(sizes), 2),
        "misses": misses,
    }


def print_table(result: dict[str, Any]) -> None:
    for mode, table in result["modes"].items():
        print(f"\n== {result['variant']} / scope={mode}")
        print(f"{'config':24} {'hit@k':>6} {'mrr':>6} {'facts':>6} {'rprec':>6}")
        for name, m in table.items():
            print(
                f"{name:24} {m['page_hit']:6.3f} {m['mrr']:6.3f} "
                f"{m['facts'] or 0:6.3f} {m['report_precision']:6.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="base")
    parser.add_argument("--modes", default="pipeline,oracle,global")
    args = parser.parse_args()
    result = evaluate(args.variant, tuple(args.modes.split(",")))
    print_table(result)
    print(write_json(f"retrieval_{args.variant}.json", result))


if __name__ == "__main__":
    main()
