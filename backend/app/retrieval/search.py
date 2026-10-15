"""Hybrid retrieval: pgvector cosine + Postgres full-text, fused with Reciprocal Rank Fusion,
then reranked with a cross-encoder."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Chunk, Report
from app.retrieval.embedder import get_embedder, get_reranker

RRF_K = 60  # standard smoothing constant; higher values flatten the rank bonus


@dataclass
class Hit:
    chunk: Chunk
    report: Report
    score: float  # RRF score, or the reranker probability once reranked


def retrieve(
    db: Session, query: str, report_ids: list[str] | None = None, top_k: int | None = None
) -> list[Hit]:
    """The top chunks for a query, best first: both searches fetch `retrieval_candidates`
    chunks, the fused list is reranked and cut to `top_k` (default `rerank_top_k`)."""
    settings = get_settings()
    candidates = settings.retrieval_candidates
    query_vector = get_embedder().embed_query(query)
    rankings = [
        vector_search(db, query_vector, candidates, report_ids),
        fulltext_search(db, query, candidates, report_ids),
    ]
    hits = _load_hits(db, rrf_merge(rankings)[:candidates])
    _rerank(query, hits)
    return hits[: top_k or settings.rerank_top_k]


def rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked lists of ids, best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def vector_search(
    db: Session, query_vector: list[float], limit: int, report_ids: list[str] | None
) -> list[str]:
    """Chunk ids by cosine distance to the query vector, nearest first."""
    where, params = _report_filter(report_ids)
    rows = db.execute(
        sql(
            f"""
            SELECT c.id FROM chunks c
            WHERE c.embedding IS NOT NULL {where}
            ORDER BY c.embedding <=> CAST(:vec AS vector)
            LIMIT :limit
            """
        ),
        {"vec": str(query_vector), "limit": limit, **params},
    )
    return [row[0] for row in rows]


def fulltext_search(db: Session, query: str, limit: int, report_ids: list[str] | None) -> list[str]:
    """Chunk ids by Postgres full-text rank, best first."""
    where, params = _report_filter(report_ids)
    rows = db.execute(
        sql(
            f"""
            SELECT c.id FROM chunks c,
                 websearch_to_tsquery('english', :q) AS q
            WHERE c.tsv @@ q {where}
            ORDER BY ts_rank_cd(c.tsv, q) DESC
            LIMIT :limit
            """
        ),
        {"q": query, "limit": limit, **params},
    )
    return [row[0] for row in rows]


def _report_filter(report_ids: list[str] | None) -> tuple[str, dict]:
    """SQL fragment and bind params restricting a chunk query to the given reports."""
    if report_ids:
        return "AND c.report_id = ANY(:report_ids)", {"report_ids": report_ids}
    return "", {}


def _load_hits(db: Session, ranked: list[tuple[str, float]]) -> list[Hit]:
    """Turn (chunk id, score) pairs into hits with their chunk and report rows, keeping order."""
    if not ranked:
        return []
    chunk_ids = [chunk_id for chunk_id, _ in ranked]
    chunks = {c.id: c for c in db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()}
    report_ids = {chunk.report_id for chunk in chunks.values()}
    reports = {r.id: r for r in db.query(Report).filter(Report.id.in_(report_ids)).all()}
    return [
        Hit(chunk=chunks[chunk_id], report=reports[chunks[chunk_id].report_id], score=score)
        for chunk_id, score in ranked
        if chunk_id in chunks
    ]


def _rerank(query: str, hits: list[Hit]) -> None:
    """Replace RRF scores with cross-encoder probabilities and re-sort, best first.
    A no-op when no reranker is configured."""
    reranker = get_reranker()
    if reranker is None or not hits:
        return
    scores = reranker.score(query, [_passage(hit) for hit in hits])
    for hit, score in zip(hits, scores, strict=True):
        hit.score = score
    hits.sort(key=lambda hit: hit.score, reverse=True)


def _passage(hit: Hit) -> str:
    """What the reranker sees: a short header naming the report, then the chunk text."""
    return f"{hit.report.company} {hit.report.fiscal_year}\n{hit.chunk.content}"


def format_context(hits: list[Hit]) -> str:
    """Render hits as the model sees them: a header with the citable chunk id, then the text."""
    return "\n\n".join(f"{_context_header(hit)}\n{hit.chunk.content}" for hit in hits)


def _context_header(hit: Hit) -> str:
    chunk, report = hit.chunk, hit.report
    return f"[chunk {chunk.id}] {report.company} | Annual Report {report.fiscal_year} | p. {chunk.page}"
