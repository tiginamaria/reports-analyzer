"""Hybrid retrieval: pgvector cosine + Postgres full-text, fused with Reciprocal Rank Fusion,
then reranked with a cross-encoder."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.ingestion.chunker import contextual_header
from app.models import INDEXED_STATUSES, Chunk, Report
from app.retrieval.embedder import get_embedder, get_reranker

RRF_K = 60  # standard smoothing constant; higher values flatten the rank bonus
# Both rankings weigh the same in the fusion. The any-word full-text list always fills its
# share of the candidates and so displaces about half of the vector candidates; weighting it
# down to keep the vector depth was measured (eval `fts_weight` configs) to equal vector-only
# search and lose three questions, so the full-text-only candidates are worth their slots.
# Tuned on the eval set (docs/MVP_EVAL.md): 40 candidates missed answers ranked 41 to 60 by
# fusion.
RETRIEVAL_CANDIDATES = 60  # chunks fetched by each search and kept after fusion, for reranking
RERANK_TOP_K = 8  # chunks kept after reranking, per query
# Reranker probability below which a chunk is dropped. On the eval set the best chunk of every
# answerable question scored 0.95 or more, and questions about companies outside the corpus
# 0.10 to 0.18 (docs/MVP_EVAL.md), so this removes what such a query would otherwise return.
RERANK_MIN_SCORE = 0.2


@dataclass
class Hit:
    chunk: Chunk
    report: Report
    score: float  # RRF score, or the reranker probability once reranked


def retrieve(
    db: Session,
    query: str,
    report_ids: list[str] | None = None,
    top_k: int | None = None,
    min_score: float = RERANK_MIN_SCORE,
) -> list[Hit]:
    """The top chunks for a query, best first: vector and full-text search fetch
    `RETRIEVAL_CANDIDATES` chunks each, fused with RRF; the fused list is reranked, chunks
    scoring below `min_score` are dropped, and the rest is cut to `top_k` (default
    `RERANK_TOP_K`). Without a reranker nothing is dropped: RRF scores are not
    probabilities."""
    candidates = RETRIEVAL_CANDIDATES
    query_vector = get_embedder().embed_query(query)
    rankings = [
        vector_search(db, query_vector, candidates, report_ids),
        fulltext_search(db, query, candidates, report_ids),
    ]
    hits = _load_hits(db, rrf_merge(rankings)[:candidates])
    if rerank(query, hits):
        hits = [hit for hit in hits if hit.score >= min_score]
    return hits[: top_k or RERANK_TOP_K]


def vector_search(
    db: Session, query_vector: list[float], limit: int, report_ids: list[str] | None
) -> list[str]:
    """Chunk ids by cosine distance to the query vector, nearest first."""
    where, params = _report_filter(report_ids)
    rows = db.execute(
        sql(
            f"""
            SELECT c.id FROM chunks c
            WHERE {where}
            ORDER BY c.embedding <=> CAST(:vec AS vector)
            LIMIT :limit
            """
        ),
        {"vec": str(query_vector), "limit": limit, **params},
    )
    return [row[0] for row in rows]


def fulltext_search(db: Session, query: str, limit: int, report_ids: list[str] | None) -> list[str]:
    """Chunk ids by Postgres full-text rank, best first. A chunk matches any word of the query
    (`plainto_tsquery` with its ANDs turned into ORs, so text is never parsed as search
    syntax)."""
    where, params = _report_filter(report_ids)
    rows = db.execute(
        sql(
            f"""
            SELECT c.id FROM chunks c,
                 (SELECT replace(plainto_tsquery('english', :q)::text, ' & ', ' | ')::tsquery
                  AS q) t
            WHERE c.tsv @@ t.q AND {where}
            ORDER BY ts_rank_cd(c.tsv, t.q) DESC
            LIMIT :limit
            """
        ),
        {"q": query, "limit": limit, **params},
    )
    return [row[0] for row in rows]


def rrf_merge(
    rankings: list[list[str]], k: int = RRF_K, weights: tuple[float, ...] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked lists of ids, best first. `weights` scales
    each list's contribution (all 1.0 by default)."""
    weights = weights or (1.0,) * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def rerank(query: str, hits: list[Hit]) -> bool:
    """Replace RRF scores with cross-encoder probabilities and re-sort, best first. Returns
    False, leaving the hits as they are, when no reranker is configured."""
    reranker = get_reranker()
    if reranker is None:
        return False
    if hits:
        scores = reranker.score(query, [_with_header(hit) for hit in hits])
        for hit, score in zip(hits, scores, strict=True):
            hit.score = score
        hits.sort(key=lambda hit: hit.score, reverse=True)
    return True


def _report_filter(report_ids: list[str] | None) -> tuple[str, dict]:
    """SQL condition and bind params restricting a chunk query to indexed reports, and to
    `report_ids` if given. Reports still embedding, or that failed, have chunks too, and a
    filter naming only such reports finds nothing rather than everything."""
    where = "c.report_id IN (SELECT id FROM reports WHERE status = ANY(:indexed))"
    params: dict = {"indexed": [str(status) for status in INDEXED_STATUSES]}
    if report_ids:
        where += " AND c.report_id = ANY(:report_ids)"
        params["report_ids"] = report_ids
    return where, params


def _load_hits(db: Session, ranked: list[tuple[str, float]]) -> list[Hit]:
    """Turn (chunk id, score) pairs into hits with their chunk and report rows, keeping order."""
    rows = (
        db.query(Chunk, Report)
        .join(Report, Report.id == Chunk.report_id)
        .filter(Chunk.id.in_([chunk_id for chunk_id, _ in ranked]))
        .all()
    )
    by_id = {chunk.id: (chunk, report) for chunk, report in rows}
    return [Hit(*by_id[chunk_id], score) for chunk_id, score in ranked if chunk_id in by_id]


def _with_header(hit: Hit) -> str:
    """The chunk text behind its contextual header, as it was embedded; what the reranker
    scores."""
    report, chunk = hit.report, hit.chunk
    return f"{contextual_header(report.company, report.fiscal_year, chunk.page)}\n{chunk.content}"


def format_context(hits: list[Hit]) -> str:
    """Render hits as the answer model sees them: the citable chunk id, then the chunk with its
    header."""
    return "\n\n".join(f"[chunk {hit.chunk.id}] {_with_header(hit)}" for hit in hits)
