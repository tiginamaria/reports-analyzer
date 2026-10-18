"""Run the datapoint registry against an ingested report."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from app.extraction import prompts
from app.extraction.numbers import parse_number
from app.extraction.schemas import ExtractedItem, ExtractionResult, KeptItems, ReportMetadata
from app.llm import LLMError, LLMProvider
from app.models import Chunk, Datapoint, Report
from app.retrieval.search import RETRIEVAL_CANDIDATES, Hit, format_context, retrieve
from app.textutil import normalize
from app.verify import verify_against_chunk

log = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

# Metadata detection reads only the start of the report.
METADATA_PAGES = 3
METADATA_CHARS_PER_PAGE = 2500
# Defaults of the tunable fields of a registry entry (see `DatapointSpec`). A registry entry
# or an eval experiment overrides them per datapoint.
EXTRACTION_TOP_K = 10  # candidate chunks passed to the model
EXTRACTION_MIN_SCORE = 0.0  # reranker floor for candidates: keep every one, recall matters most
EXTRACTION_MAX_TOKENS = 64000  # output budget of one call; far above any goal list
PER_QUERY_MIN = 4  # at least this many chunks are kept from each retrieval query...
PER_QUERY_EXTRA = 2  # ...and `top_k / queries + this` otherwise, so the union can fill top_k


@dataclass
class DatapointSpec:
    """One entry of the registry: what to extract and how to find candidate chunks. The
    fields after `instructions` tune candidate retrieval and the model call; their defaults
    are the module constants above."""

    key: str
    label: str
    multi: bool
    queries: list[str]
    instructions: str
    top_k: int = EXTRACTION_TOP_K
    min_score: float = EXTRACTION_MIN_SCORE  # the chat's floor is `search.RERANK_MIN_SCORE`
    max_tokens: int = EXTRACTION_MAX_TOKENS
    per_query: int | None = None  # chunks kept per retrieval query; None derives it from top_k
    candidates: int = RETRIEVAL_CANDIDATES  # chunks each search fetches for reranking
    # For a multi-value entry: chunks per model call. One call over 24 chunks lists the
    # prominent goals and skims the rest; several calls over smaller batches read them all.
    # None sends every chunk in one call.
    batch_size: int | None = None


def load_registry(path: Path = REGISTRY_PATH) -> list[DatapointSpec]:
    with path.open() as f:
        entries = yaml.safe_load(f)
    return [DatapointSpec(**entry) for entry in entries]


def extract_report(db: Session, report: Report, provider: LLMProvider) -> list[Datapoint]:
    """Replace the report's datapoints with a fresh extraction of every registry entry."""
    db.query(Datapoint).filter(Datapoint.report_id == report.id).delete()
    created: list[Datapoint] = []
    for spec in load_registry():
        created.extend(_extract_datapoints(db, report, spec, provider))
    db.commit()
    return created


def _extract_datapoints(
    db: Session,
    report: Report,
    spec: DatapointSpec,
    provider: LLMProvider,
    hits: list[Hit] | None = None,
) -> list[Datapoint]:
    """Retrieve candidate chunks for one registry entry, ask the model, verify every quote.
    A failed LLM call is logged and yields nothing; the rest of the registry still runs.
    `hits` lets a caller that already retrieved the candidates pass them in."""
    if hits is None:
        hits = _candidate_hits(db, report, spec)
    if not hits:
        return []
    size = spec.batch_size if spec.multi and spec.batch_size else len(hits)
    items: list[ExtractedItem] = []
    for start in range(0, len(hits), size):
        try:
            result = provider.complete_json(
                system=prompts.EXTRACT,
                user=_extraction_prompt(report, spec, hits[start : start + size]),
                schema=ExtractionResult,
                max_tokens=spec.max_tokens,
            )
        except LLMError as exc:
            log.error("Extraction of %s failed for %s: %s", spec.key, report.company, exc)
            continue
        if result.found:
            items.extend(item for item in result.items if item.value)
    if not spec.multi:
        items = items[:1]
    items = list(_unique(items))
    if spec.multi and size < len(hits) and len(items) > 1:
        items = _consolidate(items, report, spec, provider)
    chunks_by_id = {hit.chunk.id: hit.chunk for hit in hits}
    datapoints = [
        _build_datapoint(db, report, spec, item, chunks_by_id.get(item.chunk_id), provider)
        for item in items
    ]
    db.add_all(datapoints)
    return datapoints


def _consolidate(
    items: list[ExtractedItem], report: Report, spec: DatapointSpec, provider: LLMProvider
) -> list[ExtractedItem]:
    """The reduce step after batched extraction: one short call over the collected items
    drops duplicates found by several batches and items that are not concrete commitments.
    The quotes are not re-sent; the model sees labels, values and periods. Keeps everything
    when the call fails."""
    listing = "\n".join(
        f"{i}. {item.label} | {item.value} {item.unit or ''} | {item.period or ''}"
        for i, item in enumerate(items)
    )
    user = (
        f"Report: {report.company}, fiscal year {report.fiscal_year}. "
        f"Datapoint: {spec.label}.\n\n{listing}"
    )
    try:
        kept = provider.complete_json(
            system=prompts.CONSOLIDATE, user=user, schema=KeptItems, max_tokens=2000
        )
    except LLMError as exc:
        log.warning("Consolidation of %s skipped for %s: %s", spec.key, report.company, exc)
        return items
    indices = [i for i in dict.fromkeys(kept.keep) if 0 <= i < len(items)]
    return [items[i] for i in indices] or items


def _extraction_prompt(report: Report, spec: DatapointSpec, hits: list[Hit]) -> str:
    return (
        f"Report: {report.company}, fiscal year {report.fiscal_year}.\n"
        f"Task: {spec.instructions.strip()}\n\n"
        f"Chunks:\n\n{format_context(hits)}"
    )


def _unique(items: list[ExtractedItem]) -> Iterator[ExtractedItem]:
    """Drop items the model returned twice: the same label, value and period, or the same
    quote (two batches can each find the same sentence)."""
    seen: set[str] = set()
    for item in items:
        keys = {normalize(f"{item.label}|{item.value}|{item.period}"), normalize(item.quote)}
        if not keys & seen:
            seen |= keys
            yield item


def _build_datapoint(
    db: Session,
    report: Report,
    spec: DatapointSpec,
    item: ExtractedItem,
    chunk: Chunk | None,
    provider: LLMProvider,
) -> Datapoint:
    """A datapoint row with its quote verified. `chunk` is None when the model cited a chunk
    id it was never given; such a quote cannot be checked and counts as failed."""
    if chunk is None:
        status = "failed"
    else:
        status = verify_against_chunk(db, chunk, item.quote)
    return Datapoint(
        report_id=report.id,
        key=spec.key,
        label=item.label or spec.label,
        value=item.value,
        numeric_value=parse_number(item.value),
        unit=_unit_without_repeated_percent(item.value, item.unit),
        period=item.period,
        qualifiers={q.name: q.value for q in item.qualifiers},
        quote=item.quote,
        chunk_id=chunk.id if chunk else None,
        page=chunk.page if chunk else None,
        verification=status,
        model=provider.model_name(),
    )


def _unit_without_repeated_percent(value: str | None, unit: str | None) -> str | None:
    """The unit minus its leading "%" when the value already ends with one: the model writes
    a goal as value "75%", unit "% reduction in scope 1 and 2 emissions", which would show as
    "75% % reduction in ...". A unit that was only "%" becomes None."""
    if not value or not unit or not value.rstrip().endswith("%"):
        return unit
    rest = unit.lstrip()
    if not rest.startswith("%"):
        return unit
    return rest[1:].strip() or None


def _candidate_hits(db: Session, report: Report, spec: DatapointSpec) -> list[Hit]:
    """Union of the best hits of every query, each chunk once with its highest score, cut to
    `top_k` and returned in page order so the model reads the report front to back."""
    per_query = spec.per_query or max(
        PER_QUERY_MIN, spec.top_k // len(spec.queries) + PER_QUERY_EXTRA
    )
    best_by_chunk: dict[str, Hit] = {}
    for query in spec.queries:
        hits = retrieve(
            db,
            query,
            report_ids=[report.id],
            candidates=spec.candidates,
            top_k=per_query,
            min_score=spec.min_score,
        )
        for hit in hits:
            current = best_by_chunk.get(hit.chunk.id)
            if current is None or current.score < hit.score:
                best_by_chunk[hit.chunk.id] = hit
    top = sorted(best_by_chunk.values(), key=lambda hit: hit.score, reverse=True)[: spec.top_k]
    return sorted(top, key=lambda hit: (hit.chunk.page, hit.chunk.id))


def detect_metadata(
    provider: LLMProvider,
    pages: list[str],
    filename: str,
    first_pages: int = METADATA_PAGES,
    chars_per_page: int = METADATA_CHARS_PER_PAGE,
) -> ReportMetadata:
    """Company, fiscal year and title, read by the small model from the first `first_pages`
    pages, `chars_per_page` characters of each."""
    sample = "\n\n".join(page[:chars_per_page] for page in pages[:first_pages])
    user = f"Filename: {filename}\n\nFirst pages:\n{sample}"
    return provider.complete_json(
        system=prompts.DETECT_METADATA, user=user, schema=ReportMetadata, small=True
    )
