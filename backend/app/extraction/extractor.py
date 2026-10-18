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
from app.extraction.schemas import ExtractedItem, ExtractionResult, ReportMetadata
from app.llm import LLMError, LLMProvider
from app.models import Chunk, Datapoint, Report
from app.retrieval.search import Hit, format_context, retrieve
from app.textutil import normalize
from app.verify import verify_against_chunk

log = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

# Metadata detection reads only the start of the report.
METADATA_PAGES = 3
METADATA_CHARS_PER_PAGE = 2500


@dataclass
class DatapointSpec:
    """One entry of the registry: what to extract and how to find candidate chunks."""

    key: str
    label: str
    multi: bool
    queries: list[str]
    instructions: str
    top_k: int = 10


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
    db: Session, report: Report, spec: DatapointSpec, provider: LLMProvider
) -> list[Datapoint]:
    """Retrieve candidate chunks for one registry entry, ask the model, verify every quote.
    A failed LLM call is logged and yields nothing; the rest of the registry still runs."""
    hits = _candidate_hits(db, report, spec)
    if not hits:
        return []
    try:
        result = provider.complete_json(
            system=prompts.EXTRACT,
            user=_extraction_prompt(report, spec, hits),
            schema=ExtractionResult,
            max_tokens=6000,
        )
    except LLMError as exc:
        log.error("Extraction of %s failed for %s: %s", spec.key, report.company, exc)
        return []
    if not result.found:
        return []

    items = result.items if spec.multi else result.items[:1]
    chunks_by_id = {hit.chunk.id: hit.chunk for hit in hits}
    datapoints = [
        _build_datapoint(db, report, spec, item, chunks_by_id.get(item.chunk_id), provider)
        for item in _unique(items)
    ]
    db.add_all(datapoints)
    return datapoints


def _extraction_prompt(report: Report, spec: DatapointSpec, hits: list[Hit]) -> str:
    return (
        f"Report: {report.company}, fiscal year {report.fiscal_year}.\n"
        f"Task: {spec.instructions.strip()}\n\n"
        f"Chunks:\n\n{format_context(hits)}"
    )


def _unique(items: list[ExtractedItem]) -> Iterator[ExtractedItem]:
    """Drop items the model returned twice (same label, value and period)."""
    seen: set[str] = set()
    for item in items:
        key = normalize(f"{item.label}|{item.value}|{item.period}")
        if key not in seen:
            seen.add(key)
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
        unit=item.unit,
        period=item.period,
        qualifiers={q.name: q.value for q in item.qualifiers},
        quote=item.quote,
        chunk_id=chunk.id if chunk else None,
        page=chunk.page if chunk else None,
        verification=status,
        model=provider.model_name(),
    )


def _candidate_hits(db: Session, report: Report, spec: DatapointSpec) -> list[Hit]:
    """Union of the best hits of every query, each chunk once with its highest score, cut to
    `top_k` and returned in page order so the model reads the report front to back."""
    per_query = max(4, spec.top_k // len(spec.queries) + 2)
    best_by_chunk: dict[str, Hit] = {}
    for query in spec.queries:
        # no score floor: the report is known to be the right one, and goal recall needs
        # every candidate it can get
        for hit in retrieve(db, query, report_ids=[report.id], top_k=per_query, min_score=0.0):
            current = best_by_chunk.get(hit.chunk.id)
            if current is None or current.score < hit.score:
                best_by_chunk[hit.chunk.id] = hit
    top = sorted(best_by_chunk.values(), key=lambda hit: hit.score, reverse=True)[: spec.top_k]
    return sorted(top, key=lambda hit: (hit.chunk.page, hit.chunk.id))


def detect_metadata(provider: LLMProvider, pages: list[str], filename: str) -> ReportMetadata:
    """Company, fiscal year and title, read by the small model from the first pages."""
    sample = "\n\n".join(page[:METADATA_CHARS_PER_PAGE] for page in pages[:METADATA_PAGES])
    user = f"Filename: {filename}\n\nFirst pages:\n{sample}"
    return provider.complete_json(
        system=prompts.DETECT_METADATA, user=user, schema=ReportMetadata, small=True
    )
