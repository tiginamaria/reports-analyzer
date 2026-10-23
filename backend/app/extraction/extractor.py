"""Run the datapoint registry against an ingested report."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.extraction import prompts
from app.extraction.numbers import parse_number
from app.extraction.schemas import ExtractedItem, ExtractionResult, ReportMetadata
from app.ingestion.cancellation import IngestionCancelled
from app.llm import LLMProvider
from app.models import Chunk, Datapoint, Extraction, ExtractionStatus, Report
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
EXTRACTION_WORKERS = 6  # registry entries whose model calls run at the same time
PER_QUERY_MIN = 4  # at least this many chunks are kept from each retrieval query...
PER_QUERY_EXTRA = 2  # ...and `top_k / queries + this` otherwise, so the union can fill top_k
MAX_MESSAGE_LENGTH = 2000  # of the error stored on a failed entry


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


def load_registry(path: Path = REGISTRY_PATH) -> list[DatapointSpec]:
    with path.open() as f:
        entries = yaml.safe_load(f)
    return [DatapointSpec(**entry) for entry in entries]


def registry_entries(keys: Iterable[str] | None = None) -> list[DatapointSpec]:
    """The registry, or only the entries named in `keys`, in registry order. Raises KeyError
    for a key the registry does not have."""
    specs = load_registry()
    if keys is None:
        return specs
    wanted = set(keys)
    if unknown := wanted - {spec.key for spec in specs}:
        raise KeyError(f"Unknown datapoint keys: {', '.join(sorted(unknown))}")
    return [spec for spec in specs if spec.key in wanted]


Check = Callable[[], Awaitable[None]]
Progress = Callable[[int, int, str], Awaitable[None]]


async def extract_report(
    db: AsyncSession,
    report: Report,
    provider: LLMProvider,
    keys: Iterable[str] | None = None,
    check: Check | None = None,
    on_progress: Progress | None = None,
) -> list[Datapoint]:
    """Extract every registry entry (or only `keys`) of the report, replacing the entries'
    datapoints, and record an outcome per entry in `extractions`: `extracted`, `not_found`
    or `error` with the message.

    `check` runs before every retrieval and every write and may raise to stop the work
    (ingestion passes its cancellation check: a deleted report stops here without an
    outcome being recorded; a model call already in flight is left to finish).
    `on_progress(done, total, activity)` is called as entries advance.

    Candidate retrieval runs entry by entry (the reranker runs on one GPU behind a lock and
    the session is not thread-safe); the model calls, which are independent and spend their
    time waiting on the provider, run concurrently, `EXTRACTION_WORKERS` at a time;
    verification and storage follow entry by entry. A failed model call is recorded
    on its entry and the rest of the registry still runs. A failure outside one entry's
    call (the database, the reranker) marks every entry still pending as failed and
    propagates."""
    specs = registry_entries(keys)
    total = len(specs)

    async def stop_check() -> None:
        if check is not None:
            await check()

    async def progress(done: int, activity: str) -> None:
        if on_progress is not None:
            await on_progress(done, total, activity)

    await stop_check()
    outcomes = await start_extraction(db, report, [spec.key for spec in specs])
    try:
        await db.execute(
            delete(Datapoint).where(Datapoint.report_id == report.id, Datapoint.key.in_(outcomes))
        )
        hits: dict[str, list[Hit]] = {}
        for number, spec in enumerate(specs):
            await stop_check()
            await progress(0, f"Finding passages for {spec.label} ({number + 1} of {total})")
            hits[spec.key] = await _candidate_hits(db, report, spec)
        await stop_check()
        await progress(0, f"Asking the model for {total} datapoint(s)")
        answers = await _ask_model_for_all(report, specs, provider, hits)
        created: list[Datapoint] = []
        for number, spec in enumerate(specs):
            await stop_check()
            created.extend(
                await _finish_entry(
                    db,
                    report,
                    spec,
                    outcomes[spec.key],
                    answers[spec.key],
                    hits[spec.key],
                    provider,
                )
            )
            await progress(number + 1, f"Stored {spec.label} ({number + 1} of {total})")
        await stop_check()
        await db.commit()
        return created
    except IngestionCancelled:
        await db.rollback()  # the report is gone; nothing to record
        raise
    except Exception as exc:
        await db.rollback()
        await record_failure(db, report, exc, keys=list(outcomes))
        raise


async def _ask_model_for_all(
    report: Report, specs: list[DatapointSpec], provider: LLMProvider, hits: dict[str, list[Hit]]
) -> dict[str, list[ExtractedItem] | Exception]:
    """Every entry's model call, `EXTRACTION_WORKERS` in flight at once. Returns each
    entry's items, or the exception its call raised, so one failure never hides the other
    entries' results."""
    limit = asyncio.Semaphore(EXTRACTION_WORKERS)

    async def ask(spec: DatapointSpec) -> list[ExtractedItem]:
        async with limit:
            return await _ask_model(report, spec, provider, hits[spec.key])

    results = await asyncio.gather(*(ask(spec) for spec in specs), return_exceptions=True)
    answers: dict[str, list[ExtractedItem] | Exception] = {}
    for spec, result in zip(specs, results, strict=True):
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result  # cancellation and the like are not an entry's failure
        answers[spec.key] = result
    return answers


async def _extraction_rows(
    db: AsyncSession, report: Report, keys: list[str]
) -> dict[str, Extraction]:
    result = await db.execute(
        select(Extraction).where(Extraction.report_id == report.id, Extraction.key.in_(keys))
    )
    return {row.key: row for row in result.scalars()}


async def start_extraction(
    db: AsyncSession, report: Report, keys: Iterable[str]
) -> dict[str, Extraction]:
    """Set the entries to `pending` (creating their rows the first time) and commit, so the
    UI shows them running before the first model call returns."""
    keys = list(keys)
    rows = await _extraction_rows(db, report, keys)
    for key in keys:
        if key not in rows:
            rows[key] = Extraction(report_id=report.id, key=key)
            db.add(rows[key])
        rows[key].status = ExtractionStatus.PENDING
        rows[key].message = None
    await db.commit()
    return rows


async def record_failure(
    db: AsyncSession, report: Report, exc: Exception, keys: Iterable[str] | None = None
) -> None:
    """Mark every entry of the report (or of `keys`) that has no outcome yet, or is still
    pending, as failed with the error, and commit. Entries that already finished are kept."""
    message = _error_text(exc)
    keys = list(keys) if keys is not None else [spec.key for spec in load_registry()]
    rows = await _extraction_rows(db, report, keys)
    for key in keys:
        row = rows.get(key)
        if row is None:
            row = Extraction(report_id=report.id, key=key)
            db.add(row)
        elif row.status != ExtractionStatus.PENDING:
            continue
        row.status = ExtractionStatus.ERROR
        row.message = message
    await db.commit()


async def _finish_entry(
    db: AsyncSession,
    report: Report,
    spec: DatapointSpec,
    outcome: Extraction,
    answer: list[ExtractedItem] | Exception,
    hits: list[Hit],
    provider: LLMProvider,
) -> list[Datapoint]:
    """Store what the model returned for one entry and set the entry's outcome; a failed
    call becomes `error` with its message, so it is never mistaken for "nothing found"."""
    outcome.model = provider.model_name()
    if isinstance(answer, Exception):
        log.error(
            "Extraction of %s failed for %s %s",
            spec.key,
            report.company,
            report.fiscal_year,
            exc_info=answer,
        )
        outcome.status = ExtractionStatus.ERROR
        outcome.message = _error_text(answer)
        return []
    datapoints = await _store(db, report, spec, answer, hits, provider)
    outcome.status = ExtractionStatus.EXTRACTED if datapoints else ExtractionStatus.NOT_FOUND
    outcome.message = None
    return datapoints


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_MESSAGE_LENGTH]


async def _extract_datapoints(
    db: AsyncSession,
    report: Report,
    spec: DatapointSpec,
    provider: LLMProvider,
    hits: list[Hit] | None = None,
) -> list[Datapoint]:
    """One registry entry end to end, without an outcome row: retrieve candidate chunks
    (unless `hits` is given), ask the model, verify every quote and add the datapoints to the
    session. A failed model call raises. Used by the evaluation scripts."""
    if hits is None:
        hits = await _candidate_hits(db, report, spec)
    items = await _ask_model(report, spec, provider, hits)
    return await _store(db, report, spec, items, hits, provider)


async def _ask_model(
    report: Report, spec: DatapointSpec, provider: LLMProvider, hits: list[Hit]
) -> list[ExtractedItem]:
    """The model call for one entry over its candidate chunks. A failed call raises (an
    `LLMError` from the provider); the caller records it on the entry instead of treating
    it as "nothing found"."""
    if not hits:
        return []
    result = await provider.complete_json(
        system=prompts.EXTRACT,
        user=_extraction_prompt(report, spec, hits),
        schema=ExtractionResult,
        max_tokens=spec.max_tokens,
    )
    items = [item for item in result.items if item.value] if result.found else []
    if not spec.multi:
        items = items[:1]
    return list(_unique(items))


async def _store(
    db: AsyncSession,
    report: Report,
    spec: DatapointSpec,
    items: list[ExtractedItem],
    hits: list[Hit],
    provider: LLMProvider,
) -> list[Datapoint]:
    """Verify every item's quote against its chunk and add the datapoints to the session."""
    chunks_by_id = {hit.chunk.id: hit.chunk for hit in hits}
    datapoints = [
        await _build_datapoint(db, report, spec, item, chunks_by_id.get(item.chunk_id), provider)
        for item in items
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
    """Drop items the model returned twice: the same label, value and period, or the same
    quote (a target restated in two chunks)."""
    seen: set[str] = set()
    for item in items:
        keys = {normalize(f"{item.label}|{item.value}|{item.period}"), normalize(item.quote)}
        if not keys & seen:
            seen |= keys
            yield item


async def _build_datapoint(
    db: AsyncSession,
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
        status = await verify_against_chunk(db, chunk, item.quote)
    return Datapoint(
        report_id=report.id,
        key=spec.key,
        label=_fit("label", item.label or spec.label),
        value=_fit("value", item.value),
        numeric_value=parse_number(item.value),
        unit=_fit("unit", _unit_without_repeated_percent(item.value, item.unit)),
        period=_fit("period", item.period),
        qualifiers={q.name: q.value for q in item.qualifiers},
        quote=item.quote,
        chunk_id=chunk.id if chunk else None,
        page=chunk.page if chunk else None,
        verification=status,
        model=provider.model_name(),
    )


def _fit(column: str, text: str | None) -> str | None:
    """`text` cut to the length of the datapoints column, so a long qualifier the model wrote
    ("FY2025 (average during the year; no reliable year-end figure)") can never fail the
    insert and with it the whole ingestion."""
    limit = getattr(Datapoint.__table__.c[column].type, "length", None)
    return text if text is None or limit is None else text[:limit]


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


async def _candidate_hits(db: AsyncSession, report: Report, spec: DatapointSpec) -> list[Hit]:
    """Union of the best hits of every query, each chunk once with its highest score, cut to
    `top_k` and returned in page order so the model reads the report front to back."""
    per_query = spec.per_query or max(
        PER_QUERY_MIN, spec.top_k // len(spec.queries) + PER_QUERY_EXTRA
    )
    best_by_chunk: dict[str, Hit] = {}
    for query in spec.queries:
        hits = await retrieve(
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


async def detect_metadata(
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
    return await provider.complete_json(
        system=prompts.DETECT_METADATA, user=user, schema=ReportMetadata, small=True
    )
