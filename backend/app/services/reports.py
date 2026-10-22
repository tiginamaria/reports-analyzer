"""Reports: upload, status, extraction retry, deletion, the PDF and its pages as images."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pymupdf
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.extraction.extractor import registry_entries
from app.ingestion import cancellation
from app.ingestion.pipeline import begin_reextraction
from app.models import ExtractionStatus, Report, ReportStatus
from app.repositories.reports import ReportRepository
from app.schemas import ReportDetail, ReportOut
from app.services.errors import ConflictError, InvalidRequestError, NotFoundError, UploadRejected
from app.services.storage import ReportStorage

log = logging.getLogger(__name__)

PAGE_RENDER_SCALE = 2.0  # 144 dpi: crisp on high-density screens at ~300 KB per page


@dataclass
class AcceptedUpload:
    report: Report
    path: Path  # where the PDF now lives; what the ingestion task reads


@dataclass
class ReextractionPlan:
    report: Report
    keys: list[str]  # registry entries to run again


def report_label(report: Report) -> str:
    """`Shell 2025`; company and year are empty until ingestion has detected them, so the
    file name is the fallback."""
    parts = [report.company, str(report.fiscal_year or "")]
    return " ".join(part for part in parts if part) or report.filename


class ReportService:
    def __init__(self, db: AsyncSession, storage: ReportStorage | None = None) -> None:
        self.db = db
        self.reports = ReportRepository(db)
        self.storage = storage or ReportStorage()

    async def list_all(self) -> list[Report]:
        return list(await self.reports.list())

    async def get(self, report_id: str, *, with_extractions: bool = False) -> Report:
        report = await self.reports.get(report_id, with_extractions=with_extractions)
        if report is None:
            raise NotFoundError("Report not found")
        return report

    async def detail(self, report_id: str) -> ReportDetail:
        report = await self.get(report_id, with_extractions=True)
        return ReportDetail(
            **ReportOut.model_validate(report).model_dump(),
            chunk_count=await self.reports.chunk_count(report_id),
            datapoint_count=await self.reports.datapoint_count(report_id),
        )

    async def accept_upload(self, source: BinaryIO, name: str) -> AcceptedUpload:
        """Store one upload under its hash and create its `pending` report row. The caller
        schedules the ingestion. Raises UploadRejected when the file is not a PDF, is
        already ingested or cannot be saved. The temporary file never outlives this call."""
        tmp = await asyncio.to_thread(self.storage.save_temporary, source)
        try:
            if not await asyncio.to_thread(self.storage.is_pdf, tmp):
                raise UploadRejected("Not a PDF file")
            digest = await asyncio.to_thread(self.storage.sha256, tmp)
            if existing := await self.reports.by_hash(digest):
                raise UploadRejected(f"Already ingested as {report_label(existing)}")
            stored = self.storage.store(tmp, digest)
        finally:
            tmp.unlink(missing_ok=True)

        report = Report(
            company="",  # detected during ingestion
            fiscal_year=0,
            filename=name,  # original name, for display and download
            file_hash=digest,
            status=ReportStatus.PENDING,
            progress=0,
        )
        self.reports.add(report)
        try:
            await self.db.commit()
        except SQLAlchemyError as exc:
            await self.db.rollback()
            # Two uploads of the same bytes at once: the other one won the unique hash.
            if existing := await self.reports.by_hash(digest):
                raise UploadRejected(f"Already ingested as {report_label(existing)}") from exc
            stored.unlink(missing_ok=True)  # no row refers to the file any more
            log.warning("Could not save report row for %s: %s", name, exc)
            raise UploadRejected("Could not save the report, see the backend log") from exc
        await self.db.refresh(report, ["extractions"])  # the response serialises them
        return AcceptedUpload(report, stored)

    async def plan_reextraction(self, report_id: str, keys: list[str] | None) -> ReextractionPlan:
        """Validate a retry and mark its entries pending; the caller schedules the run. With
        `keys` None every entry whose last attempt failed is retried."""
        report = await self.get(report_id, with_extractions=True)
        if report.status == ReportStatus.EXTRACTING:
            raise ConflictError("Extraction is already running")
        if report.status != ReportStatus.READY:
            raise ConflictError("Only an ingested report can be extracted again")
        if keys is None:
            wanted = [e.key for e in report.extractions if e.status == ExtractionStatus.ERROR]
            if not wanted:
                raise ConflictError("No extraction failed on this report")
        else:
            try:
                wanted = [spec.key for spec in registry_entries(keys)]
            except KeyError as exc:
                raise InvalidRequestError(exc.args[0]) from exc
        await begin_reextraction(self.db, report, wanted)
        return ReextractionPlan(report, wanted)

    async def delete(self, report_id: str) -> None:
        """Delete the report, its pages, chunks and datapoints, and the PDF on disk. An
        ingestion still running for it is cancelled: it stops before its next inference
        batch or write."""
        report = await self.get(report_id)
        digest = report.file_hash
        cancellation.request_cancel(report_id)  # before the row goes, so no write slips in between
        await self.reports.delete(report)
        await self.db.commit()
        self.storage.remove(digest)

    async def pdf_path(self, report_id: str) -> Path:
        report = await self.get(report_id)
        path = self.storage.path_for(report.file_hash)
        if not path.exists():
            raise NotFoundError("PDF file is missing on disk")
        return path

    async def render_page(self, report_id: str, page_no: int) -> bytes:
        """One page rendered as PNG, so the UI can show a cited page without downloading the
        PDF. Rendering is CPU work and runs in a worker thread."""
        path = await self.pdf_path(report_id)
        png = await asyncio.to_thread(_render_page, path, page_no)
        if png is None:
            raise NotFoundError("Page not found")
        return png


def _render_page(path: Path, page_no: int) -> bytes | None:
    with pymupdf.open(path) as pdf:
        if not 1 <= page_no <= pdf.page_count:
            return None
        matrix = pymupdf.Matrix(PAGE_RENDER_SCALE, PAGE_RENDER_SCALE)
        return pdf[page_no - 1].get_pixmap(matrix=matrix).tobytes("png")
