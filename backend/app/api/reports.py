"""Report upload, status, extraction retry and deletion; the PDF and its pages as images."""

import hashlib
import logging
import shutil
import uuid
from pathlib import Path

import pymupdf
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.config import get_settings
from app.db import get_db
from app.extraction.extractor import registry_entries
from app.ingestion import cancellation
from app.ingestion.pipeline import begin_reextraction, ingest_report, rerun_extraction
from app.models import Chunk, Datapoint, ExtractionStatus, Report, ReportStatus
from app.schemas import ExtractRequest, RejectedFile, ReportDetail, ReportOut, UploadResult

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["reports"])

PDF_MAGIC = b"%PDF-"
PAGE_RENDER_SCALE = 2.0  # 144 dpi: crisp on high-density screens at ~300 KB per page
PAGE_IMAGE_CACHE = "public, max-age=86400"  # a report's pages never change


class UploadRejected(Exception):
    """A file in a batch was not accepted; the message is the reason shown to the user."""


def _reports_dir() -> Path:
    d = get_settings().reports_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stored_report_path(digest: str) -> Path:
    """PDFs are stored under their SHA-256, so the same bytes always map to one file."""
    return _reports_dir() / f"{digest}.pdf"


def _tmp_report_path() -> Path:
    """Uploads land under a random name until their hash is known."""
    return _reports_dir() / f"{uuid.uuid4()}.upload"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_pdf(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(len(PDF_MAGIC)) == PDF_MAGIC


def _report_by_hash(db: Session, digest: str) -> Report | None:
    return db.query(Report).filter(Report.file_hash == digest).first()


def _report_label(report: Report) -> str:
    """`Shell 2025`; company and year are empty until ingestion has detected them, so the
    file name is the fallback."""
    parts = [report.company, str(report.fiscal_year or "")]
    return " ".join(part for part in parts if part) or report.filename


def _get_report_or_404(db: Session, report_id: str) -> Report:
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    return report


@router.get("", response_model=list[ReportOut])
def list_reports(db: Session = Depends(get_db)) -> list[Report]:
    return (
        db.query(Report)
        .options(selectinload(Report.extractions))
        .order_by(Report.company, Report.fiscal_year.desc(), Report.filename)
        .all()
    )


@router.post("", response_model=UploadResult)
def upload_reports(
    background: BackgroundTasks,
    response: Response,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> UploadResult:
    """Upload one or more PDFs. Each accepted file starts its own ingestion, which detects
    the company and fiscal year from the report itself. Files that are not PDFs, are
    already ingested or cannot be saved are listed in `rejected` without failing the batch.
    """
    started: list[ReportOut] = []
    rejected: list[RejectedFile] = []
    for file in files:
        name = Path(file.filename or "").name or "upload.pdf"
        try:
            report = _accept_upload(db, background, file, name)
        except UploadRejected as exc:
            rejected.append(RejectedFile(filename=name, reason=str(exc)))
        else:
            started.append(ReportOut.model_validate(report))
    response.status_code = status.HTTP_201_CREATED if started else status.HTTP_200_OK
    return UploadResult(started=started, rejected=rejected)


def _accept_upload(
    db: Session,
    background: BackgroundTasks,
    file: UploadFile,
    name: str,
) -> Report:
    """Store one upload under its hash and start ingestion.

    Raises UploadRejected when the file is not a PDF or is already ingested. The temporary
    file never outlives this call."""
    tmp = _tmp_report_path()
    try:
        with tmp.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        if not _is_pdf(tmp):
            raise UploadRejected("Not a PDF file")
        digest = _sha256(tmp)
        existing = _report_by_hash(db, digest)
        if existing:
            raise UploadRejected(f"Already ingested as {_report_label(existing)}")
        stored = _stored_report_path(digest)
        tmp.replace(stored)
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
    db.add(report)
    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        # Two uploads of the same bytes at once: the other one won the unique hash.
        if existing := _report_by_hash(db, digest):
            raise UploadRejected(f"Already ingested as {_report_label(existing)}") from exc
        stored.unlink(missing_ok=True)  # no row refers to the file any more
        log.warning("Could not save report row for %s: %s", name, exc)
        raise UploadRejected("Could not save the report, see the backend log") from exc
    db.refresh(report)
    background.add_task(ingest_report, report.id, stored)
    return report


@router.get("/{report_id}", response_model=ReportDetail)
def get_report(report_id: str, db: Session = Depends(get_db)) -> ReportDetail:
    report = _get_report_or_404(db, report_id)
    chunk_count = db.query(func.count(Chunk.id)).filter(Chunk.report_id == report_id).scalar()
    dp_count = db.query(func.count(Datapoint.id)).filter(Datapoint.report_id == report_id).scalar()
    return ReportDetail(
        **ReportOut.model_validate(report).model_dump(),
        chunk_count=chunk_count or 0,
        datapoint_count=dp_count or 0,
    )


@router.post("/{report_id}/extract", response_model=ReportOut, status_code=status.HTTP_202_ACCEPTED)
def extract_again(
    report_id: str,
    body: ExtractRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Report:
    """Run the extraction again for some registry entries of an ingested report: by default
    every entry whose last attempt failed, or the `keys` given. The PDF is not parsed or
    embedded again; only the entries' datapoints are replaced. Returns the report with the
    entries `pending`; poll it until it is `ready` again."""
    report = _get_report_or_404(db, report_id)
    if report.status == ReportStatus.EXTRACTING:
        raise HTTPException(status.HTTP_409_CONFLICT, "Extraction is already running")
    if report.status != ReportStatus.READY:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Only an ingested report can be extracted again"
        )
    if body.keys is None:
        keys = [e.key for e in report.extractions if e.status == ExtractionStatus.ERROR]
        if not keys:
            raise HTTPException(status.HTTP_409_CONFLICT, "No extraction failed on this report")
    else:
        try:
            keys = [spec.key for spec in registry_entries(body.keys)]
        except KeyError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.args[0]) from exc
    begin_reextraction(db, report, keys)
    background.add_task(rerun_extraction, report.id, keys)
    return report


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(report_id: str, db: Session = Depends(get_db)) -> None:
    """Delete the report, its pages, chunks and datapoints, and the PDF on disk. An ingestion
    still running for it is cancelled: it stops before its next inference batch or write."""
    report = _get_report_or_404(db, report_id)
    stored = _stored_report_path(report.file_hash)
    cancellation.request_cancel(report_id)  # before the row goes, so no write slips in between
    db.delete(report)
    db.commit()
    stored.unlink(missing_ok=True)


@router.get("/{report_id}/pdf")
def get_pdf(report_id: str, db: Session = Depends(get_db)) -> FileResponse:
    report = _get_report_or_404(db, report_id)
    path = _stored_report_path(report.file_hash)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "PDF file is missing on disk")
    return FileResponse(path, media_type="application/pdf", filename=report.filename)


@router.get("/{report_id}/pages/{page_no}", response_class=Response)
def get_page_image(report_id: str, page_no: int, db: Session = Depends(get_db)) -> Response:
    """One page rendered as PNG, so the UI can show a cited page without downloading the PDF."""
    report = _get_report_or_404(db, report_id)
    path = _stored_report_path(report.file_hash)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "PDF file is missing on disk")
    with pymupdf.open(path) as pdf:
        if not 1 <= page_no <= pdf.page_count:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found")
        matrix = pymupdf.Matrix(PAGE_RENDER_SCALE, PAGE_RENDER_SCALE)
        png = pdf[page_no - 1].get_pixmap(matrix=matrix).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": PAGE_IMAGE_CACHE})
