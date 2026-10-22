"""Report upload, status, extraction retry and deletion; the PDF and its pages as images."""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import Reports
from app.ingestion.pipeline import ingest_report, rerun_extraction
from app.models import Report
from app.schemas import ExtractRequest, RejectedFile, ReportDetail, ReportOut, UploadResult
from app.services.errors import UploadRejected

router = APIRouter(prefix="/api/reports", tags=["reports"])

PAGE_IMAGE_CACHE = "public, max-age=86400"  # a report's pages never change


@router.get("", response_model=list[ReportOut])
async def list_reports(service: Reports) -> list[Report]:
    return await service.list_all()


@router.post("", response_model=UploadResult)
async def upload_reports(
    service: Reports,
    background: BackgroundTasks,
    response: Response,
    files: list[UploadFile] = File(...),
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
            accepted = await service.accept_upload(file.file, name)
        except UploadRejected as exc:
            rejected.append(RejectedFile(filename=name, reason=str(exc)))
        else:
            started.append(ReportOut.model_validate(accepted.report))
            background.add_task(ingest_report, accepted.report.id, accepted.path)
    response.status_code = status.HTTP_201_CREATED if started else status.HTTP_200_OK
    return UploadResult(started=started, rejected=rejected)


@router.get("/{report_id}", response_model=ReportDetail)
async def get_report(report_id: str, service: Reports) -> ReportDetail:
    return await service.detail(report_id)


@router.post("/{report_id}/extract", response_model=ReportOut, status_code=status.HTTP_202_ACCEPTED)
async def extract_again(
    report_id: str, body: ExtractRequest, service: Reports, background: BackgroundTasks
) -> Report:
    """Run the extraction again for some registry entries of an ingested report: by default
    every entry whose last attempt failed, or the `keys` given. The PDF is not parsed or
    embedded again; only the entries' datapoints are replaced. Returns the report with the
    entries `pending`; poll it until it is `ready` again."""
    plan = await service.plan_reextraction(report_id, body.keys)
    background.add_task(rerun_extraction, plan.report.id, plan.keys)
    return plan.report


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(report_id: str, service: Reports) -> None:
    """Delete the report, its pages, chunks and datapoints, and the PDF on disk. An ingestion
    still running for it is cancelled."""
    await service.delete(report_id)


@router.get("/{report_id}/pdf")
async def get_pdf(report_id: str, service: Reports) -> FileResponse:
    report = await service.get(report_id)
    path = await service.pdf_path(report_id)
    return FileResponse(path, media_type="application/pdf", filename=report.filename)


@router.get("/{report_id}/pages/{page_no}", response_class=Response)
async def get_page_image(report_id: str, page_no: int, service: Reports) -> Response:
    """One page rendered as PNG, so the UI can show a cited page without downloading the PDF."""
    png = await service.render_page(report_id, page_no)
    return Response(png, media_type="image/png", headers={"Cache-Control": PAGE_IMAGE_CACHE})
