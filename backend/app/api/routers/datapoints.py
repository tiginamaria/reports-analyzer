"""Pre-extracted datapoints and the registry they come from."""

from fastapi import APIRouter

from app.api.deps import Datapoints
from app.schemas import DatapointWithReport
from app.services.datapoints import DatapointService

router = APIRouter(prefix="/api/datapoints", tags=["datapoints"])


@router.get("", response_model=list[DatapointWithReport])
async def list_datapoints(
    service: Datapoints, report_id: str | None = None, key: str | None = None
) -> list[DatapointWithReport]:
    return await service.list_all(report_id, key)


@router.get("/registry")
async def registry() -> list[dict]:
    return DatapointService.registry()
