"""Health and the provider/model status."""

from fastapi import APIRouter

from app.schemas import StatusOut
from app.services.maintenance import system_status

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status", response_model=StatusOut)
async def status() -> StatusOut:
    return await system_status()
