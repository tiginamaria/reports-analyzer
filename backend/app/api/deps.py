"""Request-scoped dependencies: the database session and the services built on it."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal
from app.services.conversations import ConversationService
from app.services.datapoints import DatapointService
from app.services.reports import ReportService


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as db:
        yield db


DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_report_service(db: DbSession) -> ReportService:
    return ReportService(db)


def get_datapoint_service(db: DbSession) -> DatapointService:
    return DatapointService(db)


def get_conversation_service(db: DbSession) -> ConversationService:
    return ConversationService(db)


Reports = Annotated[ReportService, Depends(get_report_service)]
Datapoints = Annotated[DatapointService, Depends(get_datapoint_service)]
Conversations = Annotated[ConversationService, Depends(get_conversation_service)]
