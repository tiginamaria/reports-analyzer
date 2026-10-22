"""Queries on conversations and messages."""

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Conversation, Message


class ConversationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list(self, limit: int) -> Sequence[Conversation]:
        stmt = select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
        return (await self.db.execute(stmt)).scalars().all()

    async def get(
        self, conversation_id: str, *, with_messages: bool = False
    ) -> Conversation | None:
        options = [selectinload(Conversation.messages)] if with_messages else []
        return await self.db.get(Conversation, conversation_id, options=options)

    def add(self, convo: Conversation) -> None:
        self.db.add(convo)

    async def delete(self, convo: Conversation) -> None:
        await self.db.delete(convo)

    def append_message(self, convo: Conversation, message: Message) -> None:
        """Attach a message and bump the conversation so it sorts to the top of the list."""
        message.conversation_id = convo.id
        convo.updated_at = func.now()
        self.db.add(message)
