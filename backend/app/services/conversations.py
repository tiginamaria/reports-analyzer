"""Saved conversations, and answering a question in one of them."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.pipeline import ChatEvent, run_chat
from app.llm import get_provider
from app.models import Conversation, Message
from app.repositories.conversations import ConversationRepository
from app.schemas import ChatRequest, MessageOut
from app.services.errors import InvalidRequestError, NotFoundError

CONVERSATION_TITLE_LENGTH = 80
CONVERSATION_LIST_LIMIT = 100


class ConversationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.conversations = ConversationRepository(db)

    async def list_all(self) -> list[Conversation]:
        return list(await self.conversations.list(CONVERSATION_LIST_LIMIT))

    async def get(self, conversation_id: str, *, with_messages: bool = False) -> Conversation:
        convo = await self.conversations.get(conversation_id, with_messages=with_messages)
        if convo is None:
            raise NotFoundError("Conversation not found")
        return convo

    async def delete(self, conversation_id: str) -> None:
        await self.conversations.delete(await self.get(conversation_id))
        await self.db.commit()

    async def answer(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
        """Persist the question, run the chat pipeline and persist its answer. Yields a
        `conversation` event, the pipeline's `stage` events, then one `answer` (or `error`).
        The answer is stored before it is yielded, in the shape the UI reloads, so what the
        user saw live is what they see later. Any failure becomes an `error` event."""
        question = req.question.strip()
        if not question:
            raise InvalidRequestError("Question is empty")
        try:
            convo = await self._load_or_create(req.conversation_id, question)
            history = list(convo.messages)  # before this question is added
            user_message = Message(
                role="user", content=question, meta={"report_ids": req.report_ids or []}
            )
            await self._append(convo, user_message)
            yield ChatEvent("conversation", {"id": convo.id, "title": convo.title})

            async for event in run_chat(self.db, get_provider(), question, history, req.report_ids):
                if event.type == "answer":
                    event = await self._store_answer(convo, event)
                yield event
        except Exception as exc:  # reported to the client as an event
            yield ChatEvent("error", {"message": f"{type(exc).__name__}: {exc}"})

    async def _load_or_create(self, conversation_id: str | None, question: str) -> Conversation:
        """An unknown or missing id starts a new conversation titled after the question."""
        convo = None
        if conversation_id:
            convo = await self.conversations.get(conversation_id, with_messages=True)
        if convo is None:
            convo = Conversation(title=question[:CONVERSATION_TITLE_LENGTH])
            self.conversations.add(convo)
            await self.db.commit()
            await self.db.refresh(convo, ["messages"])
        return convo

    async def _append(self, convo: Conversation, message: Message) -> None:
        self.conversations.append_message(convo, message)
        await self.db.commit()

    async def _store_answer(self, convo: Conversation, event: ChatEvent) -> ChatEvent:
        """Persist the assistant's answer and return the event as the stored message."""
        message = Message(
            role="assistant",
            content=event.data["answer"],
            status=event.data["status"],
            citations=event.data["citations"],
            meta=event.data["meta"],
        )
        await self._append(convo, message)
        await self.db.refresh(message)
        return ChatEvent("answer", MessageOut.model_validate(message).model_dump(mode="json"))
