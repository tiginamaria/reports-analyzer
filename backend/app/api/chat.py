from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.chat.pipeline import ChatEvent, run_chat
from app.db import SessionLocal, get_db
from app.llm import get_provider
from app.models import Conversation, Message
from app.schemas import ChatRequest, ConversationDetail, ConversationOut, MessageOut

router = APIRouter(prefix="/api", tags=["chat"])

CONVERSATION_TITLE_LENGTH = 80
CONVERSATION_LIST_LIMIT = 100


def _get_conversation_or_404(db: Session, conversation_id: str) -> Conversation:
    convo = db.get(Conversation, conversation_id)
    if convo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return convo


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)) -> list[Conversation]:
    return (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .limit(CONVERSATION_LIST_LIMIT)
        .all()
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> Conversation:
    return _get_conversation_or_404(db, conversation_id)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> None:
    db.delete(_get_conversation_or_404(db, conversation_id))
    db.commit()


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """Server-sent events: a `conversation` event, `stage` events while the pipeline runs,
    then one `answer` (or `error`).

    The pipeline is blocking (database, local models, LLM calls), so it runs in a worker
    thread and hands events to the event loop through a queue; `None` marks the end."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Question is empty")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[ChatEvent | None] = asyncio.Queue()

    def emit(event: ChatEvent | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    worker = asyncio.create_task(asyncio.to_thread(_answer_question, req, question, emit))

    async def stream() -> AsyncIterator[dict]:
        try:
            while (event := await queue.get()) is not None:
                yield {"event": event.type, "data": json.dumps(event.data)}
        finally:
            await worker

    return EventSourceResponse(stream())


def _answer_question(
    req: ChatRequest, question: str, emit: Callable[[ChatEvent | None], None]
) -> None:
    """Worker-thread body: persist the question, run the pipeline, persist the answer."""
    with SessionLocal() as db:
        try:
            convo = _load_or_create_conversation(db, req.conversation_id, question)
            history = list(convo.messages)  # before this question is added
            user_message = Message(
                role="user", content=question, meta={"report_ids": req.report_ids or []}
            )
            _append_message(db, convo, user_message)
            emit(ChatEvent("conversation", {"id": convo.id, "title": convo.title}))

            for event in run_chat(db, get_provider(), question, history, req.report_ids):
                if event.type == "answer":
                    event = _store_answer(db, convo, event)
                emit(event)
        except Exception as exc:  # noqa: BLE001 - reported to the client as an event
            emit(ChatEvent("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            emit(None)


def _load_or_create_conversation(
    db: Session, conversation_id: str | None, question: str
) -> Conversation:
    """An unknown or missing id starts a new conversation titled after the question."""
    convo = db.get(Conversation, conversation_id) if conversation_id else None
    if convo is None:
        convo = Conversation(title=question[:CONVERSATION_TITLE_LENGTH])
        db.add(convo)
        db.commit()
    return convo


def _append_message(db: Session, convo: Conversation, message: Message) -> None:
    """Attach a message and bump the conversation so it sorts to the top of the list."""
    message.conversation_id = convo.id
    convo.updated_at = func.now()
    db.add(message)
    db.commit()


def _store_answer(db: Session, convo: Conversation, event: ChatEvent) -> ChatEvent:
    """Persist the assistant's answer and return the event as the stored message, so the
    client receives the same shape it gets when reloading the conversation."""
    message = Message(
        role="assistant",
        content=event.data["answer"],
        status=event.data["status"],
        citations=event.data["citations"],
        meta=event.data["meta"],
    )
    _append_message(db, convo, message)
    db.refresh(message)
    return ChatEvent("answer", MessageOut.model_validate(message).model_dump(mode="json"))
