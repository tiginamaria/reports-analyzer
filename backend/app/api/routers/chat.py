"""Chat endpoint (server-sent events) and saved conversations."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, status
from sse_starlette.sse import EventSourceResponse

from app.api.deps import Conversations
from app.models import Conversation
from app.schemas import ChatRequest, ConversationDetail, ConversationOut
from app.services.errors import InvalidRequestError

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(service: Conversations) -> list[Conversation]:
    return await service.list_all()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str, service: Conversations) -> Conversation:
    return await service.get(conversation_id, with_messages=True)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str, service: Conversations) -> None:
    await service.delete(conversation_id)


@router.post("/chat")
async def chat(req: ChatRequest, service: Conversations) -> EventSourceResponse:
    """Server-sent events: a `conversation` event, `stage` events while the pipeline runs,
    then one `answer` (or `error`). The pipeline is async, so its events stream straight
    from the generator; an empty question is a 400 before the stream starts."""
    if not req.question.strip():
        raise InvalidRequestError("Question is empty")

    async def stream() -> AsyncIterator[dict]:
        async for event in service.answer(req):
            yield {"event": event.type, "data": json.dumps(event.data)}

    return EventSourceResponse(stream())
