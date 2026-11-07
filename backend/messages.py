"""Endpoint returning the logged-in user's saved chat history."""

from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Depends
from message_store import MessageStore
from pydantic import BaseModel
from user_store import User


class StoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[str]
    message_id: str | None
    rating: Literal["up", "down"] | None


def create_messages_router(store: MessageStore, get_current_user: Callable[..., User]) -> APIRouter:
    router = APIRouter(prefix="/api/messages", tags=["messages"])

    @router.get("", response_model=list[StoredMessage])
    def list_messages(session_id: str | None = None, user: User = Depends(get_current_user)):
        return store.list_messages(user.id, session_id=session_id)

    return router
