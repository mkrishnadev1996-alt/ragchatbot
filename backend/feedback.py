"""Thumbs up/down feedback endpoint for assistant answers."""

import logging
from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Depends
from feedback_store import FeedbackStore
from pydantic import BaseModel, Field
from user_store import User

logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=64)
    # null clears a previously given rating (the user toggled their vote off)
    rating: Literal["up", "down"] | None
    session_id: str | None = Field(default=None, max_length=128)


def create_feedback_router(
    store: FeedbackStore, get_current_user: Callable[..., User]
) -> APIRouter:
    router = APIRouter(prefix="/api/feedback", tags=["feedback"])

    @router.post("", status_code=204)
    def submit_feedback(body: FeedbackRequest, user: User = Depends(get_current_user)):
        if body.rating is None:
            store.clear_rating(body.message_id, user.id)
        else:
            store.set_rating(body.message_id, user.id, body.rating, body.session_id)
        logger.info("Feedback %s (user=%d, message=%s)", body.rating, user.id, body.message_id)

    return router
