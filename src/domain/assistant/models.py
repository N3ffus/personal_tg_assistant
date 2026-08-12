from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.domain.assistant.enums import ActionType


class DomainModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


class ChatAction(DomainModel):
    type: Literal[ActionType.CHAT]
    text: str


class CreateTaskAction(DomainModel):
    type: Literal[ActionType.CREATE_TASK]
    title: str


class CreateEventAction(DomainModel):
    type: Literal[ActionType.CREATE_EVENT]
    title: str
    starts_at: datetime


class SaveNoteAction(DomainModel):
    type: Literal[ActionType.SAVE_NOTE]
    text: str


AssistantAction = ChatAction | CreateTaskAction | CreateEventAction | SaveNoteAction


class AssistantDecision(DomainModel):
    action: AssistantAction
