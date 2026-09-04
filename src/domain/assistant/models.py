from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class ListEventsAction(DomainModel):
    type: Literal[ActionType.LIST_EVENTS]


class UpdateEventAction(DomainModel):
    type: Literal[ActionType.UPDATE_EVENT]
    title: str
    starts_at: datetime


class DeleteEventAction(DomainModel):
    type: Literal[ActionType.DELETE_EVENT]
    title: str


class SaveNoteAction(DomainModel):
    type: Literal[ActionType.SAVE_NOTE]
    text: str


AssistantAction = Annotated[
    ChatAction
    | CreateTaskAction
    | CreateEventAction
    | ListEventsAction
    | UpdateEventAction
    | DeleteEventAction
    | SaveNoteAction,
    Field(discriminator="type"),
]


class AssistantDecision(DomainModel):
    action: AssistantAction
