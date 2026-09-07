from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.domain.assistant.enums import ActionType
from src.domain.assistant.retrieval import EventQuery, TaskQuery


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


class ListTasksAction(TaskQuery):
    type: Literal[ActionType.LIST_TASKS]


class DeleteTaskAction(DomainModel):
    type: Literal[ActionType.DELETE_TASK]
    title: str = Field(min_length=1)


class DeleteAllTasksAction(DomainModel):
    type: Literal[ActionType.DELETE_ALL_TASKS]


class CreateEventAction(DomainModel):
    type: Literal[ActionType.CREATE_EVENT]
    title: str
    starts_at: datetime


class ListEventsAction(EventQuery):
    type: Literal[ActionType.LIST_EVENTS]


class UpdateEventAction(DomainModel):
    type: Literal[ActionType.UPDATE_EVENT]
    event_title: str = Field(min_length=1)
    title: str
    starts_at: datetime


class DeleteEventAction(DomainModel):
    type: Literal[ActionType.DELETE_EVENT]
    title: str = Field(min_length=1)


class DeleteAllEventsAction(DomainModel):
    type: Literal[ActionType.DELETE_ALL_EVENTS]


class SaveNoteAction(DomainModel):
    type: Literal[ActionType.SAVE_NOTE]
    text: str


AssistantAction = Annotated[
    ChatAction
    | CreateTaskAction
    | ListTasksAction
    | DeleteTaskAction
    | DeleteAllTasksAction
    | CreateEventAction
    | ListEventsAction
    | UpdateEventAction
    | DeleteEventAction
    | DeleteAllEventsAction
    | SaveNoteAction,
    Field(discriminator="type"),
]


class AssistantDecision(DomainModel):
    actions: list[AssistantAction] = Field(min_length=1, max_length=10)
