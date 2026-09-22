from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.domain.assistant.enums import ActionType
from src.domain.assistant.retrieval import EventQuery, TaskQuery
from src.domain.knowledge.models import (
    DEFAULT_KNOWLEDGE_RESULTS,
    MAX_KNOWLEDGE_RESULTS,
    KnowledgeText,
)


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


class RememberKnowledgeAction(DomainModel):
    """Store durable knowledge. The namespace is added by the backend."""

    type: Literal[ActionType.REMEMBER_KNOWLEDGE]
    content: KnowledgeText


class SearchKnowledgeAction(DomainModel):
    """Look knowledge up. The backend restricts the search to the current user."""

    type: Literal[ActionType.SEARCH_KNOWLEDGE]
    query: str = Field(min_length=1, max_length=400)
    mode: Literal[
        "semantic", "profile", "recent_watched_films", "watched_film_catalogue"
    ] = "semantic"
    limit: int = Field(
        default=DEFAULT_KNOWLEDGE_RESULTS, ge=1, le=MAX_KNOWLEDGE_RESULTS
    )


MAX_FORGOTTEN_FACTS = 20


class ForgetKnowledgeAction(DomainModel):
    """Erase what the user asked to forget, in two turns.

    The first turn names the fact in ``query``; the application recalls the
    candidates and shows them with their ``ref``. The answer turn repeats the
    action with ``refs`` of exactly the facts the request covers, so a similar
    but different fact (the previous employer) is never erased by resemblance.
    """

    type: Literal[ActionType.FORGET_KNOWLEDGE]
    query: str = Field(min_length=1, max_length=400)
    refs: list[str] = Field(default_factory=list, max_length=MAX_FORGOTTEN_FACTS)


# A batch has to outnumber what the user asked for: most of it is filtered out
# against a catalogue of hundreds of watched films.
MAX_FILM_CANDIDATES = 30
MAX_FILM_ALIASES = 8


class FilmCandidate(DomainModel):
    title: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=MAX_FILM_ALIASES)
    # Shown to the user: a renamed or invented film is obvious next to its year.
    year: int | None = Field(default=None, ge=1888, le=2100)
    # A blank reason must not invalidate the whole decision: the model dropped
    # one on a fifteen-candidate list and the user got an error instead of an
    # answer. The title alone is still a usable recommendation.
    reason: str = Field(default="", max_length=300)


class RecommendFilmsAction(DomainModel):
    type: Literal[ActionType.RECOMMEND_FILMS]
    # An empty list is tolerated on purpose: a model that sends one would
    # otherwise fail validation and cost the user the whole reply, while the
    # executor already answers «не удалось подобрать» for an empty selection.
    candidates: list[FilmCandidate] = Field(
        min_length=0, max_length=MAX_FILM_CANDIDATES
    )
    limit: int = Field(default=3, ge=1, le=10)


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
    | SaveNoteAction
    | RememberKnowledgeAction
    | SearchKnowledgeAction
    | ForgetKnowledgeAction
    | RecommendFilmsAction,
    Field(discriminator="type"),
]


class AssistantDecision(DomainModel):
    actions: list[AssistantAction] = Field(min_length=1, max_length=10)
