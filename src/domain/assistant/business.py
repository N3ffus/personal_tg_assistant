from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from src.domain.assistant.models import (
    CreateEventAction,
    CreateTaskAction,
    DomainModel,
    SaveNoteAction,
)


class BusinessTask(CreateTaskAction):
    title: str = Field(min_length=1, max_length=250)


class BusinessEvent(CreateEventAction):
    title: str = Field(min_length=1, max_length=250)
    starts_at: AwareDatetime


class BusinessNote(SaveNoteAction):
    text: str = Field(min_length=1, max_length=1500)


BusinessAction = Annotated[
    BusinessTask | BusinessEvent | BusinessNote, Field(discriminator="type")
]


class BusinessIntent(DomainModel):
    task_assignee: Literal["owner", "peer", "third_party", "unclear"] = Field(
        default="unclear",
        description="Who will perform the task, resolved from the whole dialog, not who sent the message.",
    )
    task_basis: Literal[
        "peer_request", "owner_commitment", "owner_acceptance", "none"
    ] = Field(
        default="none",
        description="Owner task basis: direct peer assignment, own promise, or acceptance of a specific peer request. Otherwise none.",
    )
    action: BusinessAction
    source_message_ids: list[int] = Field(min_length=1, max_length=20)
    owner_confirmation_message_id: int | None
    owner_confirmation_quote: str = Field(max_length=1000)
    peer_request_message_id: int | None = None
    peer_request_quote: str = Field(default="", max_length=1000)


class BusinessDecision(DomainModel):
    actions: list[BusinessIntent] = Field(max_length=10)


class StoredBusinessAction(DomainModel):
    id: str
    action: BusinessAction
    status: Literal["pending", "running", "completed"]
    result: str | None
