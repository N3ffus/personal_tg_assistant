from datetime import datetime
from typing import Protocol

from src.domain.assistant.deletions import DeletionTarget
from src.domain.calendar.models import CalendarEvent


class CalendarError(Exception):
    """Expected Google Calendar integration failure."""


class CalendarNotConnectedError(CalendarError):
    pass


class CalendarAuthorizationError(CalendarError):
    pass


class CalendarEventNotFoundError(CalendarError):
    pass


class CalendarClient(Protocol):
    async def find_events(
        self, *, user_id: int, title: str | None
    ) -> list[DeletionTarget]: ...

    async def create_event(
        self, *, user_id: int, title: str, starts_at: datetime
    ) -> CalendarEvent: ...

    async def list_events(
        self, *, user_id: int, now: datetime
    ) -> list[CalendarEvent]: ...

    async def update_event(
        self, *, user_id: int, event_id: str, title: str, starts_at: datetime
    ) -> CalendarEvent: ...

    async def delete_event(self, *, user_id: int, event_id: str) -> None: ...

    async def disconnect(self, *, user_id: int) -> None: ...


class PendingOperationStore(Protocol):
    async def create_operation(
        self, *, user_id: int, kind: str, payload: dict[str, object]
    ) -> str: ...

    async def consume_operation(
        self, *, operation_id: str, user_id: int
    ) -> tuple[str, dict[str, object]] | None: ...

    async def consume_latest_operation(
        self, *, user_id: int, kind: str
    ) -> dict[str, object] | None: ...
