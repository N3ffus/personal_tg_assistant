from datetime import datetime
from typing import Protocol

from src.domain.reminders.models import CalendarWatch, Schedule, StoredReminder


class ReminderStore(Protocol):
    async def add(
        self, *, owner_id: int, text: str, schedule: Schedule, next_run_at: datetime
    ) -> StoredReminder: ...

    async def list_reminders(self, *, owner_id: int) -> list[StoredReminder]: ...

    async def delete(self, *, owner_id: int, reminder_id: int) -> bool: ...

    async def due(self, *, now: datetime) -> list[StoredReminder]: ...

    async def advance(
        self, *, reminder_id: int, expected: datetime, next_run_at: datetime | None
    ) -> bool:
        """Move a delivered reminder on, or drop it when ``next_run_at`` is None.

        ``expected`` is the firing that was delivered: a reminder that changed
        in the meantime is left alone.
        """
        ...

    async def get_watch(self, *, owner_id: int) -> CalendarWatch | None: ...

    async def save_watch(self, watch: CalendarWatch) -> None: ...

    async def due_watches(self, *, now: datetime) -> list[CalendarWatch]: ...

    async def was_announced(
        self, *, owner_id: int, event_id: str, starts_at: datetime
    ) -> bool: ...

    async def mark_announced(
        self, *, owner_id: int, event_id: str, starts_at: datetime, now: datetime
    ) -> None: ...

    async def forget_announced_before(self, *, moment: datetime) -> None: ...


class Notifier(Protocol):
    async def send(self, *, user_id: int, text: str) -> None: ...
