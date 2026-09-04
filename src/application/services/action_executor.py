from datetime import datetime

from src.application.ports.calendar import CalendarClient, PendingOperationStore
from src.application.ports.tasks import TaskTrackerClient
from src.domain.assistant.models import (
    AssistantAction,
    ChatAction,
    CreateEventAction,
    CreateTaskAction,
    DeleteEventAction,
    ListEventsAction,
    SaveNoteAction,
    UpdateEventAction,
)
from src.domain.calendar.models import CalendarEvent


class ActionExecutor:
    def __init__(
        self,
        *,
        calendar: CalendarClient,
        pending_operations: PendingOperationStore,
        task_tracker: TaskTrackerClient,
    ) -> None:
        self._calendar = calendar
        self._pending_operations = pending_operations
        self._task_tracker = task_tracker

    async def execute(
        self,
        action: AssistantAction,
        *,
        user_id: int,
        now: datetime,
    ) -> str:
        if isinstance(action, ChatAction):
            return action.text

        if isinstance(action, CreateTaskAction):
            task = await self._task_tracker.create_task(title=action.title)
            return f"✅ Создана задача {task.identifier}: {task.title}\n{task.url}"

        if isinstance(action, CreateEventAction):
            event = await self._calendar.create_event(
                user_id=user_id,
                title=action.title,
                starts_at=action.starts_at,
            )
            return self._event_result("Создано событие", event)

        if isinstance(action, ListEventsAction):
            events = await self._calendar.list_events(user_id=user_id, now=now)
            return self.format_events(events)

        if isinstance(action, UpdateEventAction):
            payload = await self._pending_operations.consume_latest_operation(
                user_id=user_id,
                kind="update",
            )
            if payload is None:
                return "Сначала выберите событие для изменения через /calendar."
            event = await self._calendar.update_event(
                user_id=user_id,
                event_id=str(payload["event_id"]),
                title=action.title,
                starts_at=action.starts_at,
            )
            return self._event_result("Событие изменено", event)

        if isinstance(action, DeleteEventAction):
            return f"Выберите «Удалить» у события «{action.title}» через /calendar."

        if isinstance(action, SaveNoteAction):
            return f"📝 Понял, нужно сохранить заметку:\n{action.text}"

        raise ValueError(f"Unsupported action: {type(action)!r}")

    @staticmethod
    def format_events(events: list[CalendarEvent]) -> str:
        if not events:
            return "Ближайших событий не найдено."
        lines = ["📅 Ближайшие события:"]
        lines.extend(
            f"• {event.starts_at:%d.%m %H:%M} — {event.title}" for event in events
        )
        return "\n".join(lines)

    @staticmethod
    def _event_result(verb: str, event: CalendarEvent) -> str:
        result = f"✅ {verb}: {event.title}\nВремя: {event.starts_at:%d.%m.%Y %H:%M}"
        return f"{result}\n{event.html_link}" if event.html_link else result
