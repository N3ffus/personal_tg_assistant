from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.application.services.action_executor import ActionExecutor
from src.domain.assistant.enums import ActionType
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
from src.domain.tasks.models import CreatedTask

NOW = datetime(2026, 9, 3, 9, 0, tzinfo=UTC)
EVENT_START = datetime(2026, 9, 4, 15, 30, tzinfo=UTC)


def event(*, html_link: str | None = "https://calendar.test/event-1") -> CalendarEvent:
    return CalendarEvent(
        event_id="event-1",
        title="Стоматолог",
        starts_at=EVENT_START,
        ends_at=EVENT_START + timedelta(hours=1),
        html_link=html_link,
    )


def dependencies(
    *,
    calendar_event: CalendarEvent | None = None,
    listed_events: list[CalendarEvent] | None = None,
    operation: dict[str, object] | None = None,
) -> tuple[ActionExecutor, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    returned_event = calendar_event or event()
    calendar = SimpleNamespace(
        create_event=AsyncMock(return_value=returned_event),
        list_events=AsyncMock(return_value=listed_events or []),
        update_event=AsyncMock(return_value=returned_event),
        delete_event=AsyncMock(),
        disconnect=AsyncMock(),
    )
    pending = SimpleNamespace(
        consume_latest_operation=AsyncMock(return_value=operation),
    )
    task_tracker = SimpleNamespace(
        create_task=AsyncMock(
            return_value=CreatedTask(
                identifier="ENG-42",
                title="Купить продукты",
                url="https://linear.app/acme/issue/ENG-42/buy-groceries",
            )
        )
    )
    executor = ActionExecutor(
        calendar=calendar,
        pending_operations=pending,
        task_tracker=task_tracker,
    )
    return executor, calendar, pending, task_tracker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (ChatAction(type=ActionType.CHAT, text="Привет"), "Привет"),
        (
            DeleteEventAction(type=ActionType.DELETE_EVENT, title="Стоматолог"),
            "Выберите «Удалить» у события «Стоматолог» через /calendar.",
        ),
        (
            SaveNoteAction(type=ActionType.SAVE_NOTE, text="Люблю Python"),
            "📝 Понял, нужно сохранить заметку:\nЛюблю Python",
        ),
    ],
)
async def test_non_calendar_actions_return_expected_response(
    action: AssistantAction,
    expected: str,
) -> None:
    executor, calendar, pending, task_tracker = dependencies()

    result = await executor.execute(
        action,
        user_id=42,
        now=NOW,
    )

    assert result == expected
    calendar.create_event.assert_not_awaited()
    calendar.list_events.assert_not_awaited()
    calendar.update_event.assert_not_awaited()
    calendar.delete_event.assert_not_awaited()
    pending.consume_latest_operation.assert_not_awaited()
    task_tracker.create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_task_calls_linear_and_returns_link() -> None:
    executor, calendar, pending, task_tracker = dependencies()

    result = await executor.execute(
        CreateTaskAction(type=ActionType.CREATE_TASK, title="Купить продукты"),
        user_id=42,
        now=NOW,
    )

    task_tracker.create_task.assert_awaited_once_with(title="Купить продукты")
    calendar.create_event.assert_not_awaited()
    pending.consume_latest_operation.assert_not_awaited()
    assert result == (
        "✅ Создана задача ENG-42: Купить продукты\n"
        "https://linear.app/acme/issue/ENG-42/buy-groceries"
    )


@pytest.mark.asyncio
async def test_create_event_calls_calendar_and_returns_link() -> None:
    executor, calendar, _, _ = dependencies()
    action = CreateEventAction(
        type=ActionType.CREATE_EVENT,
        title="Стоматолог",
        starts_at=EVENT_START,
    )

    result = await executor.execute(action, user_id=42, now=NOW)

    calendar.create_event.assert_awaited_once_with(
        user_id=42,
        title="Стоматолог",
        starts_at=EVENT_START,
    )
    assert result == (
        "✅ Создано событие: Стоматолог\n"
        "Время: 04.09.2026 15:30\n"
        "https://calendar.test/event-1"
    )


@pytest.mark.asyncio
async def test_event_result_omits_missing_link() -> None:
    executor, _, _, _ = dependencies(calendar_event=event(html_link=None))
    action = CreateEventAction(
        type=ActionType.CREATE_EVENT,
        title="Стоматолог",
        starts_at=EVENT_START,
    )

    result = await executor.execute(action, user_id=42, now=NOW)

    assert result == "✅ Создано событие: Стоматолог\nВремя: 04.09.2026 15:30"


@pytest.mark.asyncio
async def test_list_events_passes_current_time_and_formats_results() -> None:
    executor, calendar, _, _ = dependencies(listed_events=[event()])
    action = ListEventsAction(type=ActionType.LIST_EVENTS)

    result = await executor.execute(action, user_id=42, now=NOW)

    calendar.list_events.assert_awaited_once_with(user_id=42, now=NOW)
    assert result == "📅 Ближайшие события:\n• 04.09 15:30 — Стоматолог"


@pytest.mark.asyncio
async def test_list_events_formats_empty_result() -> None:
    executor, _, _, _ = dependencies(listed_events=[])

    result = await executor.execute(
        ListEventsAction(type=ActionType.LIST_EVENTS),
        user_id=42,
        now=NOW,
    )

    assert result == "Ближайших событий не найдено."


@pytest.mark.asyncio
async def test_update_event_requires_prior_server_side_selection() -> None:
    executor, calendar, pending, _ = dependencies(operation=None)
    action = UpdateEventAction(
        type=ActionType.UPDATE_EVENT,
        title="Новый стоматолог",
        starts_at=EVENT_START,
    )

    result = await executor.execute(action, user_id=42, now=NOW)

    pending.consume_latest_operation.assert_awaited_once_with(
        user_id=42,
        kind="update",
    )
    calendar.update_event.assert_not_awaited()
    assert result == "Сначала выберите событие для изменения через /calendar."


@pytest.mark.asyncio
async def test_update_event_uses_selected_event_id_once() -> None:
    updated = CalendarEvent(
        event_id="selected-event",
        title="Новый стоматолог",
        starts_at=EVENT_START,
        ends_at=EVENT_START + timedelta(hours=1),
        html_link="https://calendar.test/selected-event",
    )
    executor, calendar, pending, _ = dependencies(
        calendar_event=updated,
        operation={"event_id": "selected-event"},
    )
    action = UpdateEventAction(
        type=ActionType.UPDATE_EVENT,
        title="Новый стоматолог",
        starts_at=EVENT_START,
    )

    result = await executor.execute(action, user_id=42, now=NOW)

    pending.consume_latest_operation.assert_awaited_once_with(
        user_id=42,
        kind="update",
    )
    calendar.update_event.assert_awaited_once_with(
        user_id=42,
        event_id="selected-event",
        title="Новый стоматолог",
        starts_at=EVENT_START,
    )
    assert result.startswith("✅ Событие изменено: Новый стоматолог")


@pytest.mark.asyncio
async def test_unknown_action_is_rejected() -> None:
    executor, _, _, _ = dependencies()

    with pytest.raises(ValueError, match="Unsupported action"):
        await executor.execute(object(), user_id=42, now=NOW)  # type: ignore[arg-type]
