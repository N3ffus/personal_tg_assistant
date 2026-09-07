from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from src.application.ports.calendar import CalendarError, CalendarNotConnectedError
from src.application.ports.tasks import (
    TaskCreationUncertainError,
    TaskTrackerError,
)
from src.application.services.action_executor import ActionExecutor
from src.domain.assistant.deletions import DeletionTarget
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    AssistantAction,
    ChatAction,
    CreateEventAction,
    CreateTaskAction,
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


def target(number: int) -> DeletionTarget:
    return DeletionTarget(
        id=f"event-{number}",
        title="Стоматолог",
        label=f"0{number}.09.2026 15:30 +0000 — Стоматолог",
    )


def dependencies(
    *,
    calendar_event: CalendarEvent | None = None,
    listed_events: list[CalendarEvent] | None = None,
    found_events: list[DeletionTarget] | None = None,
    operation: dict[str, object] | None = None,
) -> tuple[ActionExecutor, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    returned_event = calendar_event or event()
    calendar = SimpleNamespace(
        create_event=AsyncMock(return_value=returned_event),
        list_events=AsyncMock(return_value=listed_events or []),
        find_events=AsyncMock(return_value=found_events or []),
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
async def test_execute_many_creates_all_tasks_in_source_order() -> None:
    executor, calendar, pending, task_tracker = dependencies()
    task_tracker.create_task.side_effect = [
        CreatedTask(identifier="ENG-1", title="Посмотреть фильм", url="url-1"),
        CreatedTask(identifier="ENG-2", title="Поботать LLM-ки", url="url-2"),
        CreatedTask(identifier="ENG-3", title="Отдохнуть", url="url-3"),
    ]
    actions = [
        CreateTaskAction(type=ActionType.CREATE_TASK, title="Посмотреть фильм"),
        CreateTaskAction(type=ActionType.CREATE_TASK, title="Поботать LLM-ки"),
        CreateTaskAction(type=ActionType.CREATE_TASK, title="Отдохнуть"),
    ]

    result = await executor.execute_many(actions, user_id=42, now=NOW)

    assert task_tracker.create_task.await_args_list == [
        call(title="Посмотреть фильм"),
        call(title="Поботать LLM-ки"),
        call(title="Отдохнуть"),
    ]
    calendar.create_event.assert_not_awaited()
    pending.consume_latest_operation.assert_not_awaited()
    assert result == (
        "✅ Создана задача ENG-1: Посмотреть фильм\nurl-1\n\n"
        "✅ Создана задача ENG-2: Поботать LLM-ки\nurl-2\n\n"
        "✅ Создана задача ENG-3: Отдохнуть\nurl-3"
    )


@pytest.mark.asyncio
async def test_execute_many_supports_mixed_task_and_calendar_event() -> None:
    executor, calendar, _, task_tracker = dependencies()
    task_tracker.create_task.return_value = CreatedTask(
        identifier="ENG-1",
        title="Купить молоко",
        url="task-url",
    )
    actions: list[AssistantAction] = [
        CreateTaskAction(type=ActionType.CREATE_TASK, title="Купить молоко"),
        CreateEventAction(
            type=ActionType.CREATE_EVENT,
            title="Стоматолог",
            starts_at=EVENT_START,
        ),
    ]

    result = await executor.execute_many(actions, user_id=42, now=NOW)

    task_tracker.create_task.assert_awaited_once_with(title="Купить молоко")
    calendar.create_event.assert_awaited_once_with(
        user_id=42,
        title="Стоматолог",
        starts_at=EVENT_START,
    )
    assert result == (
        "✅ Создана задача ENG-1: Купить молоко\ntask-url\n\n"
        "✅ Создано событие: Стоматолог\n"
        "Время: 04.09.2026 15:30\n"
        "https://calendar.test/event-1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            TaskCreationUncertainError(),
            "⚠️ Linear мог создать задачу «Первая». "
            "Проверьте список задач перед повтором.",
        ),
        (
            TaskTrackerError(),
            "❌ Не удалось создать задачу «Первая» в Linear.",
        ),
    ],
)
async def test_execute_many_reports_task_failure_and_continues(
    error: TaskTrackerError,
    expected: str,
) -> None:
    executor, _, _, task_tracker = dependencies()
    task_tracker.create_task.side_effect = [
        error,
        CreatedTask(identifier="ENG-2", title="Вторая", url="url-2"),
    ]
    actions = [
        CreateTaskAction(type=ActionType.CREATE_TASK, title="Первая"),
        CreateTaskAction(type=ActionType.CREATE_TASK, title="Вторая"),
    ]

    result = await executor.execute_many(actions, user_id=42, now=NOW)

    assert result == f"{expected}\n\n✅ Создана задача ENG-2: Вторая\nurl-2"
    assert task_tracker.create_task.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            CalendarNotConnectedError(),
            "❌ Не удалось создать событие «Стоматолог»: подключите календарь "
            "командой /calendar_connect.",
        ),
        (
            CalendarError(),
            "❌ Не удалось создать событие «Стоматолог»: календарь недоступен.",
        ),
    ],
)
async def test_execute_many_reports_calendar_failure(
    error: CalendarError,
    expected: str,
) -> None:
    executor, calendar, _, _ = dependencies()
    calendar.create_event.side_effect = error
    action = CreateEventAction(
        type=ActionType.CREATE_EVENT,
        title="Стоматолог",
        starts_at=EVENT_START,
    )

    result = await executor.execute_many([action], user_id=42, now=NOW)

    assert result == expected


@pytest.mark.asyncio
async def test_execute_many_rejects_empty_batch() -> None:
    executor, _, _, _ = dependencies()

    with pytest.raises(ValueError, match="At least one action"):
        await executor.execute_many([], user_id=42, now=NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (ChatAction(type=ActionType.CHAT, text="Привет"), "Привет"),
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

    calendar.list_events.assert_awaited_once()
    assert calendar.list_events.call_args.kwargs["now"] == NOW
    assert calendar.list_events.call_args.kwargs["query"].date_from == NOW
    assert not isinstance(result, str)
    assert "04.09.2026 15:30 UTC" in result.pages[0].text
    assert "Стоматолог" in result.pages[0].text
    assert "https://calendar.test/event-1" in result.pages[0].text


@pytest.mark.asyncio
async def test_list_events_formats_empty_result() -> None:
    executor, _, _, _ = dependencies(listed_events=[])

    result = await executor.execute(
        ListEventsAction(type=ActionType.LIST_EVENTS),
        user_id=42,
        now=NOW,
    )

    assert not isinstance(result, str)
    assert "ничего не найдено" in result.pages[0].text


def update_action() -> UpdateEventAction:
    return UpdateEventAction(
        type=ActionType.UPDATE_EVENT,
        event_title="стоматолог",
        title="Новый стоматолог",
        starts_at=EVENT_START,
    )


@pytest.mark.asyncio
async def test_update_event_without_matches_asks_for_the_current_title() -> None:
    executor, calendar, _, _ = dependencies(found_events=[])

    result = await executor.execute(update_action(), user_id=42, now=NOW)

    calendar.find_events.assert_awaited_once_with(user_id=42, title="стоматолог")
    calendar.update_event.assert_not_awaited()
    assert result == ("Событие «стоматолог» не найдено. Уточните его текущее название.")


@pytest.mark.asyncio
async def test_update_event_never_guesses_between_several_matches() -> None:
    executor, calendar, _, _ = dependencies(found_events=[target(1), target(2)])

    result = await executor.execute(update_action(), user_id=42, now=NOW)

    calendar.update_event.assert_not_awaited()
    assert isinstance(result, str)
    assert "Уточните, какое изменить" in result
    assert target(1).label in result and target(2).label in result


@pytest.mark.asyncio
async def test_update_event_uses_the_single_match_found_by_title() -> None:
    updated = CalendarEvent(
        event_id="event-1",
        title="Новый стоматолог",
        starts_at=EVENT_START,
        ends_at=EVENT_START + timedelta(hours=1),
        html_link="https://calendar.test/event-1",
    )
    executor, calendar, pending, _ = dependencies(
        calendar_event=updated, found_events=[target(1)]
    )

    result = await executor.execute(update_action(), user_id=42, now=NOW)

    calendar.find_events.assert_awaited_once_with(user_id=42, title="стоматолог")
    pending.consume_latest_operation.assert_not_awaited()
    calendar.update_event.assert_awaited_once_with(
        user_id=42,
        event_id="event-1",
        title="Новый стоматолог",
        starts_at=EVENT_START,
    )
    assert isinstance(result, str)
    assert result.startswith("✅ Событие изменено: Новый стоматолог")


@pytest.mark.asyncio
async def test_unknown_action_is_rejected() -> None:
    executor, _, _, _ = dependencies()

    with pytest.raises(ValueError, match="Unsupported action"):
        await executor.execute(object(), user_id=42, now=NOW)  # type: ignore[arg-type]
