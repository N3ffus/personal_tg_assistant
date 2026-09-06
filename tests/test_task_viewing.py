from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot
from aiogram.types import Chat, Message, User

from src.application.ports.calendar import CalendarError, CalendarNotConnectedError
from src.application.ports.tasks import TaskTrackerError
from src.application.services.action_executor import ActionExecutor
from src.application.services.explicit_commands import parse_view_request
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import AssistantDecision, ListTasksAction
from src.domain.assistant.replies import AssistantReply
from src.domain.assistant.retrieval import EventQuery, TaskQuery
from src.domain.calendar.models import CalendarEvent
from src.domain.tasks.models import Task
from src.infrastructure.telegram.calendar import _event_keyboard
from src.infrastructure.telegram.commands import BOT_COMMANDS, configure_commands
from src.infrastructure.telegram.handlers import create_router
from src.infrastructure.telegram.replies import answer_text

NOW = datetime(2026, 9, 6, 12, tzinfo=ZoneInfo("Europe/Moscow"))
TASK = Task(
    identifier="APP-42",
    title="Отчёт",
    status="In Progress",
    url="https://linear.app/example/issue/APP-42",
)
EVENT = CalendarEvent(
    event_id="event-1",
    title="Встреча",
    starts_at=NOW,
    ends_at=NOW + timedelta(hours=1),
    html_link="https://calendar.test/event-1",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "tasks", "events"),
    [
        ("/tasks", True, False),
        ("/calendar", False, True),
        ("/agenda", True, True),
        ("Покажи задачи в Linear", True, False),
        ("Покажи задачи с Linear", True, False),
        ("выведи список задач в linear", True, False),
        ("покажи все задачи в linear", True, False),
        ("список задач в linear", True, False),
        ("Покажи события календаря", False, True),
        ("Покажи задачи из Linear и события календаря", True, True),
        ("Покажи задачи с Linear и с календаря!", True, True),
    ],
)
async def test_view_requests_fetch_real_lists_without_llm_or_mutations(
    text: str,
    tasks: bool,
    events: bool,
) -> None:
    integrations = SimpleNamespace(
        list_tasks=AsyncMock(return_value=[TASK]),
        list_events=AsyncMock(return_value=[EVENT]),
        create_task=AsyncMock(),
        create_event=AsyncMock(),
        delete_task=AsyncMock(),
        delete_event=AsyncMock(),
        create_operation=AsyncMock(),
    )
    llm = SimpleNamespace(parse_message=AsyncMock(side_effect=RuntimeError("offline")))
    use_case = ProcessMessageUseCase(
        llm=llm,
        action_executor=ActionExecutor(
            calendar=integrations,
            task_tracker=integrations,
            pending_operations=integrations,
        ),
    )
    result = await use_case.execute(
        text=text, user_id=42, now=NOW, timezone="Europe/Moscow"
    )
    assert isinstance(result, AssistantReply)
    assert len(result.pages) == int(tasks) + int(events)
    result_text = "\n".join(page.text for page in result.pages)
    result = result_text
    assert (TASK.identifier in result) is tasks
    assert (EVENT.title in result) is events
    if tasks:
        assert TASK.status in result and TASK.url in result
        integrations.list_tasks.assert_awaited_once_with(query=TaskQuery())
    else:
        integrations.list_tasks.assert_not_awaited()
    if events:
        assert "06.09.2026 12:00 MSK" in result
        assert EVENT.html_link is not None and EVENT.html_link in result
        integrations.list_events.assert_awaited_once_with(
            user_id=42, now=NOW, query=EventQuery().with_default_range(NOW)
        )
    else:
        integrations.list_events.assert_not_awaited()
    llm.parse_message.assert_not_awaited()
    for name in (
        "create_task",
        "create_event",
        "delete_task",
        "delete_event",
        "create_operation",
    ):
        getattr(integrations, name).assert_not_awaited()


@pytest.mark.parametrize(
    "text",
    [
        "Не показывай задачи в Linear",
        "Покажи задачи в Linear за сегодня",
        "Покажи выполненные задачи в Linear",
        "Покажи задачи в Linear и удали APP-1",
        "Он написал: покажи задачи в Linear",
        "Покажи задачи в Linear кроме APP-1",
    ],
)
def test_qualified_or_mixed_requests_are_left_for_llm(text: str) -> None:
    assert parse_view_request(text) is None


@pytest.mark.asyncio
async def test_natural_language_list_tasks_decision_is_executable() -> None:
    llm = SimpleNamespace(
        parse_message=AsyncMock(
            return_value=AssistantDecision.model_validate(
                {"actions": [{"type": "list_tasks"}]},
            )
        )
    )
    integrations = SimpleNamespace(list_tasks=AsyncMock(return_value=[TASK]))
    use_case = ProcessMessageUseCase(
        llm=llm,
        action_executor=ActionExecutor(
            calendar=integrations,
            task_tracker=integrations,
            pending_operations=integrations,
        ),
    )
    for _ in range(2):
        reply = await use_case.execute(
            text="Какие задачи есть у меня в Linear?",
            user_id=42,
            now=NOW,
            timezone="Europe/Moscow",
        )
        assert (
            isinstance(reply, AssistantReply) and TASK.identifier in reply.pages[0].text
        )
    assert llm.parse_message.await_count == 2
    assert integrations.list_tasks.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "error", "expected", "preserved"),
    [
        (
            "list_tasks",
            TaskTrackerError(),
            "Не удалось получить задачи Linear",
            EVENT.title,
        ),
        ("list_events", CalendarError(), "календарь недоступен", TASK.identifier),
        (
            "list_events",
            CalendarNotConnectedError(),
            "/calendar_connect",
            TASK.identifier,
        ),
    ],
)
async def test_agenda_preserves_other_source_on_failure(
    source: str,
    error: Exception,
    expected: str,
    preserved: str,
) -> None:
    integrations = SimpleNamespace(
        list_tasks=AsyncMock(return_value=[TASK]),
        list_events=AsyncMock(return_value=[EVENT]),
    )
    getattr(integrations, source).side_effect = error
    executor = ActionExecutor(
        calendar=integrations,
        task_tracker=integrations,
        pending_operations=integrations,
    )
    decision = parse_view_request("/agenda")
    assert decision is not None
    reply = await executor.execute_many(decision.actions, user_id=42, now=NOW)
    assert isinstance(reply, AssistantReply)
    assert expected in reply.text
    assert preserved in reply.pages[0].text


@pytest.mark.asyncio
async def test_empty_linear_list_has_clear_message() -> None:
    integrations = SimpleNamespace(list_tasks=AsyncMock(return_value=[]))
    executor = ActionExecutor(
        calendar=integrations,
        task_tracker=integrations,
        pending_operations=integrations,
    )
    reply = await executor.execute(
        ListTasksAction(type=ActionType.LIST_TASKS), user_id=42, now=NOW
    )
    assert isinstance(reply, AssistantReply)
    assert "ничего не найдено" in reply.pages[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["tasks", "agenda", "calendar"])
@pytest.mark.parametrize("user_id", [42, 99, None])
async def test_view_command_routing_and_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    user_id: int | None,
) -> None:
    process = SimpleNamespace(execute=AsyncMock(return_value="Список"))
    router = create_router(process_message=process, timezone="UTC", allowed_user_id=42)  # type: ignore[arg-type]
    bot = AsyncMock(spec=Bot)
    bot.me.return_value = User(
        id=123, first_name="Bot", is_bot=True, username="example_bot"
    )
    message = Message(
        message_id=1,
        date=NOW,
        chat=Chat(id=42, type="private"),
        from_user=User(id=user_id, first_name="User", is_bot=False)
        if user_id
        else None,
        text=f"/{command}@example_bot",
    )
    answer = AsyncMock()
    monkeypatch.setattr(Message, "answer", answer)
    await router.propagate_event("message", message, bot=bot)
    if user_id == 42:
        assert process.execute.call_args.kwargs["text"] == f"/{command}"
        assert process.execute.call_args.kwargs["user_id"] == 42
        answer.assert_awaited_once_with("Список")
    else:
        process.execute.assert_not_awaited()
        answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_view_commands_do_not_silently_ignore_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(execute=AsyncMock(return_value="Выполненные задачи"))
    router = create_router(process_message=process, timezone="UTC", allowed_user_id=42)  # type: ignore[arg-type]
    message = Message(
        message_id=1,
        date=NOW,
        chat=Chat(id=42, type="private"),
        from_user=User(id=42, first_name="User", is_bot=False),
        text="/tasks completed",
    )
    answer = AsyncMock()
    monkeypatch.setattr(Message, "answer", answer)
    await router.propagate_event("message", message, bot=AsyncMock(spec=Bot))
    assert process.execute.call_args.kwargs["text"] == "Покажи задачи Linear: completed"
    answer.assert_awaited_once_with("Выполненные задачи")


@pytest.mark.asyncio
async def test_menu_includes_all_commands_and_is_synchronized() -> None:
    bot = AsyncMock(spec=Bot)
    await configure_commands(bot)
    bot.set_my_commands.assert_awaited_once_with(list(BOT_COMMANDS), request_timeout=10)
    assert {command.command for command in BOT_COMMANDS} == {
        "start",
        "history",
        "compact",
        "clear",
        "tasks",
        "agenda",
        "calendar",
        "calendar_connect",
        "calendar_disconnect",
    }
    bot.set_chat_menu_button.assert_awaited_once()


def test_event_display_normalizes_timed_events_but_preserves_all_day_dates() -> None:
    all_day = CalendarEvent(
        event_id="holiday",
        title="Отпуск",
        starts_at=datetime(2026, 9, 6, tzinfo=UTC),
        ends_at=datetime(2026, 9, 9, tzinfo=UTC),
        html_link=None,
        all_day=True,
    )
    text = ActionExecutor.format_events(
        [EVENT, all_day], timezone=ZoneInfo("America/Los_Angeles")
    )
    assert "06.09.2026 02:00 PDT" in text
    assert "06.09.2026–08.09.2026 (весь день)" in text
    single_day = CalendarEvent(
        event_id="holiday-2",
        title="Праздник",
        starts_at=all_day.starts_at,
        ends_at=all_day.starts_at + timedelta(days=1),
        html_link=None,
        all_day=True,
    )
    assert "06.09.2026 (весь день)" in ActionExecutor.format_events([single_day])
    keyboard = _event_keyboard([EVENT, all_day], ["timed-token", "date-token"])
    assert [b.callback_data for b in keyboard.inline_keyboard[0]] == [
        "caledit:timed-token",
        "caldel:timed-token",
    ]
    assert [b.callback_data for b in keyboard.inline_keyboard[1]] == [
        "caldel:date-token"
    ]


@pytest.mark.asyncio
async def test_long_task_list_is_split_without_losing_tasks() -> None:
    tasks = [
        Task(
            identifier=f"APP-{i}",
            title="Задача " + "😀" * 150,
            url=f"https://linear.test/{i}",
            status="Todo",
        )
        for i in range(101)
    ]
    text = ActionExecutor.format_tasks(tasks)
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    await answer_text(message, text)
    chunks = [call.args[0] for call in message.answer.call_args_list]
    assert len(chunks) > 1
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 4096 for chunk in chunks)
    assert "".join(chunks) == text
    assert all(f"{task.identifier}:" in text and task.url in text for task in tasks)
