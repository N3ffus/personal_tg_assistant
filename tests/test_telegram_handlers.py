from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, tzinfo
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageReplyMarkup, SetMyCommands
from aiogram.types import Message

from src.application.ports.calendar import (
    CalendarError,
    CalendarEventNotFoundError,
    CalendarNotConnectedError,
)
from src.application.ports.tasks import TaskCreationUncertainError, TaskTrackerError
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.domain.assistant.replies import AssistantReply, Confirmation
from src.domain.calendar.models import CalendarEvent
from src.infrastructure.telegram import calendar as calendar_handlers
from src.infrastructure.telegram import handlers as common_handlers
from src.infrastructure.telegram.commands import configure_commands

Handler = Callable[[Any], Coroutine[Any, Any, None]]
FIXED_NOW = datetime(2026, 9, 3, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [42, 99, None])
async def test_removed_profile_command_never_reaches_llm(user_id: int | None) -> None:
    process = SimpleNamespace(execute=AsyncMock())
    router = common_handlers.create_router(
        process_message=process,  # type: ignore[arg-type]
        timezone="UTC",
        allowed_user_id=42,
    )
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.text = "/profile"
    message.from_user = SimpleNamespace(id=user_id) if user_id is not None else None
    await router.propagate_event(
        update_type="message", event=message, bot=AsyncMock(spec=Bot)
    )
    process.execute.assert_not_awaited()
    if user_id == 42:
        assert "меню" in message.answer.call_args.args[0]
    else:
        message.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_menu_api_failure_does_not_abort_startup() -> None:
    bot = AsyncMock(spec=Bot)
    bot.set_my_commands.side_effect = TelegramBadRequest(
        method=SetMyCommands(commands=[]), message="temporarily unavailable"
    )
    await configure_commands(bot)
    bot.set_chat_menu_button.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_handler_sends_inline_delete_and_cancel_buttons() -> None:
    process_message = SimpleNamespace(
        execute=AsyncMock(
            return_value=AssistantReply(
                text="",
                confirmations=(
                    Confirmation(
                        text="Удалить задачу ENG-1?", operation_id="opaque-token"
                    ),
                ),
            )
        )
    )
    router = common_handlers.create_router(
        process_message=cast(ProcessMessageUseCase, process_message),
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = FakeMessage(text="Удали все задачи в Linear")
    await find_handler(router, "message", "text_handler")(message)
    keyboard = message.answers[0][1]["reply_markup"]
    buttons = keyboard.inline_keyboard[0]
    assert [b.callback_data for b in buttons] == [
        "delyes:opaque-token",
        "delno:opaque-token",
    ]
    assert "Удалить" in buttons[0].text
    assert buttons[1].text == "Отмена"


@pytest.mark.asyncio
@pytest.mark.parametrize("confirm", [True, False])
async def test_deletion_callback_resolves_button_and_removes_keyboard(
    confirm: bool,
) -> None:
    process_message = SimpleNamespace(
        resolve_deletion=AsyncMock(return_value="Удалено" if confirm else "Отменено")
    )
    router = common_handlers.create_router(
        process_message=cast(ProcessMessageUseCase, process_message),
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.edit_reply_markup = AsyncMock()
    callback = FakeCallback(
        data=f"{'delyes' if confirm else 'delno'}:token", message=message
    )
    await find_handler(router, "callback_query", "deletion_callback")(callback)
    assert callback.answers == [None]
    process_message.resolve_deletion.assert_awaited_once_with(
        user_id=42, operation_id="token", confirm=confirm
    )
    message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
    message.answer.assert_awaited_once_with("Удалено" if confirm else "Отменено")


@pytest.mark.asyncio
@pytest.mark.parametrize(("user_id", "data"), [(99, "delyes:token"), (42, None)])
async def test_deletion_callback_rejects_other_users_and_missing_data(
    user_id: int, data: str | None
) -> None:
    process_message = SimpleNamespace(resolve_deletion=AsyncMock())
    router = common_handlers.create_router(
        process_message=cast(ProcessMessageUseCase, process_message),
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    await find_handler(router, "callback_query", "deletion_callback")(
        FakeCallback(data=data, user_id=user_id)
    )
    process_message.resolve_deletion.assert_not_awaited()


@pytest.mark.asyncio
async def test_deletion_callback_handles_storage_and_keyboard_errors() -> None:
    process_message = SimpleNamespace(
        resolve_deletion=AsyncMock(side_effect=RuntimeError("private details"))
    )
    router = common_handlers.create_router(
        process_message=cast(ProcessMessageUseCase, process_message),
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.edit_reply_markup = AsyncMock()
    message.edit_reply_markup.side_effect = TelegramBadRequest(
        method=EditMessageReplyMarkup(), message="already changed"
    )
    callback = FakeCallback(data="delyes:token", message=message)
    await find_handler(router, "callback_query", "deletion_callback")(callback)
    assert "Не удалось" in message.answer.await_args.args[0]
    assert "private details" not in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_deletion_callback_without_message_still_acknowledged() -> None:
    process_message = SimpleNamespace(
        resolve_deletion=AsyncMock(return_value="Удалено")
    )
    router = common_handlers.create_router(
        process_message=cast(ProcessMessageUseCase, process_message),
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    callback = FakeCallback(data="delyes:token")
    await find_handler(router, "callback_query", "deletion_callback")(callback)
    assert callback.answers == [None]
    process_message.resolve_deletion.assert_awaited_once()


class FixedClock:
    @staticmethod
    def now(tz: tzinfo | None = None) -> datetime:
        if tz is None:
            return FIXED_NOW.replace(tzinfo=None)
        return FIXED_NOW.astimezone(tz)


class FakeMessage:
    def __init__(
        self,
        *,
        user_id: int | None = 42,
        text: str | None = None,
    ) -> None:
        self.from_user = SimpleNamespace(id=user_id) if user_id is not None else None
        self.text = text
        self.answers: list[tuple[str, dict[str, Any]]] = []

    async def answer(self, text: str, **kwargs: Any) -> None:
        self.answers.append((text, kwargs))


class FakeCallback:
    def __init__(
        self,
        *,
        data: str | None,
        user_id: int = 42,
        message: FakeMessage | None = None,
    ) -> None:
        self.data = data
        self.from_user = SimpleNamespace(id=user_id)
        self.message = message
        self.answers: list[str | None] = []

    async def answer(self, text: str | None = None) -> None:
        self.answers.append(text)


def find_handler(router: Router, observer: str, name: str) -> Handler:
    handlers = getattr(router, observer).handlers
    return cast(
        Handler,
        next(
            handler.callback
            for handler in handlers
            if handler.callback.__name__ == name
        ),
    )


def calendar_router(
    *,
    calendar: SimpleNamespace | None = None,
    oauth: SimpleNamespace | None = None,
    storage: SimpleNamespace | None = None,
) -> tuple[Router, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    calendar_double = calendar or SimpleNamespace(
        list_events=AsyncMock(return_value=[]),
        disconnect=AsyncMock(),
        delete_event=AsyncMock(),
    )
    oauth_double = oauth or SimpleNamespace(
        authorization_url=AsyncMock(return_value="https://google.test/oauth")
    )
    storage_double = storage or SimpleNamespace(
        create_operations=AsyncMock(return_value=["selection-1"]),
        create_operation=AsyncMock(return_value="operation-1"),
        consume_operation=AsyncMock(return_value=None),
    )
    router = calendar_handlers.create_calendar_router(
        timezone="Europe/Moscow",
        allowed_user_id=42,
        calendar=calendar_double,
        oauth=oauth_double,  # type: ignore[arg-type]
        storage=storage_double,  # type: ignore[arg-type]
    )
    return router, calendar_double, oauth_double, storage_double


@pytest.mark.asyncio
async def test_start_handler_explains_available_commands() -> None:
    process_message = SimpleNamespace(execute=AsyncMock())
    router = common_handlers.create_router(
        process_message=process_message,  # type: ignore[arg-type]
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = FakeMessage()

    await find_handler(router, "message", "start_handler")(message)

    assert len(message.answers) == 1
    assert "ИИ-помощник" in message.answers[0][0]
    assert "Завтра в 15:00 стоматолог" in message.answers[0][0]
    process_message.execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [None, 99])
async def test_start_handler_ignores_other_users(user_id: int | None) -> None:
    process_message = SimpleNamespace(execute=AsyncMock())
    router = common_handlers.create_router(
        process_message=process_message,  # type: ignore[arg-type]
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = FakeMessage(user_id=user_id)

    await find_handler(router, "message", "start_handler")(message)

    assert message.answers == []
    process_message.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_handler_passes_user_time_and_timezone_to_use_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common_handlers, "datetime", FixedClock)
    process_message = SimpleNamespace(execute=AsyncMock(return_value="Готово"))
    router = common_handlers.create_router(
        process_message=process_message,  # type: ignore[arg-type]
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = FakeMessage(text="Создай событие")

    await find_handler(router, "message", "text_handler")(message)

    process_message.execute.assert_awaited_once_with(
        text="Создай событие",
        now=FIXED_NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )
    assert message.answers == [("Готово", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        FakeMessage(text=None),
        FakeMessage(text=""),
        FakeMessage(user_id=None, text="hello"),
        FakeMessage(user_id=99, text="hello"),
    ],
)
async def test_text_handler_ignores_messages_without_text_or_user(
    message: FakeMessage,
) -> None:
    process_message = SimpleNamespace(execute=AsyncMock())
    router = common_handlers.create_router(
        process_message=process_message,  # type: ignore[arg-type]
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )

    await find_handler(router, "message", "text_handler")(message)

    process_message.execute.assert_not_awaited()
    assert message.answers == []


@pytest.mark.asyncio
async def test_text_handler_hides_processing_failure() -> None:
    process_message = SimpleNamespace(
        execute=AsyncMock(side_effect=RuntimeError("private details"))
    )
    router = common_handlers.create_router(
        process_message=process_message,  # type: ignore[arg-type]
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = FakeMessage(text="hello")

    await find_handler(router, "message", "text_handler")(message)

    assert message.answers == [("Не получилось обработать сообщение.", {})]


@pytest.mark.asyncio
async def test_text_handler_reports_linear_failure() -> None:
    process_message = SimpleNamespace(
        execute=AsyncMock(side_effect=TaskTrackerError("private API details"))
    )
    router = common_handlers.create_router(
        process_message=process_message,  # type: ignore[arg-type]
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = FakeMessage(text="Создай задачу")

    await find_handler(router, "message", "text_handler")(message)

    assert message.answers == [
        ("Не удалось создать задачу в Linear. Попробуйте позже.", {})
    ]
    assert "private API details" not in message.answers[0][0]


@pytest.mark.asyncio
async def test_text_handler_warns_when_linear_result_is_uncertain() -> None:
    process_message = SimpleNamespace(
        execute=AsyncMock(side_effect=TaskCreationUncertainError("private details"))
    )
    router = common_handlers.create_router(
        process_message=process_message,  # type: ignore[arg-type]
        timezone="Europe/Moscow",
        allowed_user_id=42,
    )
    message = FakeMessage(text="Создай задачу")

    await find_handler(router, "message", "text_handler")(message)

    assert message.answers == [
        (
            "Linear мог успеть создать задачу. Проверьте список задач перед повтором.",
            {},
        )
    ]
    assert "private details" not in message.answers[0][0]


@pytest.mark.asyncio
async def test_calendar_connect_sends_personal_oauth_url() -> None:
    router, _, oauth, _ = calendar_router()
    message = FakeMessage()

    await find_handler(router, "message", "connect")(message)

    oauth.authorization_url.assert_awaited_once_with(user_id=42)
    assert message.answers == [
        ("Подключите Google Calendar: https://google.test/oauth", {})
    ]


@pytest.mark.asyncio
async def test_calendar_connect_reports_oauth_failure_without_leaking_error() -> None:
    oauth = SimpleNamespace(
        authorization_url=AsyncMock(side_effect=RuntimeError("client secret"))
    )
    router, _, _, _ = calendar_router(oauth=oauth)
    message = FakeMessage()

    await find_handler(router, "message", "connect")(message)

    assert message.answers == [("Не удалось начать подключение календаря.", {})]
    assert "client secret" not in message.answers[0][0]


@pytest.mark.asyncio
async def test_calendar_connect_ignores_channel_post_without_user() -> None:
    router, _, oauth, _ = calendar_router()
    message = FakeMessage(user_id=None)

    await find_handler(router, "message", "connect")(message)

    oauth.authorization_url.assert_not_awaited()
    assert message.answers == []


@pytest.mark.asyncio
async def test_calendar_disconnect_reports_success() -> None:
    router, calendar, _, _ = calendar_router()
    message = FakeMessage()

    await find_handler(router, "message", "disconnect")(message)

    calendar.disconnect.assert_awaited_once_with(user_id=42)
    assert message.answers == [("Google Calendar отключён.", {})]


@pytest.mark.asyncio
async def test_calendar_disconnect_reports_expected_failure() -> None:
    calendar = SimpleNamespace(
        list_events=AsyncMock(),
        disconnect=AsyncMock(side_effect=CalendarError),
        delete_event=AsyncMock(),
    )
    router, _, _, _ = calendar_router(calendar=calendar)
    message = FakeMessage()

    await find_handler(router, "message", "disconnect")(message)

    assert message.answers == [("Не удалось отключить Google Calendar.", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["disconnect", "list_events"])
async def test_calendar_command_ignores_channel_post_without_user(
    handler_name: str,
) -> None:
    router, calendar, _, _ = calendar_router()
    message = FakeMessage(user_id=None)

    await find_handler(router, "message", handler_name)(message)

    calendar.disconnect.assert_not_awaited()
    calendar.list_events.assert_not_awaited()
    assert message.answers == []


@pytest.mark.asyncio
async def test_calendar_list_shows_events_with_server_verified_action_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calendar_handlers, "datetime", FixedClock)
    starts_at = datetime(2026, 9, 4, 15, 30, tzinfo=ZoneInfo("Europe/Moscow"))
    event = CalendarEvent(
        event_id="event-1",
        title="Очень важная встреча",
        starts_at=starts_at,
        ends_at=starts_at + timedelta(hours=1),
        html_link=None,
    )
    calendar = SimpleNamespace(
        list_events=AsyncMock(return_value=[event]),
        disconnect=AsyncMock(),
        delete_event=AsyncMock(),
    )
    router, _, _, storage = calendar_router(calendar=calendar)
    message = FakeMessage()

    await find_handler(router, "message", "list_events")(message)

    calendar.list_events.assert_awaited_once_with(user_id=42, now=FIXED_NOW)
    storage.create_operations.assert_awaited_once_with(
        user_id=42,
        kind="select",
        payloads=[{"event_id": "event-1"}],
    )
    text, kwargs = message.answers[0]
    assert text == (
        "📅 Ближайшие события Google Calendar (до 10):\n"
        "• 04.09.2026 15:30 MSK — Очень важная встреча"
    )
    keyboard = kwargs["reply_markup"]
    assert keyboard.inline_keyboard[0][0].callback_data == "caledit:selection-1"
    assert keyboard.inline_keyboard[0][1].callback_data == "caldel:selection-1"


@pytest.mark.asyncio
async def test_calendar_list_omits_keyboard_when_empty() -> None:
    router, calendar, _, storage = calendar_router()
    message = FakeMessage()

    await find_handler(router, "message", "list_events")(message)

    calendar.list_events.assert_awaited_once()
    storage.create_operations.assert_not_awaited()
    assert message.answers == [("Ближайших событий не найдено.", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            CalendarNotConnectedError(),
            "Подключите календарь командой /calendar_connect.",
        ),
        (CalendarError(), "Google Calendar временно недоступен."),
    ],
)
async def test_calendar_list_distinguishes_connection_and_api_errors(
    error: CalendarError,
    expected: str,
) -> None:
    calendar = SimpleNamespace(
        list_events=AsyncMock(side_effect=error),
        disconnect=AsyncMock(),
        delete_event=AsyncMock(),
    )
    router, _, _, _ = calendar_router(calendar=calendar)
    message = FakeMessage()

    await find_handler(router, "message", "list_events")(message)

    assert message.answers == [(expected, {})]


@pytest.mark.asyncio
async def test_request_update_stores_event_selection_and_prompts_for_new_values() -> (
    None
):
    storage = SimpleNamespace(
        create_operation=AsyncMock(return_value="operation-1"),
        consume_operation=AsyncMock(return_value=("select", {"event_id": "event-1"})),
    )
    router, _, _, _ = calendar_router(storage=storage)
    message = FakeMessage()
    callback = FakeCallback(data="caledit:selection-1", message=message)

    await find_handler(router, "callback_query", "request_update")(callback)

    storage.consume_operation.assert_awaited_once_with(
        operation_id="selection-1",
        user_id=42,
    )
    storage.create_operation.assert_awaited_once_with(
        user_id=42,
        kind="update",
        payload={"event_id": "event-1"},
    )
    assert callback.answers == [None]
    assert "новое название и время" in message.answers[0][0]


@pytest.mark.asyncio
async def test_request_delete_creates_one_time_confirmation() -> None:
    storage = SimpleNamespace(
        create_operation=AsyncMock(return_value="operation-1"),
        consume_operation=AsyncMock(return_value=("select", {"event_id": "event-1"})),
    )
    router, _, _, _ = calendar_router(storage=storage)
    message = FakeMessage()
    callback = FakeCallback(data="caldel:selection-1", message=message)

    await find_handler(router, "callback_query", "request_delete")(callback)

    storage.consume_operation.assert_awaited_once_with(
        operation_id="selection-1",
        user_id=42,
    )
    storage.create_operation.assert_awaited_once_with(
        user_id=42,
        kind="delete",
        payload={"event_id": "event-1"},
    )
    assert callback.answers == [None]
    text, kwargs = message.answers[0]
    assert text == "Удалить это событие?"
    buttons = kwargs["reply_markup"].inline_keyboard[0]
    assert [button.callback_data for button in buttons] == [
        "calyes:operation-1",
        "calno:operation-1",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["request_update", "request_delete"])
async def test_selection_callback_without_message_is_still_acknowledged(
    handler_name: str,
) -> None:
    storage = SimpleNamespace(
        create_operation=AsyncMock(return_value="operation-1"),
        consume_operation=AsyncMock(return_value=("select", {"event_id": "event-1"})),
    )
    router, _, _, _ = calendar_router(storage=storage)
    prefix = "caledit:" if handler_name == "request_update" else "caldel:"
    callback = FakeCallback(data=f"{prefix}selection-1", message=None)

    await find_handler(router, "callback_query", handler_name)(callback)

    storage.create_operation.assert_awaited_once()
    assert callback.answers == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [None, ("update", {"event_id": "event-1"})],
)
async def test_confirm_delete_rejects_missing_or_wrong_operation(
    operation: tuple[str, dict[str, object]] | None,
) -> None:
    storage = SimpleNamespace(
        create_operation=AsyncMock(),
        consume_operation=AsyncMock(return_value=operation),
    )
    router, calendar, _, _ = calendar_router(storage=storage)
    message = FakeMessage()
    callback = FakeCallback(data="calyes:operation-1", message=message)

    await find_handler(router, "callback_query", "confirm_delete")(callback)

    storage.consume_operation.assert_awaited_once_with(
        operation_id="operation-1",
        user_id=42,
    )
    calendar.delete_event.assert_not_awaited()
    assert callback.answers == [None]
    assert message.answers == [("Подтверждение недействительно или устарело.", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [
        (None, "Событие удалено."),
        (CalendarEventNotFoundError(), "Событие уже удалено или не найдено."),
        (CalendarError(), "Не удалось удалить событие. Попробуйте позже."),
    ],
)
async def test_confirm_delete_reports_calendar_result(
    side_effect: Exception | None,
    expected: str,
) -> None:
    calendar = SimpleNamespace(
        list_events=AsyncMock(),
        disconnect=AsyncMock(),
        delete_event=AsyncMock(side_effect=side_effect),
    )
    storage = SimpleNamespace(
        create_operation=AsyncMock(),
        consume_operation=AsyncMock(return_value=("delete", {"event_id": "event-1"})),
    )
    router, _, _, _ = calendar_router(calendar=calendar, storage=storage)
    message = FakeMessage()
    callback = FakeCallback(data="calyes:operation-1", message=message)

    await find_handler(router, "callback_query", "confirm_delete")(callback)

    calendar.delete_event.assert_awaited_once_with(user_id=42, event_id="event-1")
    assert message.answers == [(expected, {})]


@pytest.mark.asyncio
async def test_confirm_delete_without_source_message_still_deletes_event() -> None:
    storage = SimpleNamespace(
        create_operation=AsyncMock(),
        consume_operation=AsyncMock(return_value=("delete", {"event_id": "event-1"})),
    )
    router, calendar, _, _ = calendar_router(storage=storage)
    callback = FakeCallback(data="calyes:operation-1", message=None)

    await find_handler(router, "callback_query", "confirm_delete")(callback)

    calendar.delete_event.assert_awaited_once_with(user_id=42, event_id="event-1")
    assert callback.answers == [None]


@pytest.mark.asyncio
async def test_cancel_delete_consumes_confirmation_and_answers_callback() -> None:
    storage = SimpleNamespace(
        create_operation=AsyncMock(),
        consume_operation=AsyncMock(return_value=("delete", {"event_id": "event-1"})),
    )
    router, _, _, _ = calendar_router(storage=storage)
    callback = FakeCallback(data="calno:operation-1")

    await find_handler(router, "callback_query", "cancel_delete")(callback)

    storage.consume_operation.assert_awaited_once_with(
        operation_id="operation-1",
        user_id=42,
    )
    assert callback.answers == ["Удаление отменено"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_name",
    ["request_update", "request_delete", "confirm_delete", "cancel_delete"],
)
async def test_calendar_callbacks_ignore_missing_data(handler_name: str) -> None:
    router, calendar, _, storage = calendar_router()
    callback = FakeCallback(data=None, message=FakeMessage())

    await find_handler(router, "callback_query", handler_name)(callback)

    storage.create_operation.assert_not_awaited()
    storage.consume_operation.assert_not_awaited()
    calendar.delete_event.assert_not_awaited()
    assert callback.answers == []


@pytest.mark.asyncio
async def test_event_button_text_respects_telegram_limit() -> None:
    starts_at = datetime(2026, 9, 4, tzinfo=ZoneInfo("Europe/Moscow"))
    event = CalendarEvent(
        event_id="event-1",
        title="A" * 100,
        starts_at=starts_at,
        ends_at=starts_at + timedelta(hours=1),
        html_link=None,
    )

    calendar = SimpleNamespace(
        list_events=AsyncMock(return_value=[event]),
        disconnect=AsyncMock(),
        delete_event=AsyncMock(),
    )
    router, _, _, _ = calendar_router(calendar=calendar)
    message = FakeMessage()

    await find_handler(router, "message", "list_events")(message)

    keyboard = message.answers[0][1]["reply_markup"]

    assert all(len(button.text) <= 64 for button in keyboard.inline_keyboard[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["connect", "disconnect", "list_events"])
async def test_calendar_commands_ignore_users_outside_allowlist(
    handler_name: str,
) -> None:
    router, calendar, oauth, storage = calendar_router()
    message = FakeMessage(user_id=99)

    await find_handler(router, "message", handler_name)(message)

    oauth.authorization_url.assert_not_awaited()
    calendar.disconnect.assert_not_awaited()
    calendar.list_events.assert_not_awaited()
    storage.create_operations.assert_not_awaited()
    assert message.answers == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "data"),
    [
        ("request_update", "caledit:selection-1"),
        ("request_delete", "caldel:selection-1"),
        ("confirm_delete", "calyes:operation-1"),
        ("cancel_delete", "calno:operation-1"),
    ],
)
async def test_calendar_callbacks_ignore_users_outside_allowlist(
    handler_name: str,
    data: str,
) -> None:
    router, calendar, _, storage = calendar_router()
    callback = FakeCallback(data=data, user_id=99, message=FakeMessage(user_id=99))

    await find_handler(router, "callback_query", handler_name)(callback)

    storage.create_operation.assert_not_awaited()
    storage.consume_operation.assert_not_awaited()
    calendar.delete_event.assert_not_awaited()
    assert callback.answers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["request_update", "request_delete"])
async def test_calendar_selection_rejects_consumed_or_forged_token(
    handler_name: str,
) -> None:
    storage = SimpleNamespace(
        create_operation=AsyncMock(),
        consume_operation=AsyncMock(return_value=None),
    )
    router, _, _, _ = calendar_router(storage=storage)
    message = FakeMessage()
    prefix = "caledit:" if handler_name == "request_update" else "caldel:"
    callback = FakeCallback(data=f"{prefix}unknown", message=message)

    await find_handler(router, "callback_query", handler_name)(callback)

    storage.create_operation.assert_not_awaited()
    assert callback.answers == [None]
    assert message.answers == [("Кнопка недействительна или устарела.", {})]


@pytest.mark.asyncio
async def test_long_google_event_id_never_enters_telegram_callback_data() -> None:
    starts_at = datetime(2026, 9, 4, tzinfo=ZoneInfo("Europe/Moscow"))
    event_id = "a" * 1024
    event = CalendarEvent(
        event_id=event_id,
        title="Planning",
        starts_at=starts_at,
        ends_at=starts_at + timedelta(hours=1),
        html_link=None,
    )
    calendar = SimpleNamespace(
        list_events=AsyncMock(return_value=[event]),
        disconnect=AsyncMock(),
        delete_event=AsyncMock(),
    )
    router, _, _, storage = calendar_router(calendar=calendar)
    message = FakeMessage()

    await find_handler(router, "message", "list_events")(message)

    keyboard = message.answers[0][1]["reply_markup"]
    callback_data = [button.callback_data for button in keyboard.inline_keyboard[0]]
    assert callback_data == ["caledit:selection-1", "caldel:selection-1"]
    assert all(data is not None and len(data.encode()) <= 64 for data in callback_data)
    storage.create_operations.assert_awaited_once_with(
        user_id=42,
        kind="select",
        payloads=[{"event_id": event_id}],
    )
