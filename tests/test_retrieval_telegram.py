from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText
from aiogram.types import CallbackQuery, Chat, Message, User

from src.application.ports.calendar import CalendarError, CalendarNotConnectedError
from src.application.ports.tasks import TaskTrackerError
from src.domain.assistant.replies import AssistantReply, ReplyButton, ResultPage
from src.domain.assistant.retrieval import RetrievalLimitError
from src.infrastructure.telegram.handlers import create_router
from src.infrastructure.telegram.replies import answer_reply

PAGE = ResultPage("<b>Страница</b>", ((ReplyButton("Далее", "browse:token:page:1"),),))


@pytest.mark.asyncio
async def test_reply_sends_each_source_as_separate_html_page() -> None:
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    await answer_reply(
        message, AssistantReply(text="Обзор", confirmations=(), pages=(PAGE, PAGE))
    )
    assert message.answer.await_count == 3
    kwargs = message.answer.call_args.kwargs
    assert kwargs["parse_mode"] == "HTML" and kwargs["link_preview_options"].is_disabled
    assert (
        kwargs["reply_markup"].inline_keyboard[0][0].callback_data
        == "browse:token:page:1"
    )


def make_callback(
    *,
    user_id: int = 42,
    message: Message | None = None,
    data: str = "browse:token:page:1",
) -> CallbackQuery:
    return CallbackQuery(
        id="query",
        from_user=User(id=user_id, is_bot=False, first_name="User"),
        chat_instance="instance",
        message=message,
        data=data,
    )


def make_message() -> Message:
    return Message(
        message_id=1, date=datetime.now(UTC), chat=Chat(id=42, type="private")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [42, 99])
@pytest.mark.parametrize("has_message", [True, False])
async def test_callback_routing_owner_check_and_early_ack(
    monkeypatch: pytest.MonkeyPatch, user_id: int, has_message: bool
) -> None:
    ack, edit, answer = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(CallbackQuery, "answer", ack)
    monkeypatch.setattr(Message, "edit_text", edit)
    monkeypatch.setattr(Message, "answer", answer)

    async def browse(**kwargs: object) -> ResultPage:
        ack.assert_awaited_once()
        assert kwargs["user_id"] == 42
        return PAGE

    process = SimpleNamespace(browse=AsyncMock(side_effect=browse))
    router = create_router(process_message=process, timezone="UTC", allowed_user_id=42)  # type: ignore[arg-type]
    await router.propagate_event(
        "callback_query",
        make_callback(user_id=user_id, message=make_message() if has_message else None),
        bot=AsyncMock(spec=Bot),
    )
    ack.assert_awaited_once()
    if user_id == 42 and has_message:
        edit.assert_awaited_once()
        process.browse.assert_awaited_once()
    else:
        process.browse.assert_not_awaited()
        edit.assert_not_awaited()
    answer.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (CalendarNotConnectedError(), "/calendar_connect"),
        (CalendarError(), "Не удалось обновить"),
        (TaskTrackerError(), "Не удалось обновить"),
        (RetrievalLimitError(), "Уточните"),
        (TimeoutError(), "Уточните"),
        (RuntimeError("private detail"), "Не удалось открыть"),
    ],
)
async def test_callback_errors_leave_existing_page_and_explain(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str
) -> None:
    answer, edit = AsyncMock(), AsyncMock()
    monkeypatch.setattr(CallbackQuery, "answer", AsyncMock())
    monkeypatch.setattr(Message, "answer", answer)
    monkeypatch.setattr(Message, "edit_text", edit)
    process = SimpleNamespace(browse=AsyncMock(side_effect=error))
    router = create_router(process_message=process, timezone="UTC", allowed_user_id=42)  # type: ignore[arg-type]
    await router.propagate_event(
        "callback_query", make_callback(message=make_message()), bot=AsyncMock(spec=Bot)
    )
    assert expected in answer.call_args.args[0]
    assert "private detail" not in answer.call_args.args[0]
    edit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", ["message is not modified", "message can't be edited"]
)
async def test_duplicate_click_or_uneditable_message(
    monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    answer = AsyncMock()
    monkeypatch.setattr(CallbackQuery, "answer", AsyncMock())
    monkeypatch.setattr(Message, "answer", answer)
    monkeypatch.setattr(
        Message,
        "edit_text",
        AsyncMock(
            side_effect=TelegramBadRequest(
                method=EditMessageText(text="page"), message=error
            )
        ),
    )
    process = SimpleNamespace(browse=AsyncMock(return_value=PAGE))
    router = create_router(process_message=process, timezone="UTC", allowed_user_id=42)  # type: ignore[arg-type]
    await router.propagate_event(
        "callback_query", make_callback(message=make_message()), bot=AsyncMock(spec=Bot)
    )
    assert answer.await_count == (0 if error == "message is not modified" else 1)


@pytest.mark.asyncio
async def test_calendar_confirmation_keeps_results_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer, edit = AsyncMock(), AsyncMock()
    monkeypatch.setattr(CallbackQuery, "answer", AsyncMock())
    monkeypatch.setattr(Message, "answer", answer)
    monkeypatch.setattr(Message, "edit_text", edit)
    process = SimpleNamespace(browse=AsyncMock(return_value=PAGE))
    router = create_router(process_message=process, timezone="UTC", allowed_user_id=42)  # type: ignore[arg-type]
    await router.propagate_event(
        "callback_query",
        make_callback(message=make_message(), data="browse:token:delete:0"),
        bot=AsyncMock(spec=Bot),
    )
    answer.assert_awaited_once()
    edit.assert_not_awaited()
