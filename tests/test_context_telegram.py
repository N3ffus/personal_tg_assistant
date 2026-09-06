from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Router
from aiogram.filters import CommandObject
from aiogram.types import (
    BusinessBotRights,
    BusinessConnection,
    Chat,
    InlineKeyboardMarkup,
    Message,
    User,
)
from cryptography.fernet import Fernet

from src.application.services.context import ContextService
from src.domain.assistant.context import ContextMessage
from src.infrastructure.context.storage import ContextStorage
from src.infrastructure.telegram import context as context_handlers
from src.infrastructure.telegram.business import create_business_router
from src.infrastructure.telegram.context import create_context_router
from src.infrastructure.telegram.handlers import create_router

NOW = datetime(2026, 9, 5, 12, tzinfo=UTC)
Handler = Callable[..., Coroutine[Any, Any, None]]


def handler(router: Router, observer: str, name: str) -> Handler:
    return cast(
        Handler,
        next(
            item.callback
            for item in getattr(router, observer).handlers
            if item.callback.__name__ == name
        ),
    )


def message(
    *, owner_id: int = 42, chat_id: int = 42, chat_type: str = "private"
) -> Any:
    result = AsyncMock(spec=Message)
    result.answer = AsyncMock()
    result.edit_text = AsyncMock()
    result.answer_document = AsyncMock()
    result.from_user = SimpleNamespace(id=owner_id)
    result.chat = SimpleNamespace(id=chat_id, type=chat_type)
    result.message_id = 123
    result.text = "Привет"
    return result


def callback(data: str | None, target: Any, *, owner_id: int = 42) -> Any:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=owner_id),
        message=target,
        answer=AsyncMock(),
    )


def data_for(
    keyboard: InlineKeyboardMarkup, action: str, *, value: int | None = None
) -> str:
    return next(
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
        and button.callback_data.split(":")[2] == action
        and (value is None or button.callback_data.split(":")[3] == str(value))
    )


async def setup(tmp_path: Path) -> tuple[ContextService, Router, AsyncMock]:
    storage = ContextStorage(
        database_path=str(tmp_path / "assistant.db"),
        encryption_key=Fernet.generate_key().decode(),
    )
    await storage.initialize()
    summarize = AsyncMock(return_value="Краткое резюме")
    service = ContextService(
        storage=storage, summarizer=SimpleNamespace(summarize=summarize)
    )
    for kind, chat_id, title in [("bot", 42, "Чат с ботом"), ("business", 99, "Анна")]:
        chat = await service.ensure_chat(
            owner_id=42,
            kind=cast(Literal["bot", "business"], kind),
            chat_id=chat_id,
            title=title,
        )
        await service.record(
            owner_id=42,
            context_id=chat.id,
            message=ContextMessage(
                role="user",
                sender=title,
                text=f"Текст {title}",
                sent_at=NOW,
                message_id=1,
            ),
        )
    return (
        service,
        create_context_router(contexts=service, allowed_user_id=42),
        summarize,
    )


async def open_menu(
    router: Router, operation: str, target: Any
) -> InlineKeyboardMarkup:
    await handler(router, "message", "context_command")(
        target, CommandObject(command=operation)
    )
    return cast(InlineKeyboardMarkup, target.answer.call_args.kwargs["reply_markup"])


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["clear", "compact"])
async def test_select_all_and_execute_once(tmp_path: Path, operation: str) -> None:
    service, router, summarize = await setup(tmp_path)
    target = message()
    keyboard = await open_menu(router, operation, target)
    click = handler(router, "callback_query", "context_callback")
    empty = callback(data_for(keyboard, "run"), target)
    await click(empty)
    assert "Сначала" in empty.answer.call_args.args[0]
    await click(callback(data_for(keyboard, "all"), target))
    assert "Выбрано: 2" in target.edit_text.call_args.args[0]
    action = data_for(keyboard, "run")
    await click(callback(action, target))
    assert "✅" in target.answer.call_args.args[0]
    for chat in await service.list_chats(owner_id=42):
        context = await service.read(owner_id=42, context_id=chat.id)
        assert not context.messages
        assert bool(context.summary) == (operation == "compact")
    assert summarize.await_count == (2 if operation == "compact" else 0)
    replay = callback(action, target)
    await click(replay)
    assert "устарело" in replay.answer.call_args.args[0]
    assert summarize.await_count == (2 if operation == "compact" else 0)


@pytest.mark.asyncio
async def test_select_multiple_across_pages_and_clear_only_selection(
    tmp_path: Path,
) -> None:
    service, router, _ = await setup(tmp_path)
    for number in range(20):
        await service.ensure_chat(
            owner_id=42, kind="business", chat_id=1000 + number, title=f"Chat {number}"
        )
    target = message()
    keyboard = await open_menu(router, "clear", target)
    click = handler(router, "callback_query", "context_callback")
    token = data_for(keyboard, "all").split(":")[1]
    for page in [1, 2, 0]:
        await click(callback(f"ctx:{token}:page:{page}", target))
        current = target.edit_text.call_args.kwargs["reply_markup"]
        await click(callback(data_for(current, "toggle"), target))
    assert "Выбрано: 3" in target.edit_text.call_args.args[0]
    await click(callback(data_for(keyboard, "run"), target))
    bot, business, *_ = await service.list_chats(owner_id=42)
    assert not (await service.read(owner_id=42, context_id=bot.id)).render()
    assert (await service.read(owner_id=42, context_id=business.id)).messages


@pytest.mark.asyncio
async def test_deselect_close_and_expired_menus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, router, _ = await setup(tmp_path)
    target = message()
    keyboard = await open_menu(router, "clear", target)
    click = handler(router, "callback_query", "context_callback")
    for _ in range(2):
        await click(callback(data_for(keyboard, "all"), target))
    assert "Выбрано: 0" in target.edit_text.call_args.args[0]
    for _ in range(2):
        await click(callback(data_for(keyboard, "toggle"), target))
    assert "Выбрано: 0" in target.edit_text.call_args.args[0]
    await click(callback(data_for(keyboard, "close"), target))
    assert "закрыто" in target.edit_text.call_args.args[0]
    keyboard = await open_menu(router, "compact", target)
    monkeypatch.setattr(
        context_handlers, "time", SimpleNamespace(monotonic=lambda: float("inf"))
    )
    expired = callback(data_for(keyboard, "all"), target)
    await click(expired)
    assert "устарело" in expired.answer.call_args.args[0]


@pytest.mark.asyncio
async def test_history_pages_summary_empty_context_and_export(tmp_path: Path) -> None:
    service, router, _ = await setup(tmp_path)
    bot, business = await service.list_chats(owner_id=42)
    await service.record(
        owner_id=42,
        context_id=business.id,
        message=ContextMessage(
            role="user",
            sender="Анна",
            text="Длинная история " * 400,
            sent_at=NOW,
            message_id=2,
        ),
    )
    await service.compact(owner_id=42, context_id=bot.id)
    target = message()
    keyboard = await open_menu(router, "history", target)
    click = handler(router, "callback_query", "context_callback")
    await click(callback(data_for(keyboard, "show", value=bot.id), target))
    assert "Резюме предыдущей переписки" in target.edit_text.call_args.args[0]
    await click(callback(data_for(keyboard, "show", value=business.id), target))
    assert "Страница 1/" in target.edit_text.call_args.args[0]
    current = target.edit_text.call_args.kwargs["reply_markup"]
    await click(callback(data_for(current, "history"), target))
    assert "Страница 2/" in target.edit_text.call_args.args[0]
    await click(callback(data_for(current, "page"), target))
    assert "История контекста" in target.edit_text.call_args.args[0]
    await click(callback(data_for(keyboard, "export"), target))
    document = target.answer_document.call_args.args[0]
    assert document.filename == "chat-contexts.txt"
    text = document.data.decode()
    assert "Краткое резюме" in text and "Длинная история" in text
    await service.clear(owner_id=42, context_id=bot.id)
    await click(callback(data_for(keyboard, "show", value=bot.id), target))
    assert "Контекст пуст." in target.edit_text.call_args.args[0]
    assert target.edit_text.call_args.kwargs["parse_mode"] is None


@pytest.mark.asyncio
async def test_context_access_checks_and_invalid_buttons(tmp_path: Path) -> None:
    service, router, _ = await setup(tmp_path)
    unauthorized = message(owner_id=99)
    await handler(router, "message", "context_command")(
        unauthorized, CommandObject(command="history")
    )
    unauthorized.answer.assert_not_awaited()
    group = message(chat_type="group")
    await handler(router, "message", "context_command")(
        group, CommandObject(command="history")
    )
    assert "личном чате" in group.answer.call_args.args[0]
    target = message()
    keyboard = await open_menu(router, "clear", target)
    click = handler(router, "callback_query", "context_callback")
    token = data_for(keyboard, "all").split(":")[1]
    forged_data = [
        None,
        "ctx:broken",
        f"ctx:{token}:toggle:99999",
        f"ctx:{token}:toggle:not-a-number",
        "ctx:unknown:run:0",
    ]
    for data in forged_data:
        query = callback(data, target)
        await click(query)
        query.answer.assert_awaited_once()
    for bad_target, owner_id in [
        (target, 99),
        (None, 42),
        (group, 42),
        (message(chat_id=999), 42),
    ]:
        query = callback(data_for(keyboard, "all"), bad_target, owner_id=owner_id)
        await click(query)
        query.answer.assert_awaited_once()
    assert "Контекст" not in str(group.edit_text.call_args)
    assert all(
        [
            (await service.read(owner_id=42, context_id=chat.id)).messages
            for chat in await service.list_chats(owner_id=42)
        ]
    )


@pytest.mark.asyncio
async def test_bulk_compact_reports_partial_failure_without_losing_failed_chat(
    tmp_path: Path,
) -> None:
    service, router, summarize = await setup(tmp_path)
    summarize.side_effect = [RuntimeError("private provider error"), "Резюме Анны"]
    target = message()
    keyboard = await open_menu(router, "compact", target)
    click = handler(router, "callback_query", "context_callback")
    await click(callback(data_for(keyboard, "all"), target))
    await click(callback(data_for(keyboard, "run"), target))
    result = target.answer.call_args.args[0]
    assert "❌" in result and "✅" in result
    assert "private provider" not in result
    bot, business = await service.list_chats(owner_id=42)
    assert (await service.read(owner_id=42, context_id=bot.id)).messages
    assert (
        await service.read(owner_id=42, context_id=business.id)
    ).summary == "Резюме Анны"


@pytest.mark.asyncio
async def test_menus_are_bounded_and_storage_errors_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, router, _ = await setup(tmp_path)
    target = message()
    monkeypatch.setattr(context_handlers, "MENU_LIMIT", 2)
    first = await open_menu(router, "clear", target)
    await open_menu(router, "history", target)
    current = await open_menu(router, "history", target)
    query = callback(data_for(first, "all"), target)
    await handler(router, "callback_query", "context_callback")(query)
    assert "устарело" in query.answer.call_args.args[0]
    monkeypatch.setattr(
        service, "read", AsyncMock(side_effect=RuntimeError("private db error"))
    )
    await handler(router, "callback_query", "context_callback")(
        callback(data_for(current, "show"), target)
    )
    assert "Не удалось" in target.answer.call_args.args[0]
    monkeypatch.setattr(
        service, "list_chats", AsyncMock(side_effect=RuntimeError("private db error"))
    )
    await handler(router, "message", "context_command")(
        target, CommandObject(command="history")
    )
    assert "Не удалось открыть" in target.answer.call_args.args[0]


def business_message(
    *,
    sender_id: int = 99,
    text: str | None = "Сообщение",
    caption: str | None = None,
    business_bot: bool = False,
) -> Message:
    return Message(
        message_id=11,
        date=NOW,
        text=text,
        caption=caption,
        chat=Chat(id=99, type="private", first_name="Анна"),
        from_user=User(id=sender_id, is_bot=False, first_name="Анна"),
        sender_business_bot=User(id=100, is_bot=True, first_name="Helper")
        if business_bot
        else None,
        business_connection_id="connection",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sender_id", "business_bot", "label"),
    [(99, False, "Анна"), (42, False, "Вы"), (99, True, "Бизнес-бот")],
)
async def test_business_stores_both_sides_and_edits_without_replying(
    tmp_path: Path, sender_id: int, business_bot: bool, label: str
) -> None:
    service, _, _ = await setup(tmp_path)
    router = create_business_router(allowed_user_id=42, contexts=service)
    connection = BusinessConnection(
        id="connection",
        user=User(id=42, is_bot=False, first_name="Owner"),
        user_chat_id=42,
        date=NOW,
        is_enabled=True,
        rights=BusinessBotRights(can_read_messages=True),
    )
    bot = SimpleNamespace(
        get_business_connection=AsyncMock(return_value=connection),
        read_business_message=AsyncMock(),
        send_message=AsyncMock(),
    )
    original = business_message(
        sender_id=sender_id, business_bot=business_bot
    ).model_copy(
        update={
            "reply_to_message": business_message().model_copy(update={"message_id": 10})
        }
    )
    read = handler(router, "business_message", "read_business_message")
    await read(original, bot)
    await read(original, bot)
    await read(original.model_copy(update={"text": "Исправлено"}), bot)
    business = next(
        chat
        for chat in await service.list_chats(owner_id=42)
        if chat.kind == "business"
    )
    context = await service.read(owner_id=42, context_id=business.id)
    assert len(context.messages) == 2
    assert context.messages[-1].text == "Исправлено"
    assert context.messages[-1].sender == label
    assert context.messages[-1].reply_to_message_id == 10
    assert business.title == "Анна"
    assert bool(bot.read_business_message.await_count) == (
        sender_id == 99 and not business_bot
    )
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_business_caption_storage_and_rejected_owner(tmp_path: Path) -> None:
    service, _, _ = await setup(tmp_path)
    router = create_business_router(allowed_user_id=42, contexts=service)
    owner = User(id=42, is_bot=False, first_name="Owner")
    connection = BusinessConnection(
        id="connection", user=owner, user_chat_id=42, date=NOW, is_enabled=True
    )
    bot = SimpleNamespace(
        get_business_connection=AsyncMock(return_value=connection),
        read_business_message=AsyncMock(),
    )
    read = handler(router, "business_message", "read_business_message")
    await read(business_message(text=None, caption="Подпись фото"), bot)
    business = next(
        chat
        for chat in await service.list_chats(owner_id=42)
        if chat.kind == "business"
    )
    assert (
        "Подпись фото"
        in (await service.read(owner_id=42, context_id=business.id)).render()
    )
    await handler(router, "business_connection", "connection_changed")(
        connection.model_copy(update={"is_enabled": False})
    )
    await read(business_message(text="Нельзя сохранить"), bot)
    assert (
        "Нельзя сохранить"
        not in (await service.read(owner_id=42, context_id=business.id)).render()
    )


@pytest.mark.asyncio
async def test_common_router_supplies_chat_identity_when_context_enabled(
    tmp_path: Path,
) -> None:
    service, _, _ = await setup(tmp_path)
    process = SimpleNamespace(execute=AsyncMock(return_value="Ответ"))
    router = create_router(
        process_message=process,  # type: ignore[arg-type]
        timezone="UTC",
        allowed_user_id=42,
        contexts=service,
    )
    target = message()
    await handler(router, "message", "text_handler")(target)
    assert process.execute.call_args.kwargs["chat_id"] == 42
    assert process.execute.call_args.kwargs["message_id"] == 123


@pytest.mark.asyncio
async def test_history_with_emoji_stays_within_telegram_message_limit(
    tmp_path: Path,
) -> None:
    service, router, _ = await setup(tmp_path)
    business = next(
        chat
        for chat in await service.list_chats(owner_id=42)
        if chat.kind == "business"
    )
    await service.record(
        owner_id=42,
        context_id=business.id,
        message=ContextMessage(
            role="user",
            sender="Анна",
            text="😀" * 5000,
            sent_at=NOW,
            message_id=2,
        ),
    )
    target = message()
    keyboard = await open_menu(router, "history", target)
    click = handler(router, "callback_query", "context_callback")
    await click(callback(data_for(keyboard, "show", value=business.id), target))
    for _ in range(3):
        text = target.edit_text.call_args.args[0]
        assert len(text.encode("utf-16-le")) // 2 <= 4096
        keyboard = target.edit_text.call_args.kwargs["reply_markup"]
        next_buttons = [
            button.callback_data
            for row in keyboard.inline_keyboard
            for button in row
            if button.text == "Далее →"
        ]
        if not next_buttons:
            break
        await click(callback(next_buttons[0], target))
