import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.methods import GetBusinessConnection
from aiogram.types import (
    BusinessBotRights,
    BusinessConnection,
    Chat,
    Message,
    User,
)

from src.domain.assistant.context import ContextChat
from src.infrastructure.telegram.business import (
    REJECTED_CONNECTION_LIMIT,
    create_business_router,
)

Handler = Callable[..., Coroutine[Any, Any, None]]
ALLOWED_USER_ID = 42
CUSTOMER_USER_ID = 99
CONNECTION_ID = "business-connection-1"
FIXED_DATE = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


@pytest.mark.asyncio
@pytest.mark.parametrize("sender_id", [ALLOWED_USER_ID, CUSTOMER_USER_ID])
async def test_extraction_is_submitted_only_after_context_save(sender_id: int) -> None:
    contexts = AsyncMock()
    chat = ContextChat(
        id=3, owner_id=ALLOWED_USER_ID, kind="business", chat_id=1001, title="Анна"
    )
    contexts.ensure_chat.return_value = chat
    extractor = Mock()
    extractor.submit.side_effect = lambda _: contexts.record.assert_awaited_once()
    router = create_business_router(
        allowed_user_id=ALLOWED_USER_ID, contexts=contexts, extractor=extractor
    )
    bot = make_bot(make_connection())
    incoming = make_message(sender_id=sender_id).model_copy(
        update={
            "text": "Сделаю аудит",
            "chat": Chat(id=1001, type="private", first_name="Анна", username="anna"),
        }
    )
    await find_handler(router, "business_message", "read_business_message")(
        incoming, bot
    )
    extractor.submit.assert_called_once_with(chat)
    saved = contexts.record.await_args.kwargs["message"]
    assert saved.sender_id == sender_id and not saved.is_business_bot
    assert "@anna" in contexts.ensure_chat.await_args.kwargs["title"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["save_failure", "business_bot", "no_author", "foreign_owner"]
)
async def test_extraction_does_not_run_for_unsaved_or_untrusted_messages(
    case: str,
) -> None:
    contexts, extractor = AsyncMock(), Mock()
    contexts.ensure_chat.return_value = ContextChat(
        id=3, owner_id=ALLOWED_USER_ID, kind="business", chat_id=1001, title="Анна"
    )
    if case == "save_failure":
        contexts.record.side_effect = RuntimeError("storage unavailable")
    router = create_business_router(
        allowed_user_id=ALLOWED_USER_ID, contexts=contexts, extractor=extractor
    )
    bot = make_bot(
        make_connection(owner_id=7 if case == "foreign_owner" else ALLOWED_USER_ID)
    )
    incoming = make_message(
        sender_id=None if case == "no_author" else ALLOWED_USER_ID,
        sender_business_bot=make_user(777, is_bot=True)
        if case == "business_bot"
        else None,
    ).model_copy(update={"caption": "Сделаю аудит"})
    await find_handler(router, "business_message", "read_business_message")(
        incoming, bot
    )
    extractor.submit.assert_not_called()


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


def make_user(user_id: int, *, is_bot: bool = False) -> User:
    return User(id=user_id, is_bot=is_bot, first_name="Test")


def make_connection(
    *,
    connection_id: str = CONNECTION_ID,
    owner_id: int = ALLOWED_USER_ID,
    is_enabled: bool = True,
    can_read_messages: bool | None = True,
    has_rights: bool = True,
) -> BusinessConnection:
    rights = (
        BusinessBotRights(can_read_messages=can_read_messages) if has_rights else None
    )
    return BusinessConnection(
        id=connection_id,
        user=make_user(owner_id),
        user_chat_id=owner_id,
        date=FIXED_DATE,
        is_enabled=is_enabled,
        rights=rights,
    )


def make_message(
    *,
    business_connection_id: str | None = CONNECTION_ID,
    sender_id: int | None = CUSTOMER_USER_ID,
    sender_business_bot: User | None = None,
    message_id: int = 17,
) -> Message:
    return Message(
        message_id=message_id,
        date=FIXED_DATE,
        chat=Chat(id=1001, type="private"),
        from_user=make_user(sender_id) if sender_id is not None else None,
        sender_business_bot=sender_business_bot,
        business_connection_id=business_connection_id,
    )


def make_bot(
    connection: BusinessConnection | None = None,
) -> Any:
    return cast(
        Any,
        type(
            "BotDouble",
            (),
            {
                "get_business_connection": AsyncMock(return_value=connection),
                "read_business_message": AsyncMock(),
            },
        )(),
    )


def telegram_api_error() -> TelegramAPIError:
    method = GetBusinessConnection(business_connection_id=CONNECTION_ID)
    return TelegramAPIError(method=method, message="Telegram unavailable")


def test_business_router_registers_all_supported_update_types() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)

    assert router.resolve_used_update_types() == [
        "business_connection",
        "business_message",
        "edited_business_message",
    ]
    business_handler = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    edited_handler = find_handler(
        router,
        "edited_business_message",
        "read_business_message",
    )
    assert edited_handler is business_handler


@pytest.mark.asyncio
async def test_business_message_marks_incoming_message_as_read_from_cache() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection = make_connection()
    bot = make_bot()

    await find_handler(router, "business_connection", "connection_changed")(connection)
    await find_handler(router, "business_message", "read_business_message")(
        make_message(),
        bot,
    )

    bot.get_business_connection.assert_not_awaited()
    bot.read_business_message.assert_awaited_once_with(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=17,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("observer", ["business_message", "edited_business_message"])
async def test_message_loads_connection_after_router_restart_and_marks_as_read(
    observer: str,
) -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection = make_connection()
    bot = make_bot(connection)

    await find_handler(router, observer, "read_business_message")(
        make_message(),
        bot,
    )

    bot.get_business_connection.assert_awaited_once_with(CONNECTION_ID)
    bot.read_business_message.assert_awaited_once_with(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=17,
    )


@pytest.mark.asyncio
async def test_concurrent_messages_load_uncached_connection_once() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()
    get_started = asyncio.Event()
    release_get = asyncio.Event()

    async def blocked_get(connection_id: str) -> BusinessConnection:
        assert connection_id == CONNECTION_ID
        get_started.set()
        await release_get.wait()
        return make_connection()

    bot.get_business_connection.side_effect = blocked_get
    first = asyncio.create_task(read_message(make_message(message_id=1), bot))
    await get_started.wait()
    second = asyncio.create_task(read_message(make_message(message_id=2), bot))
    await asyncio.sleep(0)
    release_get.set()

    await asyncio.gather(first, second)

    bot.get_business_connection.assert_awaited_once_with(CONNECTION_ID)
    assert bot.read_business_message.await_count == 2
    bot.read_business_message.assert_any_await(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=1,
    )
    bot.read_business_message.assert_any_await(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=2,
    )


@pytest.mark.asyncio
async def test_message_without_business_connection_id_is_ignored() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    bot = make_bot()

    await find_handler(router, "business_message", "read_business_message")(
        make_message(business_connection_id=None),
        bot,
    )

    bot.get_business_connection.assert_not_awaited()
    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connection",
    [
        make_connection(owner_id=777),
        make_connection(is_enabled=False),
    ],
    ids=["foreign-owner", "disabled-connection"],
)
async def test_message_from_unauthorized_connection_is_ignored(
    connection: BusinessConnection,
) -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    bot = make_bot()
    await find_handler(router, "business_connection", "connection_changed")(connection)

    await find_handler(router, "business_message", "read_business_message")(
        make_message(),
        bot,
    )

    bot.get_business_connection.assert_not_awaited()
    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_outgoing_message_from_connection_owner_is_ignored() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    bot = make_bot()
    await find_handler(router, "business_connection", "connection_changed")(
        make_connection()
    )

    await find_handler(router, "business_message", "read_business_message")(
        make_message(sender_id=ALLOWED_USER_ID),
        bot,
    )

    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_message_sent_by_business_bot_is_ignored() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    bot = make_bot()
    await find_handler(router, "business_connection", "connection_changed")(
        make_connection()
    )

    await find_handler(router, "business_message", "read_business_message")(
        make_message(sender_business_bot=make_user(555, is_bot=True)),
        bot,
    )

    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connection",
    [
        make_connection(has_rights=False),
        make_connection(can_read_messages=None),
        make_connection(can_read_messages=False),
    ],
    ids=["no-rights", "right-unspecified", "right-denied"],
)
async def test_message_without_explicit_read_permission_is_ignored(
    connection: BusinessConnection,
) -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    bot = make_bot()
    await find_handler(router, "business_connection", "connection_changed")(connection)

    await find_handler(router, "business_message", "read_business_message")(
        make_message(),
        bot,
    )

    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_business_connection_api_error_is_suppressed() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    bot = make_bot()
    bot.get_business_connection.side_effect = telegram_api_error()

    await find_handler(router, "business_message", "read_business_message")(
        make_message(),
        bot,
    )

    bot.get_business_connection.assert_awaited_once_with(CONNECTION_ID)
    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mismatched_loaded_connection_is_negatively_cached() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()
    mismatched_connection = make_connection(connection_id="other-connection")
    bot.get_business_connection.return_value = mismatched_connection

    await read_message(make_message(message_id=1), bot)
    await read_message(make_message(message_id=2), bot)

    bot.get_business_connection.assert_awaited_once_with(CONNECTION_ID)
    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_business_message_api_error_is_suppressed() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    bot = make_bot()
    bot.read_business_message.side_effect = telegram_api_error()
    await find_handler(router, "business_connection", "connection_changed")(
        make_connection()
    )

    await find_handler(router, "business_message", "read_business_message")(
        make_message(),
        bot,
    )

    bot.read_business_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_connection_change_replaces_cached_permissions() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection_changed = find_handler(
        router,
        "business_connection",
        "connection_changed",
    )
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()

    await connection_changed(make_connection(can_read_messages=True))
    await read_message(make_message(message_id=1), bot)
    await connection_changed(make_connection(can_read_messages=False))
    await read_message(make_message(message_id=2), bot)

    bot.get_business_connection.assert_not_awaited()
    bot.read_business_message.assert_awaited_once_with(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_connection",
    [
        make_connection(is_enabled=False),
        make_connection(owner_id=777),
    ],
    ids=["disabled", "foreign-owner"],
)
async def test_valid_connection_change_restores_rejected_connection(
    invalid_connection: BusinessConnection,
) -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection_changed = find_handler(
        router,
        "business_connection",
        "connection_changed",
    )
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()

    await connection_changed(make_connection())
    await connection_changed(invalid_connection)
    await read_message(make_message(message_id=1), bot)
    await connection_changed(make_connection())
    await read_message(make_message(message_id=2), bot)

    bot.get_business_connection.assert_not_awaited()
    bot.read_business_message.assert_awaited_once_with(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=2,
    )


@pytest.mark.asyncio
async def test_disabled_change_wins_over_stale_connection_load() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection_changed = find_handler(
        router,
        "business_connection",
        "connection_changed",
    )
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()
    get_started = asyncio.Event()
    release_get = asyncio.Event()

    async def blocked_get(connection_id: str) -> BusinessConnection:
        assert connection_id == CONNECTION_ID
        get_started.set()
        await release_get.wait()
        return make_connection()

    bot.get_business_connection.side_effect = blocked_get
    stale_message = asyncio.create_task(read_message(make_message(message_id=1), bot))
    await get_started.wait()

    await connection_changed(make_connection(is_enabled=False))
    release_get.set()
    await stale_message
    await read_message(make_message(message_id=2), bot)

    bot.get_business_connection.assert_awaited_once_with(CONNECTION_ID)
    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_invalid_connection_change_stays_rejected() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection_changed = find_handler(
        router,
        "business_connection",
        "connection_changed",
    )
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()
    disabled_connection = make_connection(is_enabled=False)

    await connection_changed(disabled_connection)
    await connection_changed(disabled_connection)
    await read_message(make_message(), bot)

    bot.get_business_connection.assert_not_awaited()
    bot.read_business_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_connection_cache_evicts_when_bounded_limit_is_reached() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection_changed = find_handler(
        router,
        "business_connection",
        "connection_changed",
    )
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()
    connection_ids = [
        f"rejected-{index}" for index in range(REJECTED_CONNECTION_LIMIT + 1)
    ]

    for connection_id in connection_ids:
        await connection_changed(
            make_connection(connection_id=connection_id, is_enabled=False)
        )

    async def load_active(connection_id: str) -> BusinessConnection:
        return make_connection(connection_id=connection_id)

    bot.get_business_connection.side_effect = load_active
    for index, connection_id in enumerate(connection_ids):
        await read_message(
            make_message(
                business_connection_id=connection_id,
                message_id=index + 1,
            ),
            bot,
        )

    assert bot.get_business_connection.await_count == 1
    assert bot.read_business_message.await_count == 1


@pytest.mark.asyncio
async def test_fresh_connection_change_wins_when_stale_load_fails() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection_changed = find_handler(
        router,
        "business_connection",
        "connection_changed",
    )
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()
    get_started = asyncio.Event()
    release_get = asyncio.Event()

    async def blocked_failing_get(connection_id: str) -> BusinessConnection:
        assert connection_id == CONNECTION_ID
        get_started.set()
        await release_get.wait()
        raise telegram_api_error()

    bot.get_business_connection.side_effect = blocked_failing_get
    stale_message = asyncio.create_task(read_message(make_message(message_id=1), bot))
    await get_started.wait()

    await connection_changed(make_connection())
    release_get.set()
    await stale_message

    bot.get_business_connection.assert_awaited_once_with(CONNECTION_ID)
    bot.read_business_message.assert_awaited_once_with(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=1,
    )


@pytest.mark.asyncio
async def test_unexpected_load_error_is_cleaned_up_before_retry() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()
    bot.get_business_connection.side_effect = [
        RuntimeError("unexpected failure"),
        make_connection(),
    ]

    with pytest.raises(RuntimeError, match="unexpected failure"):
        await read_message(make_message(message_id=1), bot)
    await read_message(make_message(message_id=2), bot)

    assert bot.get_business_connection.await_count == 2
    bot.read_business_message.assert_awaited_once_with(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=2,
    )


@pytest.mark.asyncio
async def test_stale_unexpected_load_error_preserves_fresh_connection() -> None:
    router = create_business_router(allowed_user_id=ALLOWED_USER_ID)
    connection_changed = find_handler(
        router,
        "business_connection",
        "connection_changed",
    )
    read_message = find_handler(
        router,
        "business_message",
        "read_business_message",
    )
    bot = make_bot()
    get_started = asyncio.Event()
    release_get = asyncio.Event()

    async def blocked_failing_get(connection_id: str) -> BusinessConnection:
        assert connection_id == CONNECTION_ID
        get_started.set()
        await release_get.wait()
        raise RuntimeError("stale failure")

    bot.get_business_connection.side_effect = blocked_failing_get
    stale_message = asyncio.create_task(read_message(make_message(message_id=1), bot))
    await get_started.wait()

    await connection_changed(make_connection())
    release_get.set()
    with pytest.raises(RuntimeError, match="stale failure"):
        await stale_message
    await read_message(make_message(message_id=2), bot)

    bot.get_business_connection.assert_awaited_once_with(CONNECTION_ID)
    bot.read_business_message.assert_awaited_once_with(
        business_connection_id=CONNECTION_ID,
        chat_id=1001,
        message_id=2,
    )
