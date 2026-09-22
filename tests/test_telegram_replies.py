from typing import Any, cast

import pytest
from aiogram.types import InlineKeyboardMarkup, Message

from src.domain.assistant.replies import AssistantReply, Confirmation
from src.infrastructure.telegram.replies import (
    answer_reply,
    answer_text,
    split_text,
    typing,
)


class FakeMessage:
    def __init__(self) -> None:
        self.answers: list[tuple[str, dict[str, Any]]] = []

    async def answer(self, text: str, **kwargs: Any) -> None:
        self.answers.append((text, kwargs))


@pytest.mark.asyncio
async def test_long_confirmation_is_fully_shown_before_its_buttons() -> None:
    message = FakeMessage()
    listing = "\n".join(f"• ENG-{i}: Задача {i}" for i in range(600))
    await answer_reply(
        cast(Message, message),
        AssistantReply(
            text="Найдено",
            confirmations=(Confirmation(text=listing, operation_id="token"),),
        ),
    )
    assert message.answers[0][0] == "Найдено"
    assert "".join(text for text, _ in message.answers[1:]) == listing
    assert all(len(text) <= 4096 for text, _ in message.answers)
    assert all("reply_markup" not in args for _, args in message.answers[:-1])
    assert "reply_markup" in message.answers[-1][1]


def test_split_text_prefers_line_boundaries_and_preserves_content() -> None:
    text = "first line\nsecond line\nlast"

    chunks = split_text(text, limit=14)

    assert chunks == ["first line\n", "second line\n", "last"]
    assert "".join(chunks) == text
    assert all(len(chunk) <= 14 for chunk in chunks)


def test_split_text_hard_splits_a_single_long_line() -> None:
    assert split_text("abcdefghij", limit=4) == ["abcd", "efgh", "ij"]


def test_split_text_preserves_emoji_at_utf16_boundaries() -> None:
    text = "a😀\nb😀😀c"
    chunks = split_text(text, limit=4)
    assert "".join(chunks) == text
    assert chunks[0] == "a😀\n"
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 4 for chunk in chunks)
    with pytest.raises(ValueError, match="single character"):
        split_text("😀", limit=1)


@pytest.mark.parametrize(("text", "limit"), [("", 4096), ("text", 0)])
def test_split_text_rejects_invalid_message(text: str, limit: int) -> None:
    with pytest.raises(ValueError):
        split_text(text, limit=limit)


@pytest.mark.asyncio
async def test_answer_text_attaches_keyboard_only_to_final_chunk() -> None:
    message = FakeMessage()
    keyboard = InlineKeyboardMarkup(inline_keyboard=[])

    await answer_text(
        cast(Message, message),
        "a" * 4097,
        reply_markup=keyboard,
    )

    assert message.answers == [
        ("a" * 4096, {}),
        ("a", {"reply_markup": keyboard}),
    ]


class TypingBot:
    def __init__(self, *, fail: bool = False) -> None:
        self.actions: list[tuple[int, str]] = []
        self.fail = fail

    async def send_chat_action(self, *, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))
        if self.fail:
            from aiogram.exceptions import TelegramNetworkError
            from aiogram.methods import SendChatAction

            raise TelegramNetworkError(
                method=SendChatAction(chat_id=chat_id, action=action),
                message="timeout",
            )


def typing_message(bot: object | None) -> Message:
    from types import SimpleNamespace

    return cast(Message, SimpleNamespace(bot=bot, chat=SimpleNamespace(id=7)))


@pytest.mark.asyncio
async def test_typing_is_shown_while_the_reply_is_prepared() -> None:
    """A 49 s turn in silence reads as a lost message."""
    import asyncio

    bot = TypingBot()
    async with typing(typing_message(bot)):
        await asyncio.sleep(0)

    assert bot.actions == [(7, "typing")]


@pytest.mark.asyncio
async def test_a_failing_typing_indicator_never_reaches_the_reply() -> None:
    import asyncio

    bot = TypingBot(fail=True)
    async with typing(typing_message(bot)):
        await asyncio.sleep(0)
        result = "ответ"

    assert result == "ответ"
    assert bot.actions == [(7, "typing")]


@pytest.mark.asyncio
async def test_typing_without_a_bound_bot_is_a_no_op() -> None:
    async with typing(typing_message(None)):
        pass
