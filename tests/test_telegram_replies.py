from typing import Any, cast

import pytest
from aiogram.types import InlineKeyboardMarkup, Message

from src.domain.assistant.replies import AssistantReply, Confirmation
from src.infrastructure.telegram.replies import answer_reply, answer_text, split_text


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
