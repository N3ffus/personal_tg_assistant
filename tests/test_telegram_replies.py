from typing import Any, cast

import pytest
from aiogram.types import InlineKeyboardMarkup, Message

from src.infrastructure.telegram.replies import answer_text, split_text


class FakeMessage:
    def __init__(self) -> None:
        self.answers: list[tuple[str, dict[str, Any]]] = []

    async def answer(self, text: str, **kwargs: Any) -> None:
        self.answers.append((text, kwargs))


def test_split_text_prefers_line_boundaries_and_preserves_content() -> None:
    text = "first line\nsecond line\nlast"

    chunks = split_text(text, limit=14)

    assert chunks == ["first line\n", "second line\n", "last"]
    assert "".join(chunks) == text
    assert all(len(chunk) <= 14 for chunk in chunks)


def test_split_text_hard_splits_a_single_long_line() -> None:
    assert split_text("abcdefghij", limit=4) == ["abcd", "efgh", "ij"]


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
