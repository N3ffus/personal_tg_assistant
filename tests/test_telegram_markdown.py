from typing import Any, cast

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage
from aiogram.types import Message

from src.infrastructure.telegram.markdown import markdown_to_html
from src.infrastructure.telegram.replies import answer_text


@pytest.mark.parametrize(
    ("markdown", "html"),
    [
        ("**Интерстеллар** (2014)", "<b>Интерстеллар</b> (2014)"),
        ("*курсив* и _тоже_", "<i>курсив</i> и <i>тоже</i>"),
        ("***оба***", "<b><i>оба</i></b>"),
        ("~~было~~", "<s>было</s>"),
        ("`a < b`", "<code>a &lt; b</code>"),
        ("### Итог", "<b>Итог</b>"),
        ("- пункт\n  * вложенный", "• пункт\n  • вложенный"),
        ("> цитата\n> ещё", "<blockquote>цитата\nещё</blockquote>"),
        (
            "[сайт](https://example.com/?a=1&b=2)",
            '<a href="https://example.com/?a=1&amp;b=2">сайт</a>',
        ),
        ("```python\nif a < b:\n    pass\n```", "<pre>if a &lt; b:\n    pass</pre>"),
        ("```\nне закрыт", "<pre>не закрыт</pre>"),
    ],
)
def test_markdown_becomes_telegram_html(markdown: str, html: str) -> None:
    assert markdown_to_html(markdown) == html


@pytest.mark.parametrize(
    "text",
    [
        "Готово",
        "snake_case и __init__ остаются",
        "2 * 3 * 4 = 24",
        "1. Первый\n2. Второй",
        "#хэштег",
        "[не ссылка](javascript:alert)",
    ],
)
def test_plain_text_is_left_untouched(text: str) -> None:
    assert markdown_to_html(text) == text


def test_html_in_the_reply_is_escaped() -> None:
    assert markdown_to_html("<b>x</b> & **y**") == "&lt;b&gt;x&lt;/b&gt; &amp; <b>y</b>"


class FakeMessage:
    def __init__(self, *, reject_html: bool = False) -> None:
        self.answers: list[tuple[str, dict[str, Any]]] = []
        self._reject_html = reject_html

    async def answer(self, text: str, **kwargs: Any) -> None:
        if self._reject_html and kwargs.get("parse_mode") == "HTML":
            raise TelegramBadRequest(
                method=SendMessage(chat_id=1, text=text),
                message="Bad Request: can't parse entities",
            )
        self.answers.append((text, kwargs))


@pytest.mark.asyncio
async def test_formatted_reply_is_sent_as_html() -> None:
    message = FakeMessage()

    await answer_text(cast(Message, message), "Посмотри **Дюну**")

    assert message.answers == [("Посмотри <b>Дюну</b>", {"parse_mode": "HTML"})]


@pytest.mark.asyncio
async def test_rejected_html_falls_back_to_the_original_text() -> None:
    message = FakeMessage(reject_html=True)

    await answer_text(cast(Message, message), "Посмотри **Дюну**")

    assert message.answers == [("Посмотри **Дюну**", {})]


@pytest.mark.asyncio
async def test_code_block_split_between_messages_stays_a_code_block() -> None:
    message = FakeMessage()
    code = "\n".join(f"line {i}" for i in range(1000))

    await answer_text(cast(Message, message), f"```\n{code}\n```")

    assert len(message.answers) > 1
    for text, kwargs in message.answers:
        assert text.startswith("<pre>") and text.endswith("</pre>")
        assert kwargs == {"parse_mode": "HTML"}
