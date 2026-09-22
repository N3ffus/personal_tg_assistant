import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
)

from src.domain.assistant.replies import AssistantReply, ResultPage
from src.infrastructure.telegram.markdown import FENCE, markdown_to_html

logger = logging.getLogger(__name__)

TELEGRAM_TEXT_LIMIT = 4096
# Telegram clears «печатает…» after five seconds, so it is renewed sooner.
TYPING_RENEWAL_SECONDS = 4


@asynccontextmanager
async def typing(message: Message) -> AsyncIterator[None]:
    """Show «печатает…» while a reply is being prepared.

    A turn can take tens of seconds (a 49 s one on production), and silence
    reads as a lost message. The indicator is cosmetic: a failure to send it
    is logged and never reaches the reply.
    """
    bot = message.bot
    if bot is None:
        yield
        return

    async def keep_typing() -> None:
        while True:
            try:
                await bot.send_chat_action(
                    chat_id=message.chat.id, action=ChatAction.TYPING
                )
            except TelegramAPIError:
                logger.debug("Could not send the typing indicator")
            await asyncio.sleep(TYPING_RENEWAL_SECONDS)

    indicator = asyncio.create_task(keep_typing(), name="telegram-typing")
    try:
        yield
    finally:
        indicator.cancel()
        with suppress(asyncio.CancelledError):
            await indicator


async def answer_reply(message: Message, reply: str | AssistantReply) -> None:
    if isinstance(reply, str):
        await answer_text(message, reply)
        return
    if reply.text:
        await answer_text(message, reply.text)
    for page in reply.pages:
        await answer_page(message, page)
    for confirmation in reply.confirmations:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🗑 Удалить",
                        callback_data=f"delyes:{confirmation.operation_id}",
                    ),
                    InlineKeyboardButton(
                        text="Отмена",
                        callback_data=f"delno:{confirmation.operation_id}",
                    ),
                ]
            ]
        )
        await answer_text(message, confirmation.text, reply_markup=keyboard)


def page_keyboard(page: ResultPage) -> InlineKeyboardMarkup | None:
    # A single-page result carries no buttons at all.
    if not page.buttons:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=b.text, callback_data=b.callback_data)
                for b in row
            ]
            for row in page.buttons
        ]
    )


async def answer_page(
    message: Message, page: ResultPage, *, edit: bool = False
) -> None:
    send = message.edit_text if edit else message.answer
    await send(
        page.text,
        parse_mode="HTML",
        reply_markup=page_keyboard(page),
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )


async def answer_text(
    message: Message,
    text: str,
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    chunks = _reopen_code_fences(split_text(text))
    for chunk in chunks[:-1]:
        await _answer_markdown(message, chunk)
    if reply_markup is None:
        await _answer_markdown(message, chunks[-1])
    else:
        await _answer_markdown(message, chunks[-1], reply_markup=reply_markup)


async def _answer_markdown(message: Message, text: str, **kwargs: Any) -> None:
    html = markdown_to_html(text)
    if html == text:
        await message.answer(text, **kwargs)
        return
    try:
        await message.answer(html, parse_mode="HTML", **kwargs)
    except TelegramBadRequest as error:
        # The reply must still arrive, even without its formatting.
        if "can't parse entities" not in error.message.lower():
            raise
        logger.warning("telegram.markdown.rejected")
        await message.answer(text, **kwargs)


def _reopen_code_fences(chunks: list[str]) -> list[str]:
    # A code block cut between two messages is closed and reopened across the cut.
    result: list[str] = []
    carried = False
    for chunk in chunks:
        if carried:
            chunk = "```\n" + chunk
        fences = sum(1 for line in chunk.split("\n") if FENCE.match(line))
        carried = fences % 2 == 1
        result.append(chunk + "\n```" if carried else chunk)
    return result


def split_text(text: str, *, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not text:
        raise ValueError("Telegram message must not be empty")

    chunks: list[str] = []
    remainder = text
    while remainder:
        units = 0
        end = 0
        # Telegram counts UTF-16 units; emoji can occupy two units each.
        for char in remainder[:limit]:
            width = 2 if ord(char) > 0xFFFF else 1
            if units + width > limit:
                break
            units += width
            end += 1
        if end == 0:
            raise ValueError("limit cannot fit a single character")
        if end == len(remainder):
            chunks.append(remainder)
            break
        newline = remainder.rfind("\n", 0, end)
        split_at = newline + 1 if newline > 0 else end
        chunks.append(remainder[:split_at])
        remainder = remainder[split_at:]
    return chunks
