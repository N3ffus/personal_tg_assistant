from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.domain.assistant.replies import AssistantReply

TELEGRAM_TEXT_LIMIT = 4096


async def answer_reply(message: Message, reply: str | AssistantReply) -> None:
    if isinstance(reply, str):
        await answer_text(message, reply)
        return
    if reply.text:
        await answer_text(message, reply.text)
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


async def answer_text(
    message: Message,
    text: str,
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    chunks = split_text(text)
    for chunk in chunks[:-1]:
        await message.answer(chunk)
    if reply_markup is None:
        await message.answer(chunks[-1])
    else:
        await message.answer(chunks[-1], reply_markup=reply_markup)


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
