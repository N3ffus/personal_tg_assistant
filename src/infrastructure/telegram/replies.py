from aiogram.types import InlineKeyboardMarkup, Message

TELEGRAM_TEXT_LIMIT = 4096


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
    while len(remainder) > limit:
        newline = remainder.rfind("\n", 0, limit)
        split_at = newline + 1 if newline > 0 else limit
        chunks.append(remainder[:split_at])
        remainder = remainder[split_at:]
    chunks.append(remainder)
    return chunks
