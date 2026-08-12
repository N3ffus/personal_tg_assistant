import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from src.application.use_cases.process_message import (
    ProcessMessageUseCase,
)

logger = logging.getLogger(__name__)


def create_router(
    *,
    process_message: ProcessMessageUseCase,
    timezone: str,
) -> Router:
    router = Router(
        name=__name__,
    )

    @router.message(CommandStart())
    async def start_handler(
        message: Message,
    ) -> None:
        await message.answer(
            "Привет! Я ИИ-помощник.\n\n"
            "Можешь написать, например:\n"
            "• Добавь задачу купить продукты\n"
            "• Завтра в 15:00 стоматолог\n"
            "• Запомни, что я люблю Python"
        )

    @router.message(F.text)
    async def text_handler(
        message: Message,
    ) -> None:
        text = message.text

        if not text:
            return

        try:
            now = datetime.now(
                ZoneInfo(timezone),
            )

            response = await process_message.execute(
                text=text,
                now=now,
                timezone=timezone,
            )

        except Exception:
            logger.exception(
                "Failed to process Telegram message",
            )

            await message.answer("Не получилось обработать сообщение.")
            return

        await message.answer(response)

    return router
