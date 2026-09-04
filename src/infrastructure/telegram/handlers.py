import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from src.application.ports.tasks import TaskCreationUncertainError, TaskTrackerError
from src.application.use_cases.process_message import (
    ProcessMessageUseCase,
)
from src.infrastructure.telegram.replies import answer_text

logger = logging.getLogger(__name__)


def create_router(
    *,
    process_message: ProcessMessageUseCase,
    timezone: str,
    allowed_user_id: int,
) -> Router:
    router = Router(
        name=__name__,
    )

    @router.message(CommandStart())
    async def start_handler(
        message: Message,
    ) -> None:
        if message.from_user is None or message.from_user.id != allowed_user_id:
            return

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

        if message.from_user is None:
            return

        if message.from_user.id != allowed_user_id:
            return

        try:
            now = datetime.now(
                ZoneInfo(timezone),
            )

            response = await process_message.execute(
                text=text,
                now=now,
                timezone=timezone,
                user_id=message.from_user.id,
            )

        except TaskCreationUncertainError:
            logger.exception(
                "Linear task creation result is uncertain",
            )

            await message.answer(
                "Linear мог успеть создать задачу. "
                "Проверьте список задач перед повтором."
            )
            return

        except TaskTrackerError:
            logger.exception(
                "Failed to create Linear task",
            )

            await message.answer(
                "Не удалось создать задачу в Linear. Попробуйте позже."
            )
            return

        except Exception:
            logger.exception(
                "Failed to process Telegram message",
            )

            await message.answer("Не получилось обработать сообщение.")
            return

        await answer_text(message, response)

    return router
