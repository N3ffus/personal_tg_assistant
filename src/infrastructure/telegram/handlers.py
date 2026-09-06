import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, Message

from src.application.ports.calendar import CalendarError, CalendarNotConnectedError
from src.application.ports.tasks import TaskCreationUncertainError, TaskTrackerError
from src.application.services.context import ContextService
from src.application.use_cases.process_message import (
    ProcessMessageUseCase,
)
from src.domain.assistant.replies import ResultPage
from src.domain.assistant.retrieval import RetrievalLimitError
from src.infrastructure.telegram.replies import answer_page, answer_reply, answer_text

logger = logging.getLogger(__name__)


def create_router(
    *,
    process_message: ProcessMessageUseCase,
    timezone: str,
    allowed_user_id: int,
    contexts: ContextService | None = None,
) -> Router:
    router = Router(
        name=__name__,
    )

    @router.callback_query(F.data.startswith("browse:"))
    async def browse_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        if (
            callback.from_user.id != allowed_user_id
            or not callback.data
            or not isinstance(callback.message, Message)
        ):
            return
        try:
            response = await process_message.browse(
                user_id=callback.from_user.id,
                data=callback.data,
                now=datetime.now(ZoneInfo(timezone)),
            )
        except CalendarNotConnectedError:
            response = "Подключите календарь командой /calendar_connect."
        except (RetrievalLimitError, TimeoutError):
            response = (
                "Слишком большая выборка или долгий поиск. Уточните период или фильтры."
            )
        except (CalendarError, TaskTrackerError):
            response = (
                "Не удалось обновить список. Проверьте доступ и попробуйте позже."
            )
        except Exception:
            logger.exception("Failed to navigate retrieval results")
            response = "Не удалось открыть страницу. Повторите запрос."
        if isinstance(response, ResultPage):
            try:
                # Calendar deletion confirmation is a separate message.
                await answer_page(
                    callback.message, response, edit=":delete:" not in callback.data
                )
            except TelegramBadRequest as error:
                if "message is not modified" not in error.message.lower():
                    logger.warning("Could not edit retrieval page")
                    await answer_page(callback.message, response)
        else:
            await answer_text(callback.message, response)

    @router.callback_query(F.data.startswith("delyes:") | F.data.startswith("delno:"))
    async def deletion_callback(callback: CallbackQuery) -> None:
        if callback.from_user.id != allowed_user_id or callback.data is None:
            await callback.answer()
            return
        # Acknowledge before potentially slow bulk API requests.
        await callback.answer()
        prefix, operation_id = callback.data.split(":", 1)
        try:
            response = await process_message.resolve_deletion(
                user_id=callback.from_user.id,
                operation_id=operation_id,
                confirm=prefix == "delyes",
            )
        except Exception:
            logger.exception("Failed to resolve deletion confirmation")
            response = "Не удалось обработать подтверждение. Проверьте список и повторите запрос удаления."
        if isinstance(callback.message, Message):
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramAPIError:
                logger.warning("Could not remove deletion keyboard")
        if callback.message:
            await answer_text(callback.message, response)  # type: ignore[arg-type]

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
            "• Запомни, что я люблю Python\n\n"
            "Задачи и календарь:\n"
            "/tasks — задачи Linear с фильтрами и страницами\n"
            "/calendar — события с фильтрами и страницами\n"
            "/agenda — задачи и события вместе\n"
            "По умолчанию по 10 записей, листать можно кнопками.\n"
            "Например: «События за 2026 год», «Открытые задачи со словом ремонт», "
            "«Задачи по теме отпуска по сроку».\n"
            "Фильтры можно дописать к команде: /tasks выполненные за год.\n\n"
            "Контекст чатов:\n"
            "/history — посмотреть\n"
            "/compact — суммаризовать\n"
            "/clear — очистить\n"
            "В каждой команде можно выбрать чаты или все сразу.\n\n"
            "Все команды доступны по кнопке «Меню» рядом с полем сообщения."
        )

    @router.message(Command("tasks", "agenda", "calendar"))
    async def view_handler(message: Message, command: CommandObject) -> None:
        await text_handler(message, command=command)

    @router.message(F.text.startswith("/"))
    async def unknown_command_handler(message: Message) -> None:
        if message.from_user is None or message.from_user.id != allowed_user_id:
            return
        await message.answer("Неизвестная команда. Выберите действие в меню бота.")

    @router.message(F.text)
    async def text_handler(
        message: Message,
        command: CommandObject | None = None,
    ) -> None:
        text = f"/{command.command}" if command else message.text
        if command is not None and command.args:
            subject = {
                "tasks": "задачи Linear",
                "calendar": "события Google Calendar",
                "agenda": "задачи Linear и события Google Calendar",
            }[command.command]
            text = f"Покажи {subject}: {command.args}"

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
                **(
                    {"chat_id": message.chat.id, "message_id": message.message_id}
                    if contexts is not None
                    else {}
                ),
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

        await answer_reply(message, response)

    return router
