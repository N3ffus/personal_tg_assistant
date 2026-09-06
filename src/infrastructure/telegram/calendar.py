import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.application.ports.calendar import (
    CalendarClient,
    CalendarError,
    CalendarEventNotFoundError,
    CalendarNotConnectedError,
)
from src.application.services.action_executor import ActionExecutor
from src.domain.calendar.models import CalendarEvent
from src.infrastructure.calendar.oauth import GoogleOAuthService
from src.infrastructure.calendar.storage import CalendarStorage
from src.infrastructure.telegram.replies import answer_text

logger = logging.getLogger(__name__)


def create_calendar_router(
    *,
    timezone: str,
    allowed_user_id: int,
    calendar: CalendarClient,
    oauth: GoogleOAuthService,
    storage: CalendarStorage,
) -> Router:
    router = Router(name=f"{__name__}.router")

    @router.message(Command("calendar_connect"))
    async def connect(message: Message) -> None:
        user_id = _allowed_message_user_id(message, allowed_user_id)
        if user_id is None:
            return
        try:
            url = await oauth.authorization_url(user_id=user_id)
        except Exception:
            logger.exception("Failed to create Google OAuth URL")
            await message.answer("Не удалось начать подключение календаря.")
            return
        await message.answer(f"Подключите Google Calendar: {url}")

    @router.message(Command("calendar_disconnect"))
    async def disconnect(message: Message) -> None:
        user_id = _allowed_message_user_id(message, allowed_user_id)
        if user_id is None:
            return
        try:
            await calendar.disconnect(user_id=user_id)
        except CalendarError:
            await message.answer("Не удалось отключить Google Calendar.")
            return
        await message.answer("Google Calendar отключён.")

    @router.message(Command("calendar"))
    async def list_events(message: Message) -> None:
        user_id = _allowed_message_user_id(message, allowed_user_id)
        if user_id is None:
            return
        try:
            events = await calendar.list_events(
                user_id=user_id,
                now=datetime.now(ZoneInfo(timezone)),
            )
        except CalendarNotConnectedError:
            await message.answer("Подключите календарь командой /calendar_connect.")
            return
        except CalendarError:
            await message.answer("Google Calendar временно недоступен.")
            return
        try:
            selection_ids = (
                await storage.create_operations(
                    user_id=user_id,
                    kind="select",
                    payloads=[{"event_id": event.event_id} for event in events],
                )
                if events
                else []
            )
        except Exception:
            logger.exception("Failed to store calendar event selections")
            await message.answer("Не удалось подготовить действия с событиями.")
            return
        await answer_text(
            message,
            ActionExecutor.format_events(events, timezone=ZoneInfo(timezone)),
            reply_markup=(
                _event_keyboard(events, selection_ids) if selection_ids else None
            ),
        )

    @router.callback_query(F.data.startswith("caledit:"))
    async def request_update(callback: CallbackQuery) -> None:
        if callback.from_user.id != allowed_user_id or callback.data is None:
            return
        try:
            selection = await storage.consume_operation(
                operation_id=callback.data.removeprefix("caledit:"),
                user_id=callback.from_user.id,
            )
            event_id = _operation_event_id(selection, expected_kind="select")
            if event_id is not None:
                await storage.create_operation(
                    user_id=callback.from_user.id,
                    kind="update",
                    payload={"event_id": event_id},
                )
        except Exception:
            logger.exception("Failed to prepare calendar event update")
            await callback.answer()
            await _answer_callback(callback, "Не удалось выбрать событие.")
            return
        await callback.answer()
        if event_id is None:
            await _answer_callback(callback, "Кнопка недействительна или устарела.")
            return
        if callback.message:
            await callback.message.answer(
                "Напишите новое название и время события, например: "
                "«Стоматолог завтра в 16:00»."
            )

    @router.callback_query(F.data.startswith("caldel:"))
    async def request_delete(callback: CallbackQuery) -> None:
        if callback.from_user.id != allowed_user_id or callback.data is None:
            return
        try:
            selection = await storage.consume_operation(
                operation_id=callback.data.removeprefix("caldel:"),
                user_id=callback.from_user.id,
            )
            event_id = _operation_event_id(selection, expected_kind="select")
            if event_id is not None:
                operation_id = await storage.create_operation(
                    user_id=callback.from_user.id,
                    kind="delete",
                    payload={"event_id": event_id},
                )
        except Exception:
            logger.exception("Failed to prepare calendar event deletion")
            await callback.answer()
            await _answer_callback(callback, "Не удалось выбрать событие.")
            return
        await callback.answer()
        if event_id is None:
            await _answer_callback(callback, "Кнопка недействительна или устарела.")
            return
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Да, удалить",
                        callback_data=f"calyes:{operation_id}",
                    ),
                    InlineKeyboardButton(
                        text="Отмена",
                        callback_data=f"calno:{operation_id}",
                    ),
                ]
            ]
        )
        if callback.message:
            await callback.message.answer("Удалить это событие?", reply_markup=keyboard)

    @router.callback_query(F.data.startswith("calyes:"))
    async def confirm_delete(callback: CallbackQuery) -> None:
        if callback.from_user.id != allowed_user_id or callback.data is None:
            return
        try:
            operation = await storage.consume_operation(
                operation_id=callback.data.removeprefix("calyes:"),
                user_id=callback.from_user.id,
            )
        except Exception:
            logger.exception("Failed to consume calendar deletion confirmation")
            await callback.answer()
            await _answer_callback(callback, "Не удалось проверить подтверждение.")
            return
        await callback.answer()
        event_id = _operation_event_id(operation, expected_kind="delete")
        if event_id is None:
            await _answer_callback(
                callback, "Подтверждение недействительно или устарело."
            )
            return
        try:
            await calendar.delete_event(
                user_id=callback.from_user.id,
                event_id=event_id,
            )
        except CalendarEventNotFoundError:
            text = "Событие уже удалено или не найдено."
        except CalendarError:
            text = "Не удалось удалить событие. Попробуйте позже."
        else:
            text = "Событие удалено."
        await _answer_callback(callback, text)

    @router.callback_query(F.data.startswith("calno:"))
    async def cancel_delete(callback: CallbackQuery) -> None:
        if callback.from_user.id != allowed_user_id or callback.data is None:
            return
        try:
            operation = await storage.consume_operation(
                operation_id=callback.data.removeprefix("calno:"),
                user_id=callback.from_user.id,
            )
        except Exception:
            logger.exception("Failed to cancel calendar event deletion")
            await callback.answer("Не удалось отменить удаление")
            return
        if _operation_event_id(operation, expected_kind="delete") is None:
            await callback.answer("Подтверждение недействительно или устарело")
            return
        await callback.answer("Удаление отменено")

    return router


def _event_keyboard(
    events: list[CalendarEvent], selection_ids: list[str]
) -> InlineKeyboardMarkup:
    rows = []
    for event, selection_id in zip(events, selection_ids, strict=True):
        row = []
        if not event.all_day:
            row.append(
                InlineKeyboardButton(
                    text=f"Изменить: {event.title}"[:64],
                    callback_data=f"caledit:{selection_id}",
                )
            )
        row.append(
            InlineKeyboardButton(
                text=f"Удалить: {event.title}"[:64],
                callback_data=f"caldel:{selection_id}",
            )
        )
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _allowed_message_user_id(message: Message, allowed_user_id: int) -> int | None:
    if message.from_user is None or message.from_user.id != allowed_user_id:
        return None
    return message.from_user.id


def _operation_event_id(
    operation: tuple[str, dict[str, object]] | None,
    *,
    expected_kind: str,
) -> str | None:
    if operation is None or operation[0] != expected_kind:
        return None
    event_id = operation[1].get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return None
    return event_id


async def _answer_callback(callback: CallbackQuery, text: str) -> None:
    if callback.message:
        await callback.message.answer(text)
