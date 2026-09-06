import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Literal, cast

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.application.services.context import ContextService
from src.domain.assistant.context import MAX_CONTEXT_CHARS, ContextChat
from src.infrastructure.telegram.replies import answer_text, split_text

logger = logging.getLogger(__name__)
MENU_TTL = 15 * 60
MENU_LIMIT = 32
PAGE_SIZE = 8


@dataclass
class ContextMenu:
    operation: Literal["clear", "compact", "history"]
    chat_id: int
    chats: list[ContextChat]
    created_at: float = field(default_factory=time.monotonic)
    selected: set[int] = field(default_factory=set)
    page: int = 0


def menu_view(menu: ContextMenu, token: str) -> tuple[str, InlineKeyboardMarkup]:
    def button(text: str, action: str, value: int = 0) -> InlineKeyboardButton:
        return InlineKeyboardButton(
            text=text, callback_data=f"ctx:{token}:{action}:{value}"
        )

    labels = {
        "clear": "Очистка контекста",
        "compact": "Суммаризация контекста",
        "history": "История контекста",
    }
    pages = max(1, (len(menu.chats) + PAGE_SIZE - 1) // PAGE_SIZE)
    rows = []
    for chat in menu.chats[menu.page * PAGE_SIZE : (menu.page + 1) * PAGE_SIZE]:
        label = f"{'🤖' if chat.kind == 'bot' else '👤'} {chat.title}"
        if menu.operation != "history":
            label = f"{'✅' if chat.id in menu.selected else '⬜'} {label}"
        rows.append(
            [
                button(
                    label[:60],
                    "show" if menu.operation == "history" else "toggle",
                    chat.id,
                )
            ]
        )
    navigation = []
    if menu.page:
        navigation.append(button("← Назад", "page", menu.page - 1))
    if menu.page + 1 < pages:
        navigation.append(button("Далее →", "page", menu.page + 1))
    if navigation:
        rows.append(navigation)
    text = f"{labels[menu.operation]} · страница {menu.page + 1}/{pages}\n\n"
    if menu.operation == "history":
        text += "Выберите чат для просмотра или скачайте контекст всех чатов."
        rows.append([button("📄 Все чаты — файл", "export")])
    else:
        text += "Отметьте нужные чаты и нажмите кнопку действия.\n"
        text += f"Выбрано: {len(menu.selected)} из {len(menu.chats)}.\n\n"
        text += (
            "Удалятся сообщения и резюме из памяти бота. Переписка в Telegram останется."
            if menu.operation == "clear"
            else "Сообщения в памяти заменятся кратким резюме. Переписка в Telegram останется."
        )
        all_selected = len(menu.selected) == len(menu.chats)
        rows.append(
            [button("Снять выделение" if all_selected else "Выбрать все чаты", "all")]
        )
        rows.append(
            [
                button(
                    "🗑 Очистить выбранные"
                    if menu.operation == "clear"
                    else "📝 Суммаризовать выбранные",
                    "run",
                )
            ]
        )
    rows.append([button("Закрыть", "close")])
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def create_context_router(*, contexts: ContextService, allowed_user_id: int) -> Router:
    router = Router(name=__name__)
    menus: dict[str, ContextMenu] = {}

    @router.message(Command("clear", "compact", "history"))
    async def context_command(message: Message, command: CommandObject) -> None:
        if message.from_user is None or message.from_user.id != allowed_user_id:
            return
        if message.chat.type != "private":
            await message.answer(
                "Управление контекстом доступно в личном чате с ботом."
            )
            return
        try:
            await contexts.ensure_chat(
                owner_id=allowed_user_id,
                kind="bot",
                chat_id=message.chat.id,
                title="Чат с ботом",
            )
            chats = await contexts.list_chats(owner_id=allowed_user_id)
            operation = command.command
            if operation not in {"clear", "compact", "history"}:
                return
            menu = ContextMenu(
                operation=cast(Literal["clear", "compact", "history"], operation),
                chat_id=message.chat.id,
                chats=chats,
            )
            for key in list(menus):
                if time.monotonic() - menus[key].created_at >= MENU_TTL:
                    menus.pop(key)
            if len(menus) >= MENU_LIMIT:
                menus.pop(next(iter(menus)))
            token = secrets.token_hex(8)
            menus[token] = menu
            text, keyboard = menu_view(menu, token)
            await message.answer(text, reply_markup=keyboard, parse_mode=None)
        except Exception:
            logger.exception("Failed to open context menu")
            await message.answer(
                "Не удалось открыть контекст. Попробуйте команду ещё раз."
            )

    async def show_history(
        message: Message, menu: ContextMenu, token: str, context_id: int, page: int
    ) -> None:
        chat = next(chat for chat in menu.chats if chat.id == context_id)
        context = await contexts.read(owner_id=allowed_user_id, context_id=context_id)
        content = context.render() or "Контекст пуст."
        # Leave room for the header even when each character is a UTF-16 surrogate pair.
        chunks = split_text(content, limit=1800)
        page = max(0, min(page, len(chunks) - 1))
        rows = []
        navigation = []
        for label, target in [("← Ранее", page - 1), ("Далее →", page + 1)]:
            if 0 <= target < len(chunks):
                navigation.append(
                    InlineKeyboardButton(
                        text=label,
                        callback_data=f"ctx:{token}:history:{context_id}:{target}",
                    )
                )
        if navigation:
            rows.append(navigation)
        rows.append(
            [
                InlineKeyboardButton(
                    text="К списку чатов", callback_data=f"ctx:{token}:page:{menu.page}"
                )
            ]
        )
        text = f"{chat.title}\nСообщений: {len(context.messages)}, символов: {len(context.render())}/{MAX_CONTEXT_CHARS}\nСтраница {page + 1}/{len(chunks)}\n\n{chunks[page]}"
        await message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
            parse_mode=None,
        )

    @router.callback_query(F.data.startswith("ctx:"))
    async def context_callback(callback: CallbackQuery) -> None:
        if callback.from_user.id != allowed_user_id or not isinstance(
            callback.message, Message
        ):
            await callback.answer()
            return
        message = callback.message
        parts = (callback.data or "").split(":")
        if len(parts) not in {4, 5}:
            await callback.answer("Кнопка недействительна.")
            return
        _, token, action, raw_value, *extra = parts
        menu = menus.get(token)
        if (
            menu is None
            or time.monotonic() - menu.created_at >= MENU_TTL
            or message.chat.type != "private"
            or message.chat.id != menu.chat_id
        ):
            await callback.answer("Меню устарело. Откройте команду заново.")
            return
        try:
            value = int(raw_value)
            page = int(extra[0]) if extra else 0
        except ValueError:
            await callback.answer("Кнопка недействительна.")
            return
        chat_ids = {chat.id for chat in menu.chats}
        if action in {"toggle", "show", "history"} and value not in chat_ids:
            await callback.answer("Чат недоступен.")
            return
        if action == "run" and not menu.selected:
            await callback.answer("Сначала выберите чаты.")
            return
        await callback.answer()
        try:
            if action == "close":
                menus.pop(token, None)
                await message.edit_text("Меню контекста закрыто.", reply_markup=None)
                return
            if action in {"show", "history"} and menu.operation == "history":
                await show_history(message, menu, token, value, page)
                return
            if action == "export" and menu.operation == "history":
                sections = []
                for chat in menu.chats:
                    context = await contexts.read(
                        owner_id=allowed_user_id, context_id=chat.id
                    )
                    sections.append(
                        f"=== {chat.title} ===\n{context.render() or 'Контекст пуст.'}"
                    )
                await message.answer_document(
                    BufferedInputFile(
                        "\n\n".join(sections).encode(), filename="chat-contexts.txt"
                    ),
                    caption="Текущий сохранённый контекст всех чатов.",
                )
                return
            if action == "run" and menu.operation in {"clear", "compact"}:
                # Consume before the first await: double clicks cannot run the job twice.
                if menus.pop(token, None) is None:
                    return
                selected = [chat for chat in menu.chats if chat.id in menu.selected]
                await message.edit_text(
                    f"Обрабатываю чаты: {len(selected)}…", reply_markup=None
                )
                results = []
                for chat in selected:
                    try:
                        if menu.operation == "clear":
                            await contexts.clear(
                                owner_id=allowed_user_id, context_id=chat.id
                            )
                            result = "контекст очищен"
                        else:
                            compacted = await contexts.compact(
                                owner_id=allowed_user_id, context_id=chat.id
                            )
                            result = (
                                "контекст суммаризован"
                                if compacted
                                else "нет новых сообщений для сжатия"
                            )
                        results.append(f"✅ {chat.title}: {result}")
                    except Exception:
                        logger.exception(
                            "Failed context operation for chat %s", chat.id
                        )
                        results.append(
                            f"❌ {chat.title}: не удалось обработать; контекст сохранён. Попробуйте снова."
                        )
                await answer_text(message, "\n".join(results))
                return
            if action == "page":
                menu.page = max(0, min(value, (len(menu.chats) - 1) // PAGE_SIZE))
            elif action == "toggle" and menu.operation != "history":
                menu.selected.symmetric_difference_update({value})
            elif action == "all" and menu.operation != "history":
                menu.selected = set() if menu.selected == chat_ids else chat_ids
            else:
                return
            text, keyboard = menu_view(menu, token)
            await message.edit_text(text, reply_markup=keyboard, parse_mode=None)
        except Exception:
            logger.exception("Failed to handle context menu")
            await message.answer("Не удалось обработать меню. Откройте команду заново.")

    return router
