import logging

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BotCommand, MenuButtonCommands

logger = logging.getLogger(__name__)

BOT_COMMANDS = (
    BotCommand(command="start", description="Помощь и возможности бота"),
    BotCommand(command="history", description="Посмотреть контекст чатов"),
    BotCommand(command="compact", description="Суммаризовать выбранные чаты"),
    BotCommand(command="clear", description="Очистить контекст выбранных чатов"),
    BotCommand(command="tasks", description="Задачи Linear со статусами и ссылками"),
    BotCommand(command="agenda", description="Задачи Linear и события календаря"),
    BotCommand(command="calendar", description="Ближайшие события календаря"),
    BotCommand(command="calendar_connect", description="Подключить Google Calendar"),
    BotCommand(command="calendar_disconnect", description="Отключить Google Calendar"),
)


async def configure_commands(bot: Bot) -> None:
    try:
        await bot.set_my_commands(list(BOT_COMMANDS), request_timeout=10)
        await bot.set_chat_menu_button(
            menu_button=MenuButtonCommands(), request_timeout=10
        )
    except TelegramAPIError:
        # A temporary menu API failure must not prevent polling from starting.
        logger.exception("Failed to synchronize Telegram command menu")
