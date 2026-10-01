from aiogram import Bot
from aiogram.types import LinkPreviewOptions


class TelegramNotifier:
    """Delivers the bot's own messages (reminders, calendar alerts) to the owner."""

    def __init__(self, *, bot: Bot) -> None:
        self._bot = bot

    async def send(self, *, user_id: int, text: str) -> None:
        await self._bot.send_message(
            chat_id=user_id,
            text=text,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
