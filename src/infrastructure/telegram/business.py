import asyncio
import logging

from aiogram import Bot, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BusinessConnection, Message

from src.application.services.context import ContextService
from src.domain.assistant.context import ContextMessage
from src.infrastructure.telegram.business_worker import BusinessDialogWorker

logger = logging.getLogger(__name__)
REJECTED_CONNECTION_LIMIT = 128


def create_business_router(
    *,
    allowed_user_id: int,
    contexts: ContextService | None = None,
    extractor: BusinessDialogWorker | None = None,
) -> Router:
    router = Router(name=__name__)
    connections: dict[str, BusinessConnection] = {}
    connection_loads: dict[str, asyncio.Task[BusinessConnection]] = {}
    rejected_connections: set[str] = set()

    def is_active_owner(connection: BusinessConnection) -> bool:
        return connection.user.id == allowed_user_id and connection.is_enabled

    def reject_connection(connection_id: str) -> None:
        connections.pop(connection_id, None)
        if connection_id in rejected_connections:
            return
        if len(rejected_connections) >= REJECTED_CONNECTION_LIMIT:
            rejected_connections.pop()
        rejected_connections.add(connection_id)

    async def load_connection(
        connection_id: str,
        bot: Bot,
    ) -> BusinessConnection | None:
        if connection_id in rejected_connections:
            return None

        connection = connections.get(connection_id)
        if connection is not None:
            return connection

        load = connection_loads.get(connection_id)
        if load is None:
            load = asyncio.create_task(bot.get_business_connection(connection_id))
            connection_loads[connection_id] = load

        try:
            connection = await load
        except TelegramAPIError:
            if connection_loads.get(connection_id) is load:
                connection_loads.pop(connection_id)
                logger.exception(
                    "Failed to load Telegram Business connection %s",
                    connection_id,
                )
                return None
            return connections.get(connection_id)
        except BaseException:
            if connection_loads.get(connection_id) is load:
                connection_loads.pop(connection_id)
            raise

        if connection_loads.get(connection_id) is not load:
            return connections.get(connection_id)
        connection_loads.pop(connection_id)

        if connection.id != connection_id or not is_active_owner(connection):
            reject_connection(connection_id)
            return None

        rejected_connections.discard(connection_id)
        connections[connection_id] = connection
        return connection

    @router.business_connection()
    async def connection_changed(connection: BusinessConnection) -> None:
        connection_loads.pop(connection.id, None)
        if is_active_owner(connection):
            rejected_connections.discard(connection.id)
            connections[connection.id] = connection
        else:
            reject_connection(connection.id)
        logger.info(
            "Telegram Business connection %s is now %s",
            connection.id,
            "enabled" if connection.is_enabled else "disabled",
        )

    @router.business_message()
    @router.edited_business_message()
    async def read_business_message(message: Message, bot: Bot) -> None:
        connection_id = message.business_connection_id
        if connection_id is None:
            return

        connection = await load_connection(connection_id, bot)
        if connection is None:
            return

        if contexts is not None and (message.text or message.caption):
            title = message.chat.title or message.chat.full_name or str(message.chat.id)
            if message.chat.username:
                title += f" (@{message.chat.username})"
            sender = (
                "Бизнес-бот"
                if message.sender_business_bot is not None
                else "Вы"
                if message.from_user is not None
                and message.from_user.id == allowed_user_id
                else message.from_user.full_name
                if message.from_user is not None
                else title
            )
            try:
                chat = await contexts.ensure_chat(
                    owner_id=allowed_user_id,
                    kind="business",
                    chat_id=message.chat.id,
                    title=title,
                )
                await contexts.record(
                    owner_id=allowed_user_id,
                    context_id=chat.id,
                    message=ContextMessage(
                        role="user",
                        sender=sender,
                        text=message.text or message.caption or "",
                        sent_at=message.date,
                        message_id=message.message_id,
                        reply_to_message_id=(
                            message.reply_to_message.message_id
                            if message.reply_to_message is not None
                            else None
                        ),
                        sender_id=message.from_user.id if message.from_user else None,
                        is_business_bot=message.sender_business_bot is not None,
                        is_forwarded=message.forward_origin is not None,
                    ),
                )
                if (
                    extractor is not None
                    and message.sender_business_bot is None
                    and message.from_user is not None
                ):
                    extractor.submit(chat)
            except Exception:
                logger.exception("Failed to store Telegram Business context")

        if message.sender_business_bot is not None:
            return

        if message.from_user is not None and message.from_user.id == allowed_user_id:
            return

        if connection.rights is None or connection.rights.can_read_messages is not True:
            logger.warning(
                "Telegram Business connection %s cannot mark messages as read",
                connection_id,
            )
            return

        try:
            await bot.read_business_message(
                business_connection_id=connection_id,
                chat_id=message.chat.id,
                message_id=message.message_id,
            )
        except TelegramAPIError:
            logger.exception(
                "Failed to mark Telegram Business message %s as read",
                message.message_id,
            )

    return router
