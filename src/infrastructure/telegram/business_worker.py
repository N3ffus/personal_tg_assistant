import asyncio
import logging
from html import escape

from aiogram import Bot
from aiogram.types import LinkPreviewOptions

from src.application.services.context import ContextService
from src.application.use_cases.process_business_dialog import ProcessBusinessDialog
from src.domain.assistant.context import ContextChat

logger = logging.getLogger(__name__)


class OwnerBusinessNotifier:
    def __init__(self, *, bot: Bot, owner_id: int) -> None:
        self._bot = bot
        self._owner_id = owner_id

    async def __call__(self, title: str, result: str) -> None:
        # Never use a business connection or a peer chat for notifications.
        await self._bot.send_message(
            chat_id=self._owner_id,
            text=f"<b>Из переписки с {escape(title[:150])}:</b>\n{result}",
            parse_mode="HTML",
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )


class BusinessDialogWorker:
    def __init__(
        self, *, processor: ProcessBusinessDialog, debounce_seconds: float = 7
    ) -> None:
        self._processor = processor
        self._delay = debounce_seconds
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._events: dict[int, asyncio.Event] = {}
        self._chats: dict[int, ContextChat] = {}
        self._closing = False
        self._slots = asyncio.Semaphore(4)

    def submit(self, chat: ContextChat) -> None:
        if self._closing:
            return
        self._chats[chat.id] = chat
        if chat.id in self._events:
            self._events[chat.id].set()
        else:
            event = asyncio.Event()
            self._events[chat.id] = event
            self._tasks[chat.id] = asyncio.create_task(self._run(chat.id, event))

    async def resume(self, *, contexts: ContextService, owner_id: int) -> None:
        for chat in await contexts.list_chats(owner_id=owner_id):
            if chat.kind == "business":
                self.submit(chat)

    async def _run(self, context_id: int, event: asyncio.Event) -> None:
        failures = 0
        try:
            while not self._closing:
                event.clear()
                try:
                    await asyncio.wait_for(event.wait(), timeout=self._delay)
                    continue
                except TimeoutError:
                    pass
                if self._closing:
                    return
                try:
                    async with self._slots:
                        await self._processor.process(self._chats[context_id])
                    failures = 0
                except Exception as error:
                    failures += 1
                    logger.warning(
                        "Business extraction failed: %s", type(error).__name__
                    )
                    if failures < 3:
                        continue
                if not event.is_set():
                    return
        finally:
            self._events.pop(context_id, None)
            self._tasks.pop(context_id, None)
            self._chats.pop(context_id, None)

    async def close(self) -> None:
        self._closing = True
        for event in self._events.values():
            event.set()
        tasks = list(self._tasks.values())
        if tasks:
            # Leave uncertain remote writes claimed if shutdown exceeds its grace period.
            done, pending = await asyncio.wait(tasks, timeout=20)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
