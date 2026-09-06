import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal
from weakref import WeakValueDictionary

from src.application.ports.context import ContextRepository, ContextSummarizer
from src.domain.assistant.context import (
    MAX_SUMMARY_CHARS,
    ChatContext,
    ContextChat,
    ContextMessage,
    clipped,
)


class ContextService:
    def __init__(
        self, *, storage: ContextRepository, summarizer: ContextSummarizer
    ) -> None:
        self._storage = storage
        self._summarizer = summarizer
        self._locks: WeakValueDictionary[tuple[int, int], asyncio.Lock] = (
            WeakValueDictionary()
        )

    async def ensure_chat(
        self,
        *,
        owner_id: int,
        kind: Literal["bot", "business"],
        chat_id: int,
        title: str,
    ) -> ContextChat:
        return await self._storage.ensure_chat(
            owner_id=owner_id, kind=kind, chat_id=chat_id, title=title
        )

    async def list_chats(self, *, owner_id: int) -> list[ContextChat]:
        return await self._storage.list_chats(owner_id=owner_id)

    @asynccontextmanager
    async def session(
        self, *, owner_id: int, context_id: int
    ) -> AsyncIterator[ChatContext]:
        key = (owner_id, context_id)
        lock = self._locks.setdefault(key, asyncio.Lock())
        # Serialize the entire LLM turn with clear/compact to avoid resurrecting data.
        async with lock:
            context = await self._storage.load(owner_id=owner_id, context_id=context_id)
            if context is None:
                raise ValueError("Unknown chat context")
            yield context
            await self._storage.save(
                owner_id=owner_id, context_id=context_id, context=context
            )

    async def read(self, *, owner_id: int, context_id: int) -> ChatContext:
        context = await self._storage.load(owner_id=owner_id, context_id=context_id)
        if context is None:
            raise ValueError("Unknown chat context")
        return context

    async def record(
        self, *, owner_id: int, context_id: int, message: ContextMessage
    ) -> None:
        async with self.session(owner_id=owner_id, context_id=context_id) as context:
            context.append(message)

    async def clear(self, *, owner_id: int, context_id: int) -> None:
        async with self.session(owner_id=owner_id, context_id=context_id) as context:
            context.summary = ""
            context.messages.clear()
            context.discarded_message_id = context.last_message_id

    async def compact(self, *, owner_id: int, context_id: int) -> bool:
        async with self.session(owner_id=owner_id, context_id=context_id) as context:
            if not context.messages:
                return False
            summary = (
                await asyncio.wait_for(
                    self._summarizer.summarize(text=context.render()), timeout=90
                )
            ).strip()
            if not summary:
                raise ValueError("Empty context summary")
            context.summary = clipped(summary, MAX_SUMMARY_CHARS)
            context.messages.clear()
            context.discarded_message_id = context.last_message_id
        return True
