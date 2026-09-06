from typing import Literal, Protocol

from src.domain.assistant.context import ChatContext, ContextChat


class ContextRepository(Protocol):
    async def ensure_chat(
        self,
        *,
        owner_id: int,
        kind: Literal["bot", "business"],
        chat_id: int,
        title: str,
    ) -> ContextChat: ...

    async def list_chats(self, *, owner_id: int) -> list[ContextChat]: ...

    async def load(self, *, owner_id: int, context_id: int) -> ChatContext | None: ...

    async def save(
        self, *, owner_id: int, context_id: int, context: ChatContext
    ) -> None: ...


class ContextSummarizer(Protocol):
    async def summarize(self, *, text: str) -> str: ...
