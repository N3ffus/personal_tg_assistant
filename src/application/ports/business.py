from typing import Protocol

from src.domain.assistant.business import BusinessIntent, StoredBusinessAction


class BusinessStore(Protocol):
    async def cursor(self, *, owner_id: int, context_id: int) -> int: ...

    async def prepare(
        self,
        *,
        owner_id: int,
        context_id: int,
        expected_cursor: int,
        last_message_id: int,
        intents: list[BusinessIntent],
    ) -> bool: ...

    async def outstanding(
        self,
        *,
        owner_id: int,
        context_id: int,
    ) -> list[StoredBusinessAction]: ...

    async def claim(self, *, owner_id: int, action_id: str) -> bool: ...

    async def complete(self, *, owner_id: int, action_id: str, result: str) -> None: ...

    async def notified(self, *, owner_id: int, action_id: str) -> None: ...
