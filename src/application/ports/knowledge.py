from typing import Protocol

from src.domain.knowledge.models import KnowledgeEpisode, KnowledgeFact


class KnowledgeMemoryError(Exception):
    """Expected knowledge memory failure: the agent degrades without memory."""


class KnowledgeMemory(Protocol):
    async def remember(self, *, namespace: str, episode: KnowledgeEpisode) -> None: ...

    async def search(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]: ...

    async def healthcheck(self) -> bool: ...

    async def close(self) -> None: ...
