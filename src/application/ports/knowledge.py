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

    async def recent_watched_films(
        self, *, namespace: str, limit: int
    ) -> list[KnowledgeFact]: ...

    async def close(self) -> None: ...

    async def watched_film_titles(self, *, namespace: str) -> list[str]: ...

    async def profile_facts(
        self, *, namespace: str, limit: int
    ) -> list[KnowledgeFact]: ...

    async def forget_candidates(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]: ...

    async def forget(self, *, namespace: str, refs: list[str]) -> list[str]: ...
