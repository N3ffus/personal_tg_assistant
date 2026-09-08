from typing import Protocol

from src.domain.knowledge.films import FilmRecord


class FilmDirectoryError(Exception):
    """Expected film directory failure: recommendations degrade unverified."""


class FilmDirectory(Protocol):
    """An external catalogue of real films.

    The knowledge graph only holds what the user watched, so nothing inside the
    application can tell an invented title from a real one.
    """

    async def find(self, *, title: str, year: int | None) -> FilmRecord | None: ...

    async def close(self) -> None: ...
