from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from src.application.ports.films import FilmDirectoryError
from src.domain.knowledge.films import FilmRecord
from src.infrastructure.films.tmdb import TMDBFilmDirectory

ResponseHandler = Callable[[httpx.Request], httpx.Response]


@asynccontextmanager
async def directory(handler: ResponseHandler) -> AsyncIterator[TMDBFilmDirectory]:
    client = TMDBFilmDirectory(
        api_key="test-key", transport=httpx.MockTransport(handler)
    )
    try:
        yield client
    finally:
        await client.close()


def results(*films: dict[str, Any]) -> ResponseHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "test-key"
        assert request.url.params["language"] == "ru-RU"
        return httpx.Response(200, json={"results": list(films)})

    return handler


@pytest.mark.asyncio
async def test_a_title_the_directory_does_not_know_is_not_confirmed() -> None:
    """The invented «Список контактов» returns only unrelated films."""
    handler = results(
        {"title": "Контакт", "original_title": "Contact", "release_date": "1997-07-11"},
        {"title": "Список", "original_title": "The List", "release_date": "2007-01-01"},
    )

    async with directory(handler) as client:
        assert await client.find(title="Список контактов", year=2023) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Бегущий по лезвию 2049", "Blade Runner 2049"])
async def test_a_russian_or_original_title_confirms_the_same_film(query: str) -> None:
    handler = results(
        {
            "title": "Бегущий по лезвию 2049",
            "original_title": "Blade Runner 2049",
            "release_date": "2017-10-04",
        }
    )

    async with directory(handler) as client:
        assert await client.find(title=query, year=None) == FilmRecord(
            title="Бегущий по лезвию 2049",
            original_title="Blade Runner 2049",
            year=2017,
        )


@pytest.mark.asyncio
async def test_a_year_two_years_off_names_a_different_film() -> None:
    handler = results(
        {"title": "Дюна", "original_title": "Dune", "release_date": "2021-09-15"}
    )

    async with directory(handler) as client:
        assert await client.find(title="Дюна", year=2022) is not None
        assert await client.find(title="Дюна", year=1984) is None


@pytest.mark.asyncio
async def test_a_missing_release_date_still_confirms_the_film() -> None:
    handler = results({"title": "Дюна", "original_title": "Dune"})

    async with directory(handler) as client:
        record = await client.find(title="Дюна", year=2021)

    assert record == FilmRecord(title="Дюна", original_title="Dune", year=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_an_unhappy_directory_is_reported_not_swallowed(status: int) -> None:
    async with directory(lambda request: httpx.Response(status)) as client:
        with pytest.raises(FilmDirectoryError):
            await client.find(title="Дюна", year=None)
