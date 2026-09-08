from typing import Any

import httpx

from src.application.ports.films import FilmDirectoryError
from src.domain.knowledge.films import FilmRecord, title_key

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

# A candidate whose year is a year out is still the same film; two years out is
# a different one. The model quotes release years loosely, not wrongly.
YEAR_TOLERANCE = 1


class TMDBFilmDirectory:
    """Confirms a recommended title names a real film.

    Matching is deliberately strict: a search hit counts only when the title the
    model wrote is the film's Russian or original title. A loose match would let
    exactly the invented titles through that this directory exists to catch, and
    a dropped real film costs nothing — the batch is generated oversized.
    """

    def __init__(
        self,
        *,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=10, transport=transport)

    async def find(self, *, title: str, year: int | None) -> FilmRecord | None:
        params: dict[str, Any] = {
            "api_key": self._api_key,
            "query": title,
            "language": "ru-RU",
            "include_adult": "false",
        }
        try:
            response = await self._client.get(TMDB_SEARCH_URL, params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
        except (httpx.HTTPError, ValueError) as error:
            raise FilmDirectoryError(type(error).__name__) from error
        wanted = title_key(title)
        for result in results:
            record = _record(result)
            if record is None or wanted not in {
                title_key(record.title),
                title_key(record.original_title),
            }:
                continue
            if (
                year is not None
                and record.year is not None
                and abs(record.year - year) > YEAR_TOLERANCE
            ):
                continue
            return record
        return None

    async def close(self) -> None:
        await self._client.aclose()


def _record(result: Any) -> FilmRecord | None:
    if not isinstance(result, dict):
        return None
    title = str(result.get("title") or "")
    original = str(result.get("original_title") or "")
    if not title and not original:
        return None
    released = str(result.get("release_date") or "")[:4]
    return FilmRecord(
        title=title or original,
        original_title=original or title,
        year=int(released) if released.isdigit() else None,
    )
