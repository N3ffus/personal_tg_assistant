import html
import re
from urllib.parse import urlsplit

import httpx

from src.application.ports.web import WebResult, WebSearchError

SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
MARKUP = re.compile(r"<[^>]+>")


def _plain(value: object, limit: int) -> str:
    return html.unescape(MARKUP.sub("", str(value or ""))).strip()[:limit]


class TavilyWebSearch:
    def __init__(
        self, *, api_key: str, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=15,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    async def search(self, query: str) -> list[WebResult]:
        try:
            response = await self._client.post(
                SEARCH_URL,
                json={
                    "query": query,
                    "max_results": MAX_RESULTS,
                    "search_depth": "basic",
                    "include_answer": False,
                },
            )
            response.raise_for_status()
            raw = response.json().get("results", [])
            if not isinstance(raw, list):
                raise ValueError("invalid search results")
        except (httpx.HTTPError, ValueError, AttributeError, TypeError) as error:
            raise WebSearchError(type(error).__name__) from error
        results: list[WebResult] = []
        for item in raw[:MAX_RESULTS]:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str):
                continue
            if len(url) > 1000 or any(ord(char) < 32 for char in url):
                continue
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            results.append(
                WebResult(
                    title=_plain(item.get("title"), 200),
                    url=url,
                    snippet=_plain(item.get("content"), 600),
                )
            )
        return results

    async def close(self) -> None:
        await self._client.aclose()
