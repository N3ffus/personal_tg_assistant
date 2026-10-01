import html
import re
from urllib.parse import urlsplit

import httpx

from src.application.ports.web import WebResult, WebSearchError

SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
MAX_RESULTS = 5
# Brave highlights query words with <strong> and escapes entities in titles and
# descriptions; the model and the user need plain text.
MARKUP = re.compile(r"<[^>]+>")


def _plain(value: object, limit: int) -> str:
    return html.unescape(MARKUP.sub("", str(value or ""))).strip()[:limit]


class BraveWebSearch:
    def __init__(
        self, *, api_key: str, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=10,
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            transport=transport,
        )

    async def search(self, query: str) -> list[WebResult]:
        try:
            response = await self._client.get(
                SEARCH_URL, params={"q": query, "count": MAX_RESULTS}
            )
            response.raise_for_status()
            payload = response.json()
            raw = payload.get("web", {}).get("results", [])
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
                    snippet=_plain(item.get("description"), 600),
                    age=_plain(item.get("age"), 40),
                )
            )
        return results

    async def close(self) -> None:
        await self._client.aclose()
