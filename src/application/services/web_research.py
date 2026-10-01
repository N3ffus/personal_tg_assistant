"""Everything the answer turn may read from the internet for one message."""

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from src.application.ports.web import (
    WebPage,
    WebPageBlockedError,
    WebPageError,
    WebPageReader,
    WebPageUnsupportedError,
    WebResult,
    WebSearch,
    WebSearchError,
)
from src.application.services.turn import Turn
from src.domain.assistant.models import (
    AssistantAction,
    ReadWebPageAction,
    SearchWebAction,
)

logger = logging.getLogger(__name__)

MAX_QUERIES = 3
MAX_RESULTS = 8
MAX_PAGES = 2
SOURCES_SHOWN = 3

SEARCH_DISABLED = "Поиск в интернете не настроен: нужен TAVILY_API_KEY."
SEARCH_FAILED = "Поиск в интернете сейчас недоступен. Попробуй позже."
NOTHING_FOUND = "По этому запросу ничего не нашлось в интернете."
READING_DISABLED = "Чтение страниц по ссылке не настроено."
LINK_UNKNOWN = "Открываю только ссылки из нашей переписки — пришли ссылку на страницу."
PAGE_BLOCKED = "Эту ссылку не открываю: она ведёт во внутреннюю сеть."
PAGE_UNSUPPORTED = "По ссылке нет текста, который я могу прочитать."
PAGE_FAILED = "Не удалось открыть страницу: сайт не ответил или не пустил."

LINK = re.compile(r"https?://[^\s<>\"'«»]+", re.IGNORECASE)
TRAILING = ".,;:!?\"'»"


def links_in(text: str) -> list[str]:
    """Links as a person writes them, without the punctuation that follows."""
    links: list[str] = []
    for match in LINK.findall(text):
        link = match
        while link and (
            link[-1] in TRAILING
            or (link[-1] == ")" and link.count(")") > link.count("("))
        ):
            link = link[:-1]
        links.append(link)
    return links


def _key(url: str) -> str:
    # Telegram shows a Cyrillic path decoded, and the model may echo it either
    # way; the same link must still match.
    return unquote(url).rstrip("/")


def _held(url: str, conversation: str, links: dict[str, str]) -> str | None:
    """The link to open, if the conversation already holds it."""
    key = _key(url)
    if key in links:
        return links[key]
    # «Перескажи habr.com/ru/articles/1» carries no scheme; the model adds one.
    scheme, _, rest = key.partition("://")
    if scheme in {"http", "https"} and "." in rest and rest in conversation:
        return url
    return None


def _host(url: str) -> str:
    try:
        return urlsplit(url).hostname or "?"
    except ValueError:
        return "?"


def _page_failure(error: WebPageError) -> str:
    if isinstance(error, WebPageBlockedError):
        return PAGE_BLOCKED
    if isinstance(error, WebPageUnsupportedError):
        return PAGE_UNSUPPORTED
    return PAGE_FAILED


@dataclass(frozen=True, slots=True)
class WebMaterial:
    # JSON the answer turn receives as web_results: untrusted page text.
    payload: str
    # Search result addresses the reply cites; the model never writes them.
    sources: tuple[str, ...]


class WebResearch:
    def __init__(
        self, *, search: WebSearch | None, pages: WebPageReader | None
    ) -> None:
        self._search = search
        self._pages = pages

    async def gather(
        self, actions: Sequence[AssistantAction], *, turn: Turn
    ) -> WebMaterial | str:
        """Run the requested searches and reads; a string is the reply instead."""
        queries = list(
            dict.fromkeys(
                action.query
                for action in actions
                if isinstance(action, SearchWebAction)
            )
        )[:MAX_QUERIES]
        wanted = list(
            dict.fromkeys(
                action.url
                for action in actions
                if isinstance(action, ReadWebPageAction)
            )
        )
        # Only a link the conversation already holds is opened. A page the
        # model has read cannot make it fetch an address of its own making,
        # such as one carrying the user's data in the query string.
        conversation = f"{turn.text}\n{turn.context}"
        known = {_key(link): link for link in links_in(conversation)}
        plain = unquote(conversation)
        held = [
            link
            for link in (_held(url, plain, known) for url in wanted)
            if link is not None
        ]
        if len(held) < len(wanted):
            logger.warning("web.read.unlisted count=%d", len(wanted) - len(held))
        urls = list(dict.fromkeys(held))[:MAX_PAGES]

        failures: list[str] = []
        if wanted and not urls:
            failures.append(LINK_UNKNOWN)
        if urls and self._pages is None:
            failures.append(READING_DISABLED)
            urls = []
        if queries and self._search is None:
            failures.append(SEARCH_DISABLED)
            queries = []

        found = await asyncio.gather(
            *(self._searched(query) for query in queries),
            *(self._read(url) for url in urls),
        )
        results: list[WebResult] = []
        entries: list[dict[str, str]] = []
        for item in found:
            if isinstance(item, str):
                failures.append(item)
            elif isinstance(item, WebPage):
                entries.append(
                    {"title": item.title, "url": item.url, "text": item.text}
                )
            else:
                for result in item:
                    if result not in results:
                        results.append(result)
        results = results[:MAX_RESULTS]
        if queries and not results and SEARCH_FAILED not in failures:
            failures.append(NOTHING_FOUND)
        failures = list(dict.fromkeys(failures))
        entries.extend(
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                **({"age": result.age} if result.age else {}),
            }
            for result in results
        )
        if not entries:
            return failures[0] if failures else NOTHING_FOUND
        # A failure next to real material is data too: the model can say that
        # the second link did not open instead of silently ignoring it.
        entries.extend({"error": failure} for failure in failures)
        return WebMaterial(
            payload=json.dumps(entries, ensure_ascii=False),
            sources=tuple(result.url for result in results[:SOURCES_SHOWN]),
        )

    async def _searched(self, query: str) -> list[WebResult] | str:
        assert self._search is not None
        try:
            return await self._search.search(query)
        except WebSearchError as error:
            logger.warning("web.search.failed reason=%s", error)
            return SEARCH_FAILED

    async def _read(self, url: str) -> WebPage | str:
        assert self._pages is not None
        try:
            return await self._pages.read(url)
        except WebPageError as error:
            logger.warning(
                "web.read.failed host=%s error=%s reason=%s",
                _host(url),
                type(error).__name__,
                error,
            )
            return _page_failure(error)
