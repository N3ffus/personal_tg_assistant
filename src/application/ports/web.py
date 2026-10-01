from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WebResult:
    title: str
    url: str
    snippet: str
    # How old the page is as the search service reports it («2 days ago»);
    # the model cannot tell stale news from current without it.
    age: str = ""


@dataclass(frozen=True, slots=True)
class WebPage:
    # The address the text came from, after redirects.
    url: str
    title: str
    text: str


class WebSearchError(Exception):
    """The configured search service could not answer."""


class WebPageError(Exception):
    """The page could not be read."""


class WebPageBlockedError(WebPageError):
    """The address leads into a private network; it is never fetched."""


class WebPageUnsupportedError(WebPageError):
    """The address serves something other than a text page."""


class WebSearch(Protocol):
    async def search(self, query: str) -> list[WebResult]: ...

    async def close(self) -> None: ...


class WebPageReader(Protocol):
    async def read(self, url: str) -> WebPage: ...

    async def close(self) -> None: ...
