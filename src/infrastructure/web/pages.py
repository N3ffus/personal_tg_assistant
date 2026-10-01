"""Read a public web page as plain text, never reaching the private network.

The bot runs next to Neo4j and its own health endpoint, and a link can come
from a page the model has just read. Every hop is therefore checked twice:
the resolved addresses before connecting, and the address actually connected
to afterwards, so a DNS answer that changes in between cannot slip through.
"""

import asyncio
import codecs
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser

import httpx

from src.application.ports.web import (
    WebPage,
    WebPageBlockedError,
    WebPageError,
    WebPageUnsupportedError,
)

MAX_REDIRECTS = 4
MAX_BYTES = 2_000_000
MAX_TEXT_CHARS = 12_000
TOTAL_SECONDS = 20
TEXT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
USER_AGENT = "Mozilla/5.0 (compatible; PersonalAssistant/1.0)"
META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([\w.:-]+)""", re.I)
# Content that is never the page's text: code, navigation, page chrome.
SKIPPED = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "head",
    "nav",
    "footer",
    "aside",
    "form",
    "iframe",
    "button",
    "select",
}
# The page's own content, when it marks it; menus and link lists stay outside.
FOCUS = {"main", "article"}
MIN_FOCUSED_CHARS = 200
BLOCKS = {
    "address",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}

Resolver = Callable[[str, int], Awaitable[list[str]]]


async def resolve(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    )
    return [str(info[4][0]) for info in infos]


def is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        # Text inside <main>/<article>: the page itself without its chrome.
        self.focused: list[str] = []
        self.title: list[str] = []
        self._skipped = 0
        self._focus = 0
        self._in_title = False

    def _emit(self, text: str) -> None:
        self.parts.append(text)
        if self._focus:
            self.focused.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag == "body":
            # «</head>» is optional in HTML; an unclosed head must not hide
            # the whole page.
            self._skipped = 0
        elif tag in SKIPPED:
            self._skipped += 1
        elif tag in FOCUS:
            self._focus += 1
            self._emit("\n")
        elif tag in BLOCKS:
            self._emit("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in BLOCKS:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in SKIPPED:
            self._skipped = max(0, self._skipped - 1)
        elif tag in FOCUS:
            self._emit("\n")
            self._focus = max(0, self._focus - 1)
        elif tag in BLOCKS:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        elif not self._skipped:
            self._emit(data)


def _normalized(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def html_to_text(markup: str) -> tuple[str, str]:
    """Return the page title and its readable text."""
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    focused = _normalized("".join(parser.focused))
    # A tiny <article> is a teaser card, not the page: keep the whole body then.
    text = (
        focused
        if len(focused) >= MIN_FOCUSED_CHARS
        else _normalized("".join(parser.parts))
    )
    return _normalized("".join(parser.title)), text


def _decode(body: bytes, charset: str | None) -> str:
    if charset is None:
        match = META_CHARSET.search(body[:4096])
        charset = match.group(1).decode("ascii", "ignore") if match else None
    try:
        codecs.lookup(charset or "utf-8")
    except LookupError:
        charset = None
    return body.decode(charset or "utf-8", errors="replace")


class HttpWebPageReader:
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver = resolve,
    ) -> None:
        self._resolve = resolver
        # Redirects are followed by hand so that every hop is checked; a proxy
        # from the environment would hide the address actually connected to.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(10, connect=5),
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"},
            transport=transport,
        )

    async def read(self, url: str) -> WebPage:
        try:
            async with asyncio.timeout(TOTAL_SECONDS):
                return await self._read(httpx.URL(url))
        except TimeoutError as error:
            raise WebPageError("timeout") from error
        except (httpx.HTTPError, httpx.InvalidURL) as error:
            raise WebPageError(type(error).__name__) from error

    async def close(self) -> None:
        await self._client.aclose()

    async def _read(self, url: httpx.URL) -> WebPage:
        for _ in range(MAX_REDIRECTS + 1):
            await self._ensure_public(url)
            async with self._client.stream("GET", url) as response:
                self._ensure_public_peer(response)
                if response.has_redirect_location:
                    url = response.url.join(response.headers["location"])
                    continue
                if response.status_code >= 400:
                    raise WebPageError(f"status {response.status_code}")
                media = (
                    response.headers.get("content-type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                if media and media not in TEXT_TYPES:
                    raise WebPageUnsupportedError(media)
                body = await self._body(response)
                text = _decode(body, response.charset_encoding)
                title, content = (
                    ("", _normalized(text))
                    if media == "text/plain"
                    else html_to_text(text)
                )
                if not content:
                    raise WebPageUnsupportedError("no text")
                return WebPage(
                    url=str(response.url),
                    title=title[:200],
                    text=content[:MAX_TEXT_CHARS],
                )
        raise WebPageError("too many redirects")

    async def _ensure_public(self, url: httpx.URL) -> None:
        if url.scheme not in {"http", "https"} or not url.host:
            raise WebPageUnsupportedError(f"scheme {url.scheme}")
        host = url.raw_host.decode("ascii")
        port = url.port or (443 if url.scheme == "https" else 80)
        try:
            addresses = await self._resolve(host, port)
        except OSError as error:
            raise WebPageError("dns") from error
        if not addresses or not all(is_public(address) for address in addresses):
            raise WebPageBlockedError(host)

    @staticmethod
    def _ensure_public_peer(response: httpx.Response) -> None:
        stream = response.extensions.get("network_stream")
        peer = stream.get_extra_info("server_addr") if stream is not None else None
        if peer and not is_public(str(peer[0])):
            raise WebPageBlockedError(str(peer[0]))

    @staticmethod
    async def _body(response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= MAX_BYTES:
                break
        return b"".join(chunks)[:MAX_BYTES]
