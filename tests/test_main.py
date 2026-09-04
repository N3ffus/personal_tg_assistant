import asyncio
from typing import Any

import pytest

from src.main import POLLING_CONCURRENCY_LIMIT, _run_services


class BlockingDispatcher:
    def __init__(self, *, release: asyncio.Event) -> None:
        self._release = release
        self.started = asyncio.Event()
        self.cancelled = False
        self.bot: object | None = None
        self.kwargs: dict[str, object] = {}

    async def start_polling(self, bot: object, **kwargs: object) -> None:
        self.bot = bot
        self.kwargs = kwargs
        self.started.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class BlockingServer:
    def __init__(self, *, release: asyncio.Event) -> None:
        self._release = release
        self.started = asyncio.Event()
        self.cancelled = False
        self.should_exit = False

    async def serve(self) -> None:
        self.started.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_service_supervisor_cancels_polling_when_http_server_stops() -> None:
    polling_release = asyncio.Event()
    server_release = asyncio.Event()
    dispatcher = BlockingDispatcher(release=polling_release)
    server = BlockingServer(release=server_release)
    bot = object()
    run = asyncio.create_task(
        _run_services(
            dispatcher=dispatcher,  # type: ignore[arg-type]
            bot=bot,  # type: ignore[arg-type]
            server=server,  # type: ignore[arg-type]
        )
    )
    await dispatcher.started.wait()
    await server.started.wait()

    server_release.set()
    await run

    assert dispatcher.bot is bot
    assert dispatcher.kwargs == {
        "handle_signals": False,
        "close_bot_session": False,
        "tasks_concurrency_limit": POLLING_CONCURRENCY_LIMIT,
    }
    assert dispatcher.cancelled is True
    assert server.should_exit is True


@pytest.mark.asyncio
async def test_service_supervisor_propagates_failure_and_cancels_sibling() -> None:
    server_release = asyncio.Event()
    server = BlockingServer(release=server_release)

    class FailingDispatcher:
        async def start_polling(self, bot: object, **kwargs: Any) -> None:
            await server.started.wait()
            raise RuntimeError("polling failed")

    with pytest.raises(RuntimeError, match="polling failed"):
        await _run_services(
            dispatcher=FailingDispatcher(),  # type: ignore[arg-type]
            bot=object(),  # type: ignore[arg-type]
            server=server,  # type: ignore[arg-type]
        )

    assert server.cancelled is True
    assert server.should_exit is True


@pytest.mark.asyncio
async def test_service_supervisor_cleans_up_when_parent_is_cancelled() -> None:
    polling_release = asyncio.Event()
    server_release = asyncio.Event()
    dispatcher = BlockingDispatcher(release=polling_release)
    server = BlockingServer(release=server_release)
    run = asyncio.create_task(
        _run_services(
            dispatcher=dispatcher,  # type: ignore[arg-type]
            bot=object(),  # type: ignore[arg-type]
            server=server,  # type: ignore[arg-type]
        )
    )
    await dispatcher.started.wait()
    await server.started.wait()

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert dispatcher.cancelled is True
    assert server.cancelled is True
    assert server.should_exit is True
