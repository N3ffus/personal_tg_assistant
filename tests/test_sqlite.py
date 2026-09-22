from pathlib import Path

import aiosqlite
import pytest
from cryptography.fernet import Fernet

from src.infrastructure.calendar.storage import CalendarStorage
from src.infrastructure.context.business_storage import BusinessStorage
from src.infrastructure.context.storage import ContextStorage
from src.infrastructure.sqlite import BUSY_TIMEOUT_SECONDS, connect


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", [CalendarStorage, ContextStorage, BusinessStorage])
async def test_every_storage_switches_the_database_to_write_ahead_log(
    tmp_path: Path, storage: type[CalendarStorage | ContextStorage | BusinessStorage]
) -> None:
    """Readers must not wait on a writer: polling runs 32 updates at once."""
    path = str(tmp_path / "assistant.db")

    await storage(
        database_path=path, encryption_key=Fernet.generate_key().decode()
    ).initialize()

    async with aiosqlite.connect(path) as database:
        cursor = await database.execute("PRAGMA journal_mode")
        assert await cursor.fetchone() == ("wal",)


@pytest.mark.asyncio
async def test_connections_wait_for_a_busy_database(tmp_path: Path) -> None:
    async with connect(str(tmp_path / "assistant.db")) as database:
        cursor = await database.execute("PRAGMA busy_timeout")
        assert await cursor.fetchone() == (BUSY_TIMEOUT_SECONDS * 1000,)
