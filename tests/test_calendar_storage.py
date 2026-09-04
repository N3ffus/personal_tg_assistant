import asyncio
from pathlib import Path

import aiosqlite
import pytest
from cryptography.fernet import Fernet, InvalidToken

from src.infrastructure.calendar.storage import CalendarStorage


def make_storage(database_path: Path, *, key: bytes | None = None) -> CalendarStorage:
    return CalendarStorage(
        database_path=str(database_path),
        encryption_key=(key or Fernet.generate_key()).decode(),
    )


@pytest.mark.asyncio
async def test_initialize_creates_parent_directory_and_is_idempotent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "calendar.db"
    storage = make_storage(database_path)

    await storage.initialize()
    await storage.initialize()

    assert database_path.is_file()
    async with aiosqlite.connect(database_path) as database:
        cursor = await database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
        rows = await cursor.fetchall()
    assert [row[0] for row in rows] == [
        "calendar_connections",
        "oauth_states",
        "pending_operations",
    ]


@pytest.mark.asyncio
async def test_oauth_state_is_bound_to_user_and_single_use(tmp_path: Path) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()

    state = await storage.create_oauth_state(user_id=42)

    assert await storage.consume_oauth_state(state=state) == 42
    assert await storage.consume_oauth_state(state=state) is None


@pytest.mark.asyncio
async def test_oauth_states_are_random_and_independent(tmp_path: Path) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()

    first = await storage.create_oauth_state(user_id=42)
    second = await storage.create_oauth_state(user_id=7)

    assert first != second
    assert len(first) >= 32
    assert await storage.consume_oauth_state(state=second) == 7
    assert await storage.consume_oauth_state(state=first) == 42


@pytest.mark.asyncio
async def test_new_oauth_state_invalidates_previous_state_for_same_user(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()

    previous = await storage.create_oauth_state(user_id=42)
    current = await storage.create_oauth_state(user_id=42)

    assert await storage.consume_oauth_state(state=previous) is None
    assert await storage.consume_oauth_state(state=current) == 42


@pytest.mark.asyncio
async def test_expired_oauth_state_is_rejected_and_removed(tmp_path: Path) -> None:
    database_path = tmp_path / "calendar.db"
    storage = make_storage(database_path)
    await storage.initialize()
    state = await storage.create_oauth_state(user_id=42)
    async with aiosqlite.connect(database_path) as database:
        await database.execute(
            "UPDATE oauth_states SET expires_at = ? WHERE state = ?",
            ("2000-01-01T00:00:00+00:00", state),
        )
        await database.commit()

    assert await storage.consume_oauth_state(state=state) is None
    assert await storage.consume_oauth_state(state=state) is None


@pytest.mark.asyncio
async def test_oauth_state_can_only_be_consumed_once_under_concurrency(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()
    state = await storage.create_oauth_state(user_id=42)

    results = await asyncio.gather(
        storage.consume_oauth_state(state=state),
        storage.consume_oauth_state(state=state),
    )

    assert sorted(results, key=lambda value: value is None) == [42, None]


@pytest.mark.asyncio
async def test_credentials_round_trip_encrypted_and_are_isolated_by_user(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "calendar.db"
    storage = make_storage(database_path)
    await storage.initialize()
    credentials = {
        "token": "plain-secret-token",
        "refresh_token": "plain-refresh-token",
        "expiry": "2026-09-03T12:00:00Z",
    }

    await storage.save_credentials(user_id=42, credentials=credentials)

    assert await storage.load_credentials(user_id=42) == credentials
    assert await storage.load_credentials(user_id=7) is None
    async with aiosqlite.connect(database_path) as database:
        cursor = await database.execute(
            "SELECT credentials FROM calendar_connections WHERE telegram_user_id = ?",
            (42,),
        )
        row = await cursor.fetchone()
    assert row is not None
    assert b"plain-secret-token" not in bytes(row[0])
    assert b"plain-refresh-token" not in bytes(row[0])


@pytest.mark.asyncio
async def test_saving_credentials_replaces_previous_value(tmp_path: Path) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()
    await storage.save_credentials(user_id=42, credentials={"token": "old"})

    await storage.save_credentials(user_id=42, credentials={"token": "new"})

    assert await storage.load_credentials(user_id=42) == {"token": "new"}


@pytest.mark.asyncio
async def test_credentials_cannot_be_decrypted_with_another_key(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "calendar.db"
    storage = make_storage(database_path)
    await storage.initialize()
    await storage.save_credentials(user_id=42, credentials={"token": "secret"})
    another_storage = make_storage(database_path)

    with pytest.raises(InvalidToken):
        await another_storage.load_credentials(user_id=42)


@pytest.mark.asyncio
async def test_delete_connection_is_idempotent(tmp_path: Path) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()
    await storage.save_credentials(user_id=42, credentials={"token": "secret"})

    await storage.delete_connection(user_id=42)
    await storage.delete_connection(user_id=42)

    assert await storage.load_credentials(user_id=42) is None


@pytest.mark.asyncio
async def test_pending_operation_is_bound_to_user_and_single_use(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()
    operation_id = await storage.create_operation(
        user_id=42,
        kind="delete",
        payload={"event_id": "event-1"},
    )

    assert await storage.consume_operation(operation_id=operation_id, user_id=7) is None
    assert await storage.consume_operation(operation_id=operation_id, user_id=42) == (
        "delete",
        {"event_id": "event-1"},
    )
    assert (
        await storage.consume_operation(operation_id=operation_id, user_id=42) is None
    )


@pytest.mark.asyncio
async def test_expired_pending_operation_is_rejected_and_removed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "calendar.db"
    storage = make_storage(database_path)
    await storage.initialize()
    operation_id = await storage.create_operation(
        user_id=42,
        kind="delete",
        payload={"event_id": "event-1"},
    )
    async with aiosqlite.connect(database_path) as database:
        await database.execute(
            "UPDATE pending_operations SET expires_at = ? WHERE operation_id = ?",
            ("2000-01-01T00:00:00+00:00", operation_id),
        )
        await database.commit()

    assert (
        await storage.consume_operation(operation_id=operation_id, user_id=42) is None
    )
    assert (
        await storage.consume_operation(operation_id=operation_id, user_id=42) is None
    )


@pytest.mark.asyncio
async def test_pending_operation_can_only_be_consumed_once_under_concurrency(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()
    operation_id = await storage.create_operation(
        user_id=42,
        kind="delete",
        payload={"event_id": "event-1"},
    )

    results = await asyncio.gather(
        storage.consume_operation(operation_id=operation_id, user_id=42),
        storage.consume_operation(operation_id=operation_id, user_id=42),
    )

    assert results.count(None) == 1
    assert results.count(("delete", {"event_id": "event-1"})) == 1


@pytest.mark.asyncio
async def test_consume_latest_operation_selects_latest_matching_user_and_kind(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()
    await storage.create_operation(
        user_id=42,
        kind="update",
        payload={"event_id": "older"},
    )
    await storage.create_operation(
        user_id=42,
        kind="delete",
        payload={"event_id": "delete"},
    )
    await storage.create_operation(
        user_id=7,
        kind="update",
        payload={"event_id": "other-user"},
    )
    await storage.create_operation(
        user_id=42,
        kind="update",
        payload={"event_id": "latest"},
    )

    assert await storage.consume_latest_operation(user_id=42, kind="update") == {
        "event_id": "latest"
    }
    assert await storage.consume_latest_operation(user_id=42, kind="update") == {
        "event_id": "older"
    }
    assert await storage.consume_latest_operation(user_id=42, kind="update") is None
    assert await storage.consume_latest_operation(user_id=7, kind="update") == {
        "event_id": "other-user"
    }
    assert await storage.consume_latest_operation(user_id=42, kind="delete") == {
        "event_id": "delete"
    }


@pytest.mark.asyncio
async def test_consume_latest_operation_rejects_expired_latest_record(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "calendar.db"
    storage = make_storage(database_path)
    await storage.initialize()
    await storage.create_operation(
        user_id=42,
        kind="update",
        payload={"event_id": "older-valid"},
    )
    latest_id = await storage.create_operation(
        user_id=42,
        kind="update",
        payload={"event_id": "latest-expired"},
    )
    async with aiosqlite.connect(database_path) as database:
        await database.execute(
            "UPDATE pending_operations SET expires_at = ? WHERE operation_id = ?",
            ("2000-01-01T00:00:00+00:00", latest_id),
        )
        await database.commit()

    assert await storage.consume_latest_operation(user_id=42, kind="update") is None
    assert await storage.consume_latest_operation(user_id=42, kind="update") == {
        "event_id": "older-valid"
    }


@pytest.mark.asyncio
async def test_create_operations_persists_batch_with_short_unique_tokens(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()

    operation_ids = await storage.create_operations(
        user_id=42,
        kind="select",
        payloads=[{"event_id": "first"}, {"event_id": "second"}],
    )

    assert len(operation_ids) == 2
    assert len(set(operation_ids)) == 2
    assert all(len(operation_id.encode()) <= 20 for operation_id in operation_ids)
    assert await storage.consume_operation(
        operation_id=operation_ids[0], user_id=42
    ) == ("select", {"event_id": "first"})
    assert await storage.consume_operation(
        operation_id=operation_ids[1], user_id=42
    ) == ("select", {"event_id": "second"})


@pytest.mark.asyncio
async def test_new_selection_batch_invalidates_previous_event_buttons(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path / "calendar.db")
    await storage.initialize()
    previous = await storage.create_operations(
        user_id=42,
        kind="select",
        payloads=[{"event_id": "old"}],
    )

    current = await storage.create_operations(
        user_id=42,
        kind="select",
        payloads=[{"event_id": "new"}],
    )

    assert await storage.consume_operation(operation_id=previous[0], user_id=42) is None
    assert await storage.consume_operation(operation_id=current[0], user_id=42) == (
        "select",
        {"event_id": "new"},
    )


@pytest.mark.asyncio
async def test_initialize_purges_abandoned_expired_records(tmp_path: Path) -> None:
    database_path = tmp_path / "calendar.db"
    storage = make_storage(database_path)
    await storage.initialize()
    async with aiosqlite.connect(database_path) as database:
        await database.execute(
            "INSERT INTO oauth_states VALUES (?, ?, ?)",
            ("expired-state", 42, "2000-01-01T00:00:00+00:00"),
        )
        await database.execute(
            "INSERT INTO pending_operations VALUES (?, ?, ?, ?, ?)",
            (
                "expired-operation",
                42,
                "delete",
                '{"event_id": "old"}',
                "2000-01-01T00:00:00+00:00",
            ),
        )
        await database.commit()

    await storage.initialize()

    async with aiosqlite.connect(database_path) as database:
        state_count = await (
            await database.execute(
                "SELECT COUNT(*) FROM oauth_states WHERE state = ?",
                ("expired-state",),
            )
        ).fetchone()
        operation_count = await (
            await database.execute(
                "SELECT COUNT(*) FROM pending_operations WHERE operation_id = ?",
                ("expired-operation",),
            )
        ).fetchone()
    assert state_count == (0,)
    assert operation_count == (0,)


@pytest.mark.asyncio
async def test_creating_state_purges_previous_expired_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "calendar.db"
    storage = make_storage(database_path)
    await storage.initialize()
    async with aiosqlite.connect(database_path) as database:
        await database.execute(
            "INSERT INTO oauth_states VALUES (?, ?, ?)",
            ("abandoned", 42, "2000-01-01T00:00:00+00:00"),
        )
        await database.commit()

    current = await storage.create_oauth_state(user_id=42)

    async with aiosqlite.connect(database_path) as database:
        cursor = await database.execute("SELECT state FROM oauth_states ORDER BY state")
        rows = await cursor.fetchall()
    assert [str(row[0]) for row in rows] == [current]
