import json
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import aiosqlite
from cryptography.fernet import Fernet


class CalendarStorage:
    def __init__(self, *, database_path: str, encryption_key: str) -> None:
        self._database_path = database_path
        self._cipher = Fernet(encryption_key.encode())

    async def initialize(self) -> None:
        Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._database_path) as database:
            await database.executescript(
                """
                CREATE TABLE IF NOT EXISTS oauth_states (
                    state TEXT PRIMARY KEY,
                    telegram_user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calendar_connections (
                    telegram_user_id INTEGER PRIMARY KEY,
                    credentials BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_operations (
                    operation_id TEXT PRIMARY KEY,
                    telegram_user_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS oauth_states_expires_at_idx
                    ON oauth_states (expires_at);
                CREATE INDEX IF NOT EXISTS pending_operations_expires_at_idx
                    ON pending_operations (expires_at);
                """
            )
            await self._purge_expired(database)
            await database.commit()

    async def create_oauth_state(self, *, user_id: int) -> str:
        state = secrets.token_urlsafe(32)
        expires_at = self._expires_in(minutes=10)
        async with aiosqlite.connect(self._database_path) as database:
            await self._purge_expired(database)
            await database.execute(
                "DELETE FROM oauth_states WHERE telegram_user_id = ?",
                (user_id,),
            )
            await database.execute(
                "INSERT INTO oauth_states VALUES (?, ?, ?)",
                (state, user_id, expires_at),
            )
            await database.commit()
        return state

    async def consume_oauth_state(self, *, state: str) -> int | None:
        async with aiosqlite.connect(self._database_path) as database:
            cursor = await database.execute(
                """DELETE FROM oauth_states WHERE state = ?
                RETURNING telegram_user_id, expires_at""",
                (state,),
            )
            row = await cursor.fetchone()
            await database.commit()
        if row is None or datetime.fromisoformat(row[1]) <= datetime.now(UTC):
            return None
        return int(row[0])

    async def save_credentials(
        self, *, user_id: int, credentials: Mapping[str, object]
    ) -> None:
        encrypted = self._cipher.encrypt(json.dumps(credentials).encode())
        async with aiosqlite.connect(self._database_path) as database:
            await database.execute(
                """INSERT INTO calendar_connections (telegram_user_id, credentials)
                VALUES (?, ?)
                ON CONFLICT(telegram_user_id) DO UPDATE SET
                credentials=excluded.credentials""",
                (user_id, encrypted),
            )
            await database.commit()

    async def load_credentials(self, *, user_id: int) -> dict[str, object] | None:
        async with aiosqlite.connect(self._database_path) as database:
            cursor = await database.execute(
                "SELECT credentials FROM calendar_connections WHERE telegram_user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return cast(dict[str, object], json.loads(self._cipher.decrypt(row[0])))

    async def delete_connection(self, *, user_id: int) -> None:
        async with aiosqlite.connect(self._database_path) as database:
            await database.execute(
                "DELETE FROM calendar_connections WHERE telegram_user_id = ?",
                (user_id,),
            )
            await database.commit()

    async def create_operation(
        self, *, user_id: int, kind: str, payload: dict[str, object]
    ) -> str:
        operation_ids = await self.create_operations(
            user_id=user_id,
            kind=kind,
            payloads=[payload],
        )
        return operation_ids[0]

    async def create_operations(
        self,
        *,
        user_id: int,
        kind: str,
        payloads: list[dict[str, object]],
    ) -> list[str]:
        if not payloads:
            return []

        operation_ids = [secrets.token_urlsafe(12) for _ in payloads]
        expires_at = self._expires_in(minutes=15)
        async with aiosqlite.connect(self._database_path) as database:
            await self._purge_expired(database)
            if kind == "select":
                await database.execute(
                    """DELETE FROM pending_operations
                    WHERE telegram_user_id = ? AND kind = 'select'""",
                    (user_id,),
                )
            await database.executemany(
                "INSERT INTO pending_operations VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        operation_id,
                        user_id,
                        kind,
                        json.dumps(payload),
                        expires_at,
                    )
                    for operation_id, payload in zip(
                        operation_ids, payloads, strict=True
                    )
                ],
            )
            await database.commit()
        return operation_ids

    async def consume_operation(
        self, *, operation_id: str, user_id: int
    ) -> tuple[str, dict[str, object]] | None:
        async with aiosqlite.connect(self._database_path) as database:
            cursor = await database.execute(
                """DELETE FROM pending_operations
                WHERE operation_id = ? AND telegram_user_id = ?
                RETURNING kind, payload, expires_at""",
                (operation_id, user_id),
            )
            row = await cursor.fetchone()
            await database.commit()
        if row is None or datetime.fromisoformat(row[2]) <= datetime.now(UTC):
            return None
        return str(row[0]), json.loads(row[1])

    async def consume_latest_operation(
        self, *, user_id: int, kind: str
    ) -> dict[str, object] | None:
        async with aiosqlite.connect(self._database_path) as database:
            cursor = await database.execute(
                """DELETE FROM pending_operations
                WHERE operation_id = (
                    SELECT operation_id FROM pending_operations
                    WHERE telegram_user_id = ? AND kind = ?
                    ORDER BY rowid DESC LIMIT 1
                ) AND telegram_user_id = ?
                RETURNING payload, expires_at""",
                (user_id, kind, user_id),
            )
            row = await cursor.fetchone()
            await database.commit()
        if row is None:
            return None
        if datetime.fromisoformat(row[1]) <= datetime.now(UTC):
            return None
        return cast(dict[str, object], json.loads(row[0]))

    @staticmethod
    def _expires_in(*, minutes: int) -> str:
        return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()

    @staticmethod
    async def _purge_expired(database: aiosqlite.Connection) -> None:
        now = datetime.now(UTC).isoformat()
        await database.execute("DELETE FROM oauth_states WHERE expires_at <= ?", (now,))
        await database.execute(
            "DELETE FROM pending_operations WHERE expires_at <= ?",
            (now,),
        )
