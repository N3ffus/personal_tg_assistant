import hashlib
import json
from pathlib import Path

import aiosqlite
from cryptography.fernet import Fernet

from src.domain.assistant.business import BusinessIntent, StoredBusinessAction


class BusinessStorage:
    def __init__(self, *, database_path: str, encryption_key: str) -> None:
        self._database_path = database_path
        self._cipher = Fernet(encryption_key.encode())

    async def initialize(self) -> None:
        Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._database_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS business_cursors (
                    owner_id INTEGER NOT NULL, context_id INTEGER NOT NULL,
                    last_message_id INTEGER NOT NULL,
                    PRIMARY KEY (owner_id, context_id)
                );
                CREATE TABLE IF NOT EXISTS business_actions (
                    id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL,
                    context_id INTEGER NOT NULL, payload BLOB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', result BLOB,
                    notified INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS business_outstanding
                    ON business_actions (owner_id, context_id, notified);
            """)

    async def cursor(self, *, owner_id: int, context_id: int) -> int:
        async with aiosqlite.connect(self._database_path) as db:
            cursor = await db.execute(
                "SELECT last_message_id FROM business_cursors WHERE owner_id=? AND context_id=?",
                (owner_id, context_id),
            )
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def prepare(
        self,
        *,
        owner_id: int,
        context_id: int,
        expected_cursor: int,
        last_message_id: int,
        intents: list[BusinessIntent],
    ) -> bool:
        # The cursor and immutable plan commit together, before any remote write.
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "SELECT last_message_id FROM business_cursors WHERE owner_id=? AND context_id=?",
                (owner_id, context_id),
            )
            row = await cursor.fetchone()
            if (int(row[0]) if row else 0) != expected_cursor:
                return False
            for intent in intents:
                canonical = json.dumps(
                    [
                        owner_id,
                        context_id,
                        sorted(set(intent.source_message_ids)),
                        intent.action.model_dump(mode="json"),
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                )
                action_id = hashlib.sha256(canonical.encode()).hexdigest()
                await db.execute(
                    "INSERT OR IGNORE INTO business_actions (id, owner_id, context_id, payload) VALUES (?, ?, ?, ?)",
                    (
                        action_id,
                        owner_id,
                        context_id,
                        self._cipher.encrypt(intent.action.model_dump_json().encode()),
                    ),
                )
            await db.execute(
                "INSERT INTO business_cursors VALUES (?, ?, ?) ON CONFLICT(owner_id, context_id) DO UPDATE SET last_message_id=MAX(last_message_id, excluded.last_message_id)",
                (owner_id, context_id, last_message_id),
            )
            await db.commit()
        return True

    async def outstanding(
        self, *, owner_id: int, context_id: int
    ) -> list[StoredBusinessAction]:
        async with aiosqlite.connect(self._database_path) as db:
            cursor = await db.execute(
                "SELECT id, payload, status, result FROM business_actions WHERE owner_id=? AND context_id=? AND notified=0 ORDER BY rowid",
                (owner_id, context_id),
            )
            rows = await cursor.fetchall()
        return [
            StoredBusinessAction.model_validate(
                {
                    "id": row[0],
                    "action": json.loads(self._cipher.decrypt(row[1])),
                    "status": row[2],
                    "result": self._cipher.decrypt(row[3]).decode()
                    if row[3] is not None
                    else None,
                }
            )
            for row in rows
        ]

    async def claim(self, *, owner_id: int, action_id: str) -> bool:
        async with aiosqlite.connect(self._database_path) as db:
            cursor = await db.execute(
                "UPDATE business_actions SET status='running' WHERE owner_id=? AND id=? AND status='pending'",
                (owner_id, action_id),
            )
            await db.commit()
            return cursor.rowcount == 1

    async def complete(self, *, owner_id: int, action_id: str, result: str) -> None:
        # Completed save_note payloads are the durable, encrypted note archive.
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute(
                "UPDATE business_actions SET status='completed', result=? WHERE owner_id=? AND id=?",
                (self._cipher.encrypt(result.encode()), owner_id, action_id),
            )
            await db.commit()

    async def notified(self, *, owner_id: int, action_id: str) -> None:
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute(
                "UPDATE business_actions SET notified=1 WHERE owner_id=? AND id=? AND status='completed'",
                (owner_id, action_id),
            )
            await db.commit()
