from pathlib import Path
from typing import Literal

import aiosqlite
from cryptography.fernet import Fernet

from src.domain.assistant.context import ChatContext, ContextChat


class ContextStorage:
    def __init__(self, *, database_path: str, encryption_key: str) -> None:
        self._database_path = database_path
        self._cipher = Fernet(encryption_key.encode())

    async def initialize(self) -> None:
        Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._database_path) as database:
            await database.execute(
                """CREATE TABLE IF NOT EXISTS chat_contexts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    chat_id INTEGER NOT NULL,
                    title BLOB NOT NULL,
                    payload BLOB NOT NULL,
                    UNIQUE(owner_id, kind, chat_id)
                )"""
            )
            await database.commit()

    async def ensure_chat(
        self,
        *,
        owner_id: int,
        kind: Literal["bot", "business"],
        chat_id: int,
        title: str,
    ) -> ContextChat:
        title = title[:150]
        async with aiosqlite.connect(self._database_path) as database:
            cursor = await database.execute(
                """INSERT INTO chat_contexts (owner_id, kind, chat_id, title, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, kind, chat_id) DO UPDATE SET title=excluded.title
                RETURNING id""",
                (
                    owner_id,
                    kind,
                    chat_id,
                    self._cipher.encrypt(title.encode()),
                    self._cipher.encrypt(ChatContext().model_dump_json().encode()),
                ),
            )
            row = await cursor.fetchone()
            await database.commit()
        assert row is not None
        return ContextChat(
            id=row[0], owner_id=owner_id, kind=kind, chat_id=chat_id, title=title
        )

    async def list_chats(self, *, owner_id: int) -> list[ContextChat]:
        async with aiosqlite.connect(self._database_path) as database:
            cursor = await database.execute(
                "SELECT id, kind, chat_id, title FROM chat_contexts WHERE owner_id=? ORDER BY kind, id",
                (owner_id,),
            )
            rows = await cursor.fetchall()
        return [
            ContextChat(
                id=row[0],
                owner_id=owner_id,
                kind=row[1],
                chat_id=row[2],
                title=self._cipher.decrypt(row[3]).decode(),
            )
            for row in rows
        ]

    async def load(self, *, owner_id: int, context_id: int) -> ChatContext | None:
        async with aiosqlite.connect(self._database_path) as database:
            cursor = await database.execute(
                "SELECT payload FROM chat_contexts WHERE owner_id=? AND id=?",
                (owner_id, context_id),
            )
            row = await cursor.fetchone()
        return (
            ChatContext.model_validate_json(self._cipher.decrypt(row[0]))
            if row
            else None
        )

    async def save(
        self, *, owner_id: int, context_id: int, context: ChatContext
    ) -> None:
        async with aiosqlite.connect(self._database_path) as database:
            await database.execute(
                "UPDATE chat_contexts SET payload=? WHERE owner_id=? AND id=?",
                (
                    self._cipher.encrypt(context.model_dump_json().encode()),
                    owner_id,
                    context_id,
                ),
            )
            await database.commit()
