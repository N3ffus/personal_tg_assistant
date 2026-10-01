from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from src.domain.reminders.models import CalendarWatch, Schedule, StoredReminder
from src.infrastructure.sqlite import connect, use_write_ahead_log


def _epoch(moment: datetime) -> int:
    return int(moment.timestamp())


def _moment(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, UTC)


class ReminderStorage:
    """Reminders, calendar-watch settings and the events already announced.

    The reminder text and schedule are personal, so they are stored encrypted
    like the rest of the user data; only the firing time stays readable to let
    SQLite find what is due.
    """

    def __init__(self, *, database_path: str, encryption_key: str) -> None:
        self._database_path = database_path
        self._cipher = Fernet(encryption_key.encode())

    async def initialize(self) -> None:
        Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)
        async with connect(self._database_path) as database:
            await use_write_ahead_log(database)
            await database.execute(
                """CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    next_run_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                )"""
            )
            await database.execute(
                "CREATE INDEX IF NOT EXISTS reminders_due ON reminders(next_run_at)"
            )
            await database.execute(
                """CREATE TABLE IF NOT EXISTS calendar_watches (
                    owner_id INTEGER PRIMARY KEY,
                    lead_minutes INTEGER NOT NULL,
                    interval_minutes INTEGER NOT NULL,
                    enabled INTEGER NOT NULL,
                    next_check_at INTEGER NOT NULL
                )"""
            )
            await database.execute(
                """CREATE TABLE IF NOT EXISTS calendar_announced (
                    owner_id INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    starts_at INTEGER NOT NULL,
                    announced_at INTEGER NOT NULL,
                    PRIMARY KEY (owner_id, event_id, starts_at)
                )"""
            )
            await database.commit()

    def _seal(self, text: str, schedule: Schedule) -> bytes:
        payload = f"{schedule.model_dump_json()}\n{text}"
        return self._cipher.encrypt(payload.encode())

    def _reminder(self, row: Sequence[Any]) -> StoredReminder:
        schedule_json, _, text = self._cipher.decrypt(row[2]).decode().partition("\n")
        return StoredReminder(
            id=row[0],
            owner_id=row[1],
            text=text,
            schedule=Schedule.model_validate_json(schedule_json),
            next_run_at=_moment(row[3]),
        )

    async def add(
        self, *, owner_id: int, text: str, schedule: Schedule, next_run_at: datetime
    ) -> StoredReminder:
        async with connect(self._database_path) as database:
            cursor = await database.execute(
                """INSERT INTO reminders (owner_id, payload, next_run_at, created_at)
                VALUES (?, ?, ?, ?) RETURNING id""",
                (
                    owner_id,
                    self._seal(text, schedule),
                    _epoch(next_run_at),
                    _epoch(datetime.now(UTC)),
                ),
            )
            row = await cursor.fetchone()
            await database.commit()
        assert row is not None
        return StoredReminder(
            id=row[0],
            owner_id=owner_id,
            text=text,
            schedule=schedule,
            next_run_at=_moment(_epoch(next_run_at)),
        )

    async def list_reminders(self, *, owner_id: int) -> list[StoredReminder]:
        async with connect(self._database_path) as database:
            cursor = await database.execute(
                """SELECT id, owner_id, payload, next_run_at FROM reminders
                WHERE owner_id=? ORDER BY next_run_at, id""",
                (owner_id,),
            )
            rows = await cursor.fetchall()
        return [self._reminder(row) for row in rows]

    async def delete(self, *, owner_id: int, reminder_id: int) -> bool:
        async with connect(self._database_path) as database:
            cursor = await database.execute(
                "DELETE FROM reminders WHERE id=? AND owner_id=?",
                (reminder_id, owner_id),
            )
            await database.commit()
        return cursor.rowcount == 1

    async def due(self, *, now: datetime) -> list[StoredReminder]:
        async with connect(self._database_path) as database:
            cursor = await database.execute(
                """SELECT id, owner_id, payload, next_run_at FROM reminders
                WHERE next_run_at<=? ORDER BY next_run_at, id""",
                (_epoch(now),),
            )
            rows = await cursor.fetchall()
        return [self._reminder(row) for row in rows]

    async def advance(
        self, *, reminder_id: int, expected: datetime, next_run_at: datetime | None
    ) -> bool:
        async with connect(self._database_path) as database:
            if next_run_at is None:
                cursor = await database.execute(
                    "DELETE FROM reminders WHERE id=? AND next_run_at=?",
                    (reminder_id, _epoch(expected)),
                )
            else:
                cursor = await database.execute(
                    "UPDATE reminders SET next_run_at=? WHERE id=? AND next_run_at=?",
                    (_epoch(next_run_at), reminder_id, _epoch(expected)),
                )
            await database.commit()
        return cursor.rowcount == 1

    async def get_watch(self, *, owner_id: int) -> CalendarWatch | None:
        async with connect(self._database_path) as database:
            cursor = await database.execute(
                """SELECT owner_id, lead_minutes, interval_minutes, enabled,
                next_check_at FROM calendar_watches WHERE owner_id=?""",
                (owner_id,),
            )
            row = await cursor.fetchone()
        return self._watch(row) if row else None

    async def save_watch(self, watch: CalendarWatch) -> None:
        async with connect(self._database_path) as database:
            await database.execute(
                """INSERT INTO calendar_watches
                (owner_id, lead_minutes, interval_minutes, enabled, next_check_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    lead_minutes=excluded.lead_minutes,
                    interval_minutes=excluded.interval_minutes,
                    enabled=excluded.enabled,
                    next_check_at=excluded.next_check_at""",
                (
                    watch.owner_id,
                    watch.lead_minutes,
                    watch.interval_minutes,
                    int(watch.enabled),
                    _epoch(watch.next_check_at),
                ),
            )
            await database.commit()

    async def due_watches(self, *, now: datetime) -> list[CalendarWatch]:
        async with connect(self._database_path) as database:
            cursor = await database.execute(
                """SELECT owner_id, lead_minutes, interval_minutes, enabled,
                next_check_at FROM calendar_watches
                WHERE enabled=1 AND next_check_at<=?""",
                (_epoch(now),),
            )
            rows = await cursor.fetchall()
        return [self._watch(row) for row in rows]

    @staticmethod
    def _watch(row: Sequence[Any]) -> CalendarWatch:
        return CalendarWatch(
            owner_id=row[0],
            lead_minutes=row[1],
            interval_minutes=row[2],
            enabled=bool(row[3]),
            next_check_at=_moment(row[4]),
        )

    async def was_announced(
        self, *, owner_id: int, event_id: str, starts_at: datetime
    ) -> bool:
        async with connect(self._database_path) as database:
            cursor = await database.execute(
                """SELECT 1 FROM calendar_announced
                WHERE owner_id=? AND event_id=? AND starts_at=?""",
                (owner_id, event_id, _epoch(starts_at)),
            )
            return await cursor.fetchone() is not None

    async def mark_announced(
        self, *, owner_id: int, event_id: str, starts_at: datetime, now: datetime
    ) -> None:
        async with connect(self._database_path) as database:
            await database.execute(
                """INSERT OR IGNORE INTO calendar_announced
                (owner_id, event_id, starts_at, announced_at) VALUES (?, ?, ?, ?)""",
                (owner_id, event_id, _epoch(starts_at), _epoch(now)),
            )
            await database.commit()

    async def forget_announced_before(self, *, moment: datetime) -> None:
        async with connect(self._database_path) as database:
            await database.execute(
                "DELETE FROM calendar_announced WHERE starts_at<?", (_epoch(moment),)
            )
            await database.commit()
