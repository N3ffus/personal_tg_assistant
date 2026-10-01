from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from src.domain.reminders.models import CalendarWatch, build_schedule
from src.infrastructure.reminders.storage import ReminderStorage

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
OWNER = 7

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> ReminderStorage:
    store = ReminderStorage(
        database_path=str(tmp_path / "assistant.db"),
        encryption_key=Fernet.generate_key().decode(),
    )
    await store.initialize()
    return store


async def test_a_reminder_round_trips_and_is_encrypted_at_rest(
    storage: ReminderStorage, tmp_path: Path
) -> None:
    schedule = build_schedule(repeat="yearly", day="2000-05-14", days_before=1)

    added = await storage.add(
        owner_id=OWNER, text="ДР у Маши", schedule=schedule, next_run_at=NOW
    )
    listed = await storage.list_reminders(owner_id=OWNER)

    assert [r.id for r in listed] == [added.id]
    assert listed[0].text == "ДР у Маши"
    assert listed[0].schedule == schedule
    assert listed[0].next_run_at == NOW
    assert "Маши".encode() not in (tmp_path / "assistant.db").read_bytes()


async def test_reminders_are_scoped_to_their_owner(storage: ReminderStorage) -> None:
    schedule = build_schedule(repeat="daily")
    mine = await storage.add(
        owner_id=OWNER, text="моё", schedule=schedule, next_run_at=NOW
    )
    await storage.add(owner_id=8, text="чужое", schedule=schedule, next_run_at=NOW)

    assert [r.text for r in await storage.list_reminders(owner_id=OWNER)] == ["моё"]
    assert await storage.delete(owner_id=8, reminder_id=mine.id) is False
    assert await storage.delete(owner_id=OWNER, reminder_id=mine.id) is True
    assert await storage.list_reminders(owner_id=OWNER) == []


async def test_only_due_reminders_are_returned_oldest_first(
    storage: ReminderStorage,
) -> None:
    schedule = build_schedule(repeat="daily")
    later = await storage.add(
        owner_id=OWNER,
        text="later",
        schedule=schedule,
        next_run_at=NOW + timedelta(hours=1),
    )
    second = await storage.add(
        owner_id=OWNER, text="second", schedule=schedule, next_run_at=NOW
    )
    first = await storage.add(
        owner_id=OWNER,
        text="first",
        schedule=schedule,
        next_run_at=NOW - timedelta(hours=1),
    )

    due = await storage.due(now=NOW)

    assert [r.id for r in due] == [first.id, second.id]
    assert later.id not in {r.id for r in due}


async def test_advance_moves_a_reminder_once(storage: ReminderStorage) -> None:
    schedule = build_schedule(repeat="daily")
    reminder = await storage.add(
        owner_id=OWNER, text="x", schedule=schedule, next_run_at=NOW
    )
    tomorrow = NOW + timedelta(days=1)

    assert await storage.advance(
        reminder_id=reminder.id, expected=NOW, next_run_at=tomorrow
    )
    # The same firing cannot be claimed twice.
    assert not await storage.advance(
        reminder_id=reminder.id, expected=NOW, next_run_at=tomorrow
    )
    assert (await storage.list_reminders(owner_id=OWNER))[0].next_run_at == tomorrow


async def test_advancing_to_nothing_removes_a_finished_reminder(
    storage: ReminderStorage,
) -> None:
    reminder = await storage.add(
        owner_id=OWNER,
        text="x",
        schedule=build_schedule(repeat="once", day="2026-10-01"),
        next_run_at=NOW,
    )

    assert await storage.advance(
        reminder_id=reminder.id, expected=NOW, next_run_at=None
    )
    assert await storage.list_reminders(owner_id=OWNER) == []


async def test_watch_settings_are_upserted_and_found_when_due(
    storage: ReminderStorage,
) -> None:
    assert await storage.get_watch(owner_id=OWNER) is None
    watch = CalendarWatch(
        owner_id=OWNER,
        lead_minutes=60,
        interval_minutes=30,
        enabled=True,
        next_check_at=NOW,
    )
    await storage.save_watch(watch)
    assert await storage.due_watches(now=NOW) == [watch]
    assert await storage.due_watches(now=NOW - timedelta(minutes=1)) == []

    paused = CalendarWatch(
        owner_id=OWNER,
        lead_minutes=45,
        interval_minutes=30,
        enabled=False,
        next_check_at=NOW,
    )
    await storage.save_watch(paused)

    assert await storage.get_watch(owner_id=OWNER) == paused
    assert await storage.due_watches(now=NOW) == []


async def test_announced_events_are_remembered_per_start_time(
    storage: ReminderStorage,
) -> None:
    start = NOW + timedelta(hours=2)
    await storage.mark_announced(
        owner_id=OWNER, event_id="e1", starts_at=start, now=NOW
    )
    await storage.mark_announced(
        owner_id=OWNER, event_id="e1", starts_at=start, now=NOW
    )

    assert await storage.was_announced(owner_id=OWNER, event_id="e1", starts_at=start)
    # A moved event is a new announcement.
    assert not await storage.was_announced(
        owner_id=OWNER, event_id="e1", starts_at=start + timedelta(hours=1)
    )

    await storage.forget_announced_before(moment=start + timedelta(minutes=1))
    assert not await storage.was_announced(
        owner_id=OWNER, event_id="e1", starts_at=start
    )
