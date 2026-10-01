from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from src.application.ports.calendar import CalendarError, CalendarNotConnectedError
from src.application.services.calendar_watch import CalendarWatcher, humanize_minutes
from src.application.services.reminder_scheduler import (
    GRACE,
    RETRY_DELAY,
    ReminderScheduler,
)
from src.domain.calendar.models import CalendarEvent
from src.domain.reminders.models import CalendarWatch, build_schedule
from src.infrastructure.reminders.storage import ReminderStorage

pytestmark = pytest.mark.asyncio

OWNER = 7
# 09:00 in Moscow.
NOW = datetime(2026, 10, 1, 6, 0, tzinfo=UTC)
TIMEZONE = "Europe/Moscow"


class Outbox:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.failures = 0

    async def send(self, *, user_id: int, text: str) -> None:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("telegram is down")
        assert user_id == OWNER
        self.sent.append(text)


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> ReminderStorage:
    store = ReminderStorage(
        database_path=str(tmp_path / "assistant.db"),
        encryption_key=Fernet.generate_key().decode(),
    )
    await store.initialize()
    return store


def event(title: str, starts_at: datetime, event_id: str = "e1") -> CalendarEvent:
    return CalendarEvent(
        event_id=event_id,
        title=title,
        starts_at=starts_at,
        ends_at=starts_at + timedelta(hours=1),
        html_link=None,
    )


def calendar_with(*events: CalendarEvent) -> Any:
    return SimpleNamespace(list_events=AsyncMock(return_value=list(events)))


def scheduler(
    storage: ReminderStorage, outbox: Outbox, calendar: Any | None = None
) -> ReminderScheduler:
    watcher = (
        CalendarWatcher(
            calendar=calendar, store=storage, notifier=outbox, timezone=TIMEZONE
        )
        if calendar is not None
        else None
    )
    return ReminderScheduler(
        store=storage, notifier=outbox, timezone=TIMEZONE, watcher=watcher
    )


async def test_a_due_daily_reminder_is_sent_once_and_moved_to_tomorrow(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    await storage.add(
        owner_id=OWNER,
        text="Выпить таблетки",
        schedule=build_schedule(repeat="daily", at="09:00"),
        next_run_at=NOW,
    )

    await scheduler(storage, outbox).tick(NOW)
    await scheduler(storage, outbox).tick(NOW + timedelta(seconds=30))

    assert outbox.sent == ["🔔 Выпить таблетки"]
    (reminder,) = await storage.list_reminders(owner_id=OWNER)
    assert reminder.next_run_at == NOW + timedelta(days=1)


async def test_a_one_time_reminder_disappears_after_it_is_sent(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    await storage.add(
        owner_id=OWNER,
        text="Позвонить в банк",
        schedule=build_schedule(repeat="once", day="2026-10-01", at="09:00"),
        next_run_at=NOW,
    )

    await scheduler(storage, outbox).tick(NOW)

    assert outbox.sent == ["🔔 Позвонить в банк"]
    assert await storage.list_reminders(owner_id=OWNER) == []


async def test_a_reminder_that_is_not_due_stays_quiet(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    await storage.add(
        owner_id=OWNER,
        text="потом",
        schedule=build_schedule(repeat="daily"),
        next_run_at=NOW + timedelta(minutes=1),
    )

    await scheduler(storage, outbox).tick(NOW)

    assert outbox.sent == []


async def test_a_failed_send_is_retried_later_and_not_lost(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    outbox.failures = 1
    await storage.add(
        owner_id=OWNER,
        text="Не потерять",
        schedule=build_schedule(repeat="once", day="2026-10-01", at="09:00"),
        next_run_at=NOW,
    )
    runner = scheduler(storage, outbox)

    await runner.tick(NOW)
    assert outbox.sent == []
    assert len(await storage.list_reminders(owner_id=OWNER)) == 1

    # Not hammered before the retry delay…
    await runner.tick(NOW + timedelta(seconds=30))
    assert outbox.sent == []

    # …and delivered once it passes.
    await runner.tick(NOW + RETRY_DELAY + timedelta(seconds=1))
    assert outbox.sent == ["🔔 Не потерять"]
    assert await storage.list_reminders(owner_id=OWNER) == []


async def test_a_reminder_missed_during_downtime_is_skipped_not_sent_late(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    await storage.add(
        owner_id=OWNER,
        text="Давно",
        schedule=build_schedule(repeat="daily", at="09:00"),
        next_run_at=NOW - GRACE - timedelta(hours=1),
    )

    await scheduler(storage, outbox).tick(NOW)

    assert outbox.sent == []
    (reminder,) = await storage.list_reminders(owner_id=OWNER)
    assert reminder.next_run_at > NOW


async def test_several_missed_days_send_a_single_message(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    await storage.add(
        owner_id=OWNER,
        text="Зарядка",
        schedule=build_schedule(repeat="daily", at="09:00"),
        next_run_at=NOW - timedelta(hours=3),
    )

    await scheduler(storage, outbox).tick(NOW)

    assert outbox.sent == ["🔔 Зарядка"]


async def test_the_calendar_look_ahead_announces_an_event_once(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    start = NOW + timedelta(minutes=70)
    calendar = calendar_with(event("Стоматолог", start))
    await storage.save_watch(
        CalendarWatch(
            owner_id=OWNER,
            lead_minutes=60,
            interval_minutes=60,
            enabled=True,
            next_check_at=NOW,
        )
    )
    runner = scheduler(storage, outbox, calendar)

    await runner.tick(NOW)
    # The next check is a full interval away, and the event is not repeated.
    await runner.tick(NOW + timedelta(minutes=30))
    await runner.tick(NOW + timedelta(minutes=60))

    assert outbox.sent == ["📅 Скоро: Стоматолог — 10:10, через 1 ч 10 мин"]
    assert calendar.list_events.await_count == 2
    query = calendar.list_events.await_args_list[0].kwargs["query"]
    assert query.date_from == NOW
    assert query.date_to == NOW + timedelta(minutes=120)
    assert query.all_day is False
    watch = await storage.get_watch(owner_id=OWNER)
    assert watch is not None
    assert watch.next_check_at == NOW + timedelta(minutes=120)


async def test_a_moved_event_is_announced_again(storage: ReminderStorage) -> None:
    outbox = Outbox()
    await storage.save_watch(
        CalendarWatch(
            owner_id=OWNER,
            lead_minutes=60,
            interval_minutes=30,
            enabled=True,
            next_check_at=NOW,
        )
    )
    first = calendar_with(event("Созвон", NOW + timedelta(minutes=40)))
    await scheduler(storage, outbox, first).tick(NOW)
    moved = calendar_with(event("Созвон", NOW + timedelta(minutes=100)))

    await scheduler(storage, outbox, moved).tick(NOW + timedelta(minutes=30))

    assert len(outbox.sent) == 2


async def test_events_already_started_are_not_announced(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    await storage.save_watch(
        CalendarWatch(
            owner_id=OWNER,
            lead_minutes=60,
            interval_minutes=60,
            enabled=True,
            next_check_at=NOW,
        )
    )
    calendar = calendar_with(event("Идёт сейчас", NOW - timedelta(minutes=10)))

    await scheduler(storage, outbox, calendar).tick(NOW)

    assert outbox.sent == []


@pytest.mark.parametrize("error", [CalendarNotConnectedError(), CalendarError()])
async def test_calendar_trouble_is_not_fatal_and_waits_for_the_next_interval(
    storage: ReminderStorage, error: Exception
) -> None:
    outbox = Outbox()
    await storage.save_watch(
        CalendarWatch(
            owner_id=OWNER,
            lead_minutes=60,
            interval_minutes=60,
            enabled=True,
            next_check_at=NOW,
        )
    )
    calendar = SimpleNamespace(list_events=AsyncMock(side_effect=error))

    await scheduler(storage, outbox, calendar).tick(NOW)

    assert outbox.sent == []
    watch = await storage.get_watch(owner_id=OWNER)
    assert watch is not None
    assert watch.next_check_at == NOW + timedelta(minutes=60)


async def test_a_disabled_watch_never_touches_the_calendar(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    await storage.save_watch(
        CalendarWatch(
            owner_id=OWNER,
            lead_minutes=60,
            interval_minutes=60,
            enabled=False,
            next_check_at=NOW,
        )
    )
    calendar = calendar_with()

    await scheduler(storage, outbox, calendar).tick(NOW)

    calendar.list_events.assert_not_awaited()


async def test_an_announcement_that_fails_to_send_is_tried_again(
    storage: ReminderStorage,
) -> None:
    outbox = Outbox()
    outbox.failures = 1
    await storage.save_watch(
        CalendarWatch(
            owner_id=OWNER,
            lead_minutes=60,
            interval_minutes=30,
            enabled=True,
            next_check_at=NOW,
        )
    )
    calendar = calendar_with(event("Важное", NOW + timedelta(minutes=45)))
    runner = scheduler(storage, outbox, calendar)

    await runner.tick(NOW)
    await runner.tick(NOW + timedelta(minutes=30))

    assert len(outbox.sent) == 1


async def test_the_loop_survives_a_failing_tick(storage: ReminderStorage) -> None:
    outbox = Outbox()
    runner = ReminderScheduler(
        store=storage,
        notifier=outbox,
        timezone=TIMEZONE,
        tick_seconds=0,
    )
    calls = 0

    async def broken(now: datetime) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("database is locked")
        raise KeyboardInterrupt  # stops the loop for the test

    runner.tick = broken  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt):
        await runner.run()

    assert calls == 2


@pytest.mark.parametrize(
    ("minutes", "text"),
    [(0, "меньше минуты"), (45, "45 мин"), (60, "1 ч"), (135, "2 ч 15 мин")],
)
async def test_humanized_minutes(minutes: int, text: str) -> None:
    assert humanize_minutes(minutes) == text
