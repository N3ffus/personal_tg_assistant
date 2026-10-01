import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from src.application.ports.reminders import Notifier, ReminderStore
from src.application.services.calendar_watch import CalendarWatcher
from src.domain.reminders.models import StoredReminder, next_fire

logger = logging.getLogger(__name__)

TICK_SECONDS = 30
# A reminder this overdue (the bot was down) is dropped rather than delivered
# as a stale message; recurring ones simply move to their next firing.
GRACE = timedelta(hours=24)
RETRY_DELAY = timedelta(minutes=5)


class ReminderScheduler:
    """Sends due reminders and runs the calendar look-ahead, inside the bot process."""

    def __init__(
        self,
        *,
        store: ReminderStore,
        notifier: Notifier,
        timezone: str,
        watcher: CalendarWatcher | None = None,
        tick_seconds: float = TICK_SECONDS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._notifier = notifier
        self._tz = ZoneInfo(timezone)
        self._watcher = watcher
        self._tick_seconds = tick_seconds
        self._clock = clock
        self._retry_after: dict[int, datetime] = {}

    async def run(self) -> None:
        logger.info("reminders.scheduler.started")
        while True:
            try:
                await self.tick(self._clock())
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failing tick must never take the bot down with it.
                logger.exception("reminders.scheduler.tick_failed")
            await asyncio.sleep(self._tick_seconds)

    async def tick(self, now: datetime) -> None:
        await self._deliver_due(now)
        await self._check_calendars(now)

    async def _deliver_due(self, now: datetime) -> None:
        for reminder in await self._store.due(now=now):
            if self._retry_after.get(reminder.id, now) > now:
                continue
            if now - reminder.next_run_at > GRACE:
                logger.warning(
                    "reminders.missed id=%s due=%s",
                    reminder.id,
                    reminder.next_run_at.isoformat(),
                )
            else:
                try:
                    await self._notifier.send(
                        user_id=reminder.owner_id, text=f"🔔 {reminder.text}"
                    )
                except Exception:
                    logger.warning(
                        "reminders.send_failed id=%s", reminder.id, exc_info=True
                    )
                    self._retry_after[reminder.id] = now + RETRY_DELAY
                    continue
            self._retry_after.pop(reminder.id, None)
            await self._advance(reminder, now)

    async def _advance(self, reminder: StoredReminder, now: datetime) -> None:
        following = next_fire(reminder.schedule, after=now, tz=self._tz)
        moved = await self._store.advance(
            reminder_id=reminder.id,
            expected=reminder.next_run_at,
            next_run_at=following.astimezone(UTC) if following else None,
        )
        logger.info(
            "reminders.delivered id=%s next=%s moved=%s",
            reminder.id,
            following.isoformat() if following else None,
            moved,
        )

    async def _check_calendars(self, now: datetime) -> None:
        if self._watcher is None:
            return
        for watch in await self._store.due_watches(now=now):
            try:
                await self._watcher.check(watch, now=now)
            finally:
                # Whatever happened, wait a full interval before looking again.
                # The settings are re-read: the owner may have retuned or
                # stopped the watch while the check was running.
                current = await self._store.get_watch(owner_id=watch.owner_id)
                if current is not None:
                    await self._store.save_watch(
                        replace(
                            current,
                            next_check_at=now
                            + timedelta(minutes=current.interval_minutes),
                        )
                    )
