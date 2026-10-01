import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from src.application.ports.calendar import (
    CalendarClient,
    CalendarError,
    CalendarNotConnectedError,
)
from src.application.ports.reminders import Notifier, ReminderStore
from src.domain.assistant.retrieval import EventQuery, RetrievalLimitError
from src.domain.calendar.models import CalendarEvent
from src.domain.reminders.models import CalendarWatch

logger = logging.getLogger(__name__)

# Announcements are kept a little past the event so a check never repeats one.
ANNOUNCED_RETENTION = timedelta(days=2)
MAX_EVENTS_PER_CHECK = 20


def humanize_minutes(minutes: int) -> str:
    if minutes < 1:
        return "меньше минуты"
    hours, rest = divmod(minutes, 60)
    if not hours:
        return f"{rest} мин"
    return f"{hours} ч {rest} мин" if rest else f"{hours} ч"


class CalendarWatcher:
    """Looks ahead in Google Calendar and tells the owner what starts soon.

    A check runs every ``interval_minutes``, so an event is announced at the
    last check that still precedes ``lead_minutes`` before it: the window is
    ``lead + interval`` and the message gives the real time left. Each event
    start is announced once; a moved event is announced again.
    """

    def __init__(
        self,
        *,
        calendar: CalendarClient,
        store: ReminderStore,
        notifier: Notifier,
        timezone: str,
    ) -> None:
        self._calendar = calendar
        self._store = store
        self._notifier = notifier
        self._tz = ZoneInfo(timezone)

    async def check(self, watch: CalendarWatch, *, now: datetime) -> None:
        horizon = now + timedelta(minutes=watch.lead_minutes + watch.interval_minutes)
        try:
            events = await self._calendar.list_events(
                user_id=watch.owner_id,
                now=now,
                query=EventQuery(
                    date_from=now,
                    date_to=horizon,
                    # An all-day entry has no start time to warn about.
                    all_day=False,
                    page_size=MAX_EVENTS_PER_CHECK,
                ),
            )
        except CalendarNotConnectedError:
            logger.info("calendar.watch.skipped reason=not_connected")
            return
        except (CalendarError, RetrievalLimitError):
            logger.warning("calendar.watch.failed", exc_info=True)
            return

        await self._store.forget_announced_before(moment=now - ANNOUNCED_RETENTION)
        for event in sorted(events, key=lambda item: item.starts_at):
            if event.starts_at <= now:
                continue
            if await self._store.was_announced(
                owner_id=watch.owner_id,
                event_id=event.event_id,
                starts_at=event.starts_at,
            ):
                continue
            try:
                await self._notifier.send(
                    user_id=watch.owner_id, text=self._message(event, now=now)
                )
            except Exception:
                # Not marked, so the next check tries again.
                logger.warning("calendar.watch.send_failed", exc_info=True)
                return
            await self._store.mark_announced(
                owner_id=watch.owner_id,
                event_id=event.event_id,
                starts_at=event.starts_at,
                now=now.astimezone(UTC),
            )
            logger.info("calendar.watch.announced event=%s", event.event_id)

    def _message(self, event: CalendarEvent, *, now: datetime) -> str:
        minutes = int((event.starts_at - now).total_seconds() // 60)
        start = event.starts_at.astimezone(self._tz)
        return f"📅 Скоро: {event.title} — {start:%H:%M}, через {humanize_minutes(minutes)}"
