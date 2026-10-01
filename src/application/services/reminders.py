import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from src.application.ports.reminders import ReminderStore
from src.domain.assistant.models import (
    CreateReminderAction,
    DeleteReminderAction,
    WatchCalendarAction,
)
from src.domain.reminders.models import (
    CalendarWatch,
    ScheduleError,
    StoredReminder,
    build_schedule,
    describe,
    next_fire,
)

MAX_REMINDERS = 100
DEFAULT_LEAD_MINUTES = 60
DEFAULT_INTERVAL_MINUTES = 60
LEAD_RANGE = (5, 24 * 60)
INTERVAL_RANGE = (5, 6 * 60)
_ID = re.compile(r"^#?(\d+)$")


def format_watch(watch: CalendarWatch | None) -> str:
    if watch is None or not watch.enabled:
        return "Слежение за календарём выключено."
    return (
        f"Слежу за календарём: проверяю каждые {watch.interval_minutes} мин и "
        f"предупреждаю о событиях за {watch.lead_minutes} мин до начала."
    )


class ReminderService:
    """What the owner asks of the reminders from chat; each answer is ready text."""

    def __init__(self, *, store: ReminderStore, timezone: str) -> None:
        self._store = store
        self._tz = ZoneInfo(timezone)

    def _local(self, moment: datetime) -> datetime:
        return moment.astimezone(self._tz)

    def _line(self, reminder: StoredReminder) -> str:
        return (
            f"#{reminder.id} {reminder.text} — {describe(reminder.schedule)} "
            f"(ближайшее: {self._local(reminder.next_run_at):%d.%m.%Y %H:%M})"
        )

    async def create(
        self, action: CreateReminderAction, *, user_id: int, now: datetime
    ) -> str:
        try:
            schedule = build_schedule(
                repeat=action.repeat,
                at=action.time,
                day=action.date,
                weekdays=action.weekdays,
                day_of_month=action.day_of_month,
                days_before=action.days_before,
            )
        except ScheduleError as error:
            return f"⏰ Не создал напоминание: {error}"
        first = next_fire(schedule, after=now, tz=self._tz)
        if first is None:
            return "⏰ Не создал напоминание: это время уже прошло."
        if len(await self._store.list_reminders(owner_id=user_id)) >= MAX_REMINDERS:
            return (
                f"⏰ Уже {MAX_REMINDERS} напоминаний — это предел. "
                "Удали ненужные и повтори."
            )
        reminder = await self._store.add(
            owner_id=user_id,
            text=action.text.strip(),
            schedule=schedule,
            next_run_at=first.astimezone(UTC),
        )
        return f"⏰ Напомню: {self._line(reminder)}"

    async def overview(self, *, user_id: int) -> str:
        reminders = await self._store.list_reminders(owner_id=user_id)
        watch = await self._store.get_watch(owner_id=user_id)
        lines = [f"⏰ Запланировано ({len(reminders)}):"] if reminders else []
        lines.extend(self._line(reminder) for reminder in reminders)
        if not reminders:
            lines.append("⏰ Напоминаний пока нет.")
        lines.append("")
        lines.append(f"👀 {format_watch(watch)}")
        return "\n".join(lines)

    async def delete(self, action: DeleteReminderAction, *, user_id: int) -> str:
        query = action.query.strip()
        reminders = await self._store.list_reminders(owner_id=user_id)
        by_number = _ID.match(query)
        if by_number:
            matches = [r for r in reminders if r.id == int(by_number.group(1))]
        else:
            needle = query.casefold()
            matches = [r for r in reminders if needle in r.text.casefold()]
        if not matches:
            return f"Напоминание «{query}» не найдено. Список — /reminders."
        if len(matches) > 1:
            # Cancelling is not confirmable: never guess which one is meant.
            listing = "\n".join(f"• {self._line(r)}" for r in matches)
            return f"Подходит несколько напоминаний. Уточни, какое удалить:\n{listing}"
        target = matches[0]
        await self._store.delete(owner_id=user_id, reminder_id=target.id)
        return f"🗑 Удалил напоминание: {target.text}"

    async def watch(
        self, action: WatchCalendarAction, *, user_id: int, now: datetime
    ) -> str:
        current = await self._store.get_watch(owner_id=user_id)
        lead = action.lead_minutes or (
            current.lead_minutes if current else DEFAULT_LEAD_MINUTES
        )
        interval = action.interval_minutes or (
            current.interval_minutes if current else DEFAULT_INTERVAL_MINUTES
        )
        if not LEAD_RANGE[0] <= lead <= LEAD_RANGE[1]:
            return (
                "👀 Предупреждать можно за "
                f"{LEAD_RANGE[0]}–{LEAD_RANGE[1]} минут до события."
            )
        if not INTERVAL_RANGE[0] <= interval <= INTERVAL_RANGE[1]:
            return (
                "👀 Календарь можно проверять раз в "
                f"{INTERVAL_RANGE[0]}–{INTERVAL_RANGE[1]} минут."
            )
        watch = CalendarWatch(
            owner_id=user_id,
            lead_minutes=lead,
            interval_minutes=interval,
            enabled=action.enabled,
            # Look right away: an event may already be close.
            next_check_at=now.astimezone(UTC),
        )
        await self._store.save_watch(watch)
        return f"👀 {format_watch(watch)}"
