import re
from calendar import monthrange
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

Repeat = Literal["once", "daily", "weekly", "monthly", "yearly"]
REPEATS: tuple[str, ...] = ("once", "daily", "weekly", "monthly", "yearly")
DEFAULT_TIME = time(9, 0)
MAX_DAYS_BEFORE = 30

WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


class ScheduleError(ValueError):
    """The requested schedule is incomplete or impossible; the text is for the user."""


class Schedule(BaseModel):
    """When a reminder fires, in the owner's timezone.

    The *event day* is what the schedule names (the birthday, the meeting); the
    reminder fires ``days_before`` days earlier, at ``at``.
    """

    model_config = ConfigDict(frozen=True)

    repeat: Repeat
    at: time = DEFAULT_TIME
    # once: the event day; yearly: only its month and day are used.
    day: date | None = None
    weekdays: tuple[int, ...] = ()
    day_of_month: int | None = None
    days_before: int = 0


@dataclass(frozen=True, slots=True)
class StoredReminder:
    id: int
    owner_id: int
    text: str
    schedule: Schedule
    next_run_at: datetime  # aware, UTC


@dataclass(frozen=True, slots=True)
class CalendarWatch:
    owner_id: int
    lead_minutes: int
    interval_minutes: int
    enabled: bool
    next_check_at: datetime  # aware, UTC


_TIME = re.compile(r"^(\d{1,2})(?:[:.\-\s]?(\d{2}))?$")


def parse_time(value: str | None) -> time:
    if value is None or not value.strip():
        return DEFAULT_TIME
    match = _TIME.match(value.strip())
    if match is None:
        raise ScheduleError(f"Не понял время «{value}». Укажи его как ЧЧ:ММ.")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    if hour > 23 or minute > 59:
        raise ScheduleError(f"Не понял время «{value}». Укажи его как ЧЧ:ММ.")
    return time(hour, minute)


def parse_day(value: str | None) -> date | None:
    if value is None or not value.strip():
        return None
    try:
        # A model may send a full timestamp: only the calendar day matters.
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        raise ScheduleError(
            f"Не понял дату «{value}». Укажи её как ГГГГ-ММ-ДД."
        ) from None


def build_schedule(
    *,
    repeat: str,
    at: str | None = None,
    day: str | None = None,
    weekdays: Sequence[int] = (),
    day_of_month: int | None = None,
    days_before: int = 0,
) -> Schedule:
    if repeat not in REPEATS:
        raise ScheduleError("Не понял, как часто напоминать.")
    when = parse_time(at)
    on = parse_day(day)
    if not 0 <= days_before <= MAX_DAYS_BEFORE:
        raise ScheduleError(
            f"Напоминать заранее можно не больше чем за {MAX_DAYS_BEFORE} дней."
        )
    if repeat == "daily":
        return Schedule(repeat="daily", at=when)
    if repeat == "once" or repeat == "yearly":
        if on is None:
            raise ScheduleError("Укажи дату.")
        return Schedule(
            repeat=repeat,  # type: ignore[arg-type]
            at=when,
            day=on,
            days_before=days_before,
        )
    if repeat == "weekly":
        chosen = tuple(sorted(set(weekdays)))
        if not chosen and on is not None:
            chosen = (on.weekday(),)
        if not chosen or any(not 0 <= number <= 6 for number in chosen):
            raise ScheduleError("Укажи дни недели.")
        return Schedule(repeat="weekly", at=when, weekdays=chosen, days_before=0)
    number = day_of_month if day_of_month is not None else (on.day if on else None)
    if number is None or not 1 <= number <= 31:
        raise ScheduleError("Укажи число месяца.")
    return Schedule(
        repeat="monthly", at=when, day_of_month=number, days_before=days_before
    )


def _month_day(year: int, month: int, day: int) -> date:
    # 29 February falls on the 28th in a common year; 31st on a short month's last.
    return date(year, month, min(day, monthrange(year, month)[1]))


def _event_days(schedule: Schedule, start: date) -> Iterator[date]:
    """Event days on or after ``start`` in increasing order."""
    if schedule.repeat == "once":
        if schedule.day is not None and schedule.day >= start:
            yield schedule.day
    elif schedule.repeat in ("daily", "weekly"):
        current = start
        while True:
            if schedule.repeat == "daily" or current.weekday() in schedule.weekdays:
                yield current
            current += timedelta(days=1)
    elif schedule.repeat == "monthly":
        assert schedule.day_of_month is not None
        year, month = start.year, start.month
        while True:
            candidate = _month_day(year, month, schedule.day_of_month)
            if candidate >= start:
                yield candidate
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    else:
        assert schedule.day is not None
        year = start.year
        while True:
            candidate = _month_day(year, schedule.day.month, schedule.day.day)
            if candidate >= start:
                yield candidate
            year += 1


def next_fire(schedule: Schedule, *, after: datetime, tz: ZoneInfo) -> datetime | None:
    """The first firing strictly later than ``after``; ``None`` once it has passed."""
    local = after.astimezone(tz)
    for event_day in _event_days(schedule, local.date()):
        fire = datetime.combine(
            event_day - timedelta(days=schedule.days_before), schedule.at, tzinfo=tz
        )
        if fire > local:
            return fire
    return None


def _lead(days_before: int) -> str:
    if not days_before:
        return ""
    return "за день до " if days_before == 1 else f"за {days_before} дн. до "


def describe(schedule: Schedule) -> str:
    when = f"{schedule.at:%H:%M}"
    lead = _lead(schedule.days_before)
    if schedule.repeat == "once":
        assert schedule.day is not None
        return f"один раз, {lead}{schedule.day:%d.%m.%Y}, в {when}"
    if schedule.repeat == "daily":
        return f"каждый день в {when}"
    if schedule.repeat == "weekly":
        names = ", ".join(WEEKDAYS[number] for number in schedule.weekdays)
        return f"еженедельно ({names}) в {when}"
    if schedule.repeat == "monthly":
        return f"каждый месяц, {lead}{schedule.day_of_month}-го числа, в {when}"
    assert schedule.day is not None
    return (
        f"каждый год, {lead}{schedule.day.day} {MONTHS[schedule.day.month - 1]}, "
        f"в {when}"
    )
