from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.domain.reminders.models import (
    ScheduleError,
    build_schedule,
    describe,
    next_fire,
    parse_time,
)

MOSCOW = ZoneInfo("Europe/Moscow")


def moscow(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MOSCOW)


def test_birthday_reminder_fires_the_day_before_at_nine() -> None:
    schedule = build_schedule(
        repeat="yearly", at="09:00", day="2000-05-14", days_before=1
    )

    fire = next_fire(schedule, after=moscow(2026, 10, 1, 12), tz=MOSCOW)

    assert fire == moscow(2027, 5, 13, 9)


def test_birthday_this_year_is_still_ahead() -> None:
    schedule = build_schedule(repeat="yearly", day="1990-05-14", days_before=1)

    assert next_fire(schedule, after=moscow(2026, 3, 1), tz=MOSCOW) == moscow(
        2026, 5, 13, 9
    )


def test_a_firing_exactly_now_is_not_next() -> None:
    schedule = build_schedule(repeat="daily", at="09:00")

    assert next_fire(schedule, after=moscow(2026, 10, 1, 9), tz=MOSCOW) == moscow(
        2026, 10, 2, 9
    )
    assert next_fire(schedule, after=moscow(2026, 10, 1, 8, 59), tz=MOSCOW) == moscow(
        2026, 10, 1, 9
    )


def test_daily_rolls_over_the_year() -> None:
    schedule = build_schedule(repeat="daily", at="08:30")

    assert next_fire(schedule, after=moscow(2026, 12, 31, 20), tz=MOSCOW) == moscow(
        2027, 1, 1, 8, 30
    )


def test_weekly_picks_the_nearest_listed_weekday() -> None:
    # 2026-10-01 is a Thursday; Monday is 0, Friday is 4.
    schedule = build_schedule(repeat="weekly", at="10:00", weekdays=[4, 0])

    assert next_fire(schedule, after=moscow(2026, 10, 1, 12), tz=MOSCOW) == moscow(
        2026, 10, 2, 10
    )
    assert next_fire(schedule, after=moscow(2026, 10, 2, 11), tz=MOSCOW) == moscow(
        2026, 10, 5, 10
    )


def test_weekly_without_weekdays_uses_the_weekday_of_the_given_date() -> None:
    schedule = build_schedule(repeat="weekly", day="2026-10-05")

    assert schedule.weekdays == (0,)


def test_monthly_clamps_to_a_short_month() -> None:
    schedule = build_schedule(repeat="monthly", day_of_month=31, at="09:00")

    assert next_fire(schedule, after=moscow(2026, 2, 1), tz=MOSCOW) == moscow(
        2026, 2, 28, 9
    )
    assert next_fire(schedule, after=moscow(2026, 12, 31, 10), tz=MOSCOW) == moscow(
        2027, 1, 31, 9
    )


def test_leap_day_falls_on_the_28th_in_a_common_year() -> None:
    schedule = build_schedule(repeat="yearly", day="2024-02-29")

    assert next_fire(schedule, after=moscow(2026, 1, 1), tz=MOSCOW) == moscow(
        2026, 2, 28, 9
    )
    assert next_fire(schedule, after=moscow(2027, 3, 1), tz=MOSCOW) == moscow(
        2028, 2, 29, 9
    )


def test_once_has_one_firing_and_then_none() -> None:
    schedule = build_schedule(repeat="once", at="15:00", day="2026-10-03")

    assert next_fire(schedule, after=moscow(2026, 10, 1), tz=MOSCOW) == moscow(
        2026, 10, 3, 15
    )
    assert next_fire(schedule, after=moscow(2026, 10, 3, 15), tz=MOSCOW) is None


def test_monthly_reminder_can_fire_before_the_day() -> None:
    schedule = build_schedule(repeat="monthly", day_of_month=5, days_before=2)

    assert next_fire(schedule, after=moscow(2026, 10, 1), tz=MOSCOW) == moscow(
        2026, 10, 3, 9
    )


def test_after_is_compared_in_the_owners_timezone() -> None:
    schedule = build_schedule(repeat="daily", at="09:00")
    # 06:30 UTC is 09:30 in Moscow, so today's 09:00 has already passed.
    after = datetime(2026, 10, 1, 6, 30, tzinfo=UTC)

    assert next_fire(schedule, after=after, tz=MOSCOW) == moscow(2026, 10, 2, 9)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, time(9, 0)),
        ("", time(9, 0)),
        ("9:05", time(9, 5)),
        ("21", time(21, 0)),
        ("18.30", time(18, 30)),
        ("0930", time(9, 30)),
    ],
)
def test_time_parsing(value: str | None, expected: time) -> None:
    assert parse_time(value) == expected


@pytest.mark.parametrize("value", ["25:00", "9:77", "утром"])
def test_unreadable_time_is_refused(value: str) -> None:
    with pytest.raises(ScheduleError):
        parse_time(value)


@pytest.mark.parametrize(
    "arguments",
    [
        {"repeat": "sometimes"},
        {"repeat": "once"},
        {"repeat": "yearly"},
        {"repeat": "weekly"},
        {"repeat": "weekly", "weekdays": [7]},
        {"repeat": "monthly"},
        {"repeat": "monthly", "day_of_month": 32},
        {"repeat": "once", "day": "завтра"},
        {"repeat": "yearly", "day": "2000-05-14", "days_before": 31},
    ],
)
def test_incomplete_schedules_are_refused_with_a_message(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ScheduleError):
        build_schedule(**arguments)  # type: ignore[arg-type]


def test_a_timestamp_in_the_date_field_is_reduced_to_its_day() -> None:
    schedule = build_schedule(repeat="once", day="2026-10-03T15:00:00+03:00")

    assert schedule.day == date(2026, 10, 3)


def test_descriptions_read_naturally() -> None:
    assert (
        describe(build_schedule(repeat="yearly", day="2000-05-14", days_before=1))
        == "каждый год, за день до 14 мая, в 09:00"
    )
    assert describe(build_schedule(repeat="daily", at="8:15")) == "каждый день в 08:15"
    assert (
        describe(build_schedule(repeat="weekly", weekdays=[0, 2], at="10:00"))
        == "еженедельно (пн, ср) в 10:00"
    )
    assert (
        describe(build_schedule(repeat="once", day="2026-10-03", at="15:00"))
        == "один раз, 03.10.2026, в 15:00"
    )
