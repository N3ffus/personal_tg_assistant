from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from src.application.services.reminders import MAX_REMINDERS, ReminderService
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    CreateReminderAction,
    DeleteReminderAction,
    WatchCalendarAction,
)
from src.infrastructure.reminders.storage import ReminderStorage

pytestmark = pytest.mark.asyncio

OWNER = 7
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> ReminderStorage:
    store = ReminderStorage(
        database_path=str(tmp_path / "assistant.db"),
        encryption_key=Fernet.generate_key().decode(),
    )
    await store.initialize()
    return store


@pytest.fixture
def service(storage: ReminderStorage) -> ReminderService:
    return ReminderService(store=storage, timezone="Europe/Moscow")


def create(**fields: object) -> CreateReminderAction:
    return CreateReminderAction(type=ActionType.CREATE_REMINDER, **fields)  # type: ignore[arg-type]


def delete(query: str) -> DeleteReminderAction:
    return DeleteReminderAction(type=ActionType.DELETE_REMINDER, query=query)


async def test_a_birthday_reminder_is_set_for_the_day_before_at_nine(
    service: ReminderService, storage: ReminderStorage
) -> None:
    reply = await service.create(
        create(
            text="Завтра день рождения у Маши",
            repeat="yearly",
            date="1999-05-14",
            time="09:00",
            days_before=1,
        ),
        user_id=OWNER,
        now=NOW,
    )

    (stored,) = await storage.list_reminders(owner_id=OWNER)
    assert stored.next_run_at == datetime(2027, 5, 13, 6, 0, tzinfo=UTC)
    assert "каждый год, за день до 14 мая, в 09:00" in reply
    assert "13.05.2027 09:00" in reply


async def test_time_defaults_to_nine_in_the_morning(
    service: ReminderService, storage: ReminderStorage
) -> None:
    await service.create(
        create(text="Позвонить маме", repeat="daily"), user_id=OWNER, now=NOW
    )

    (stored,) = await storage.list_reminders(owner_id=OWNER)
    assert stored.next_run_at == datetime(2026, 10, 2, 6, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"repeat": "once"}, "Укажи дату"),
        ({"repeat": "weekly"}, "Укажи дни недели"),
        ({"repeat": "once", "date": "2026-10-01", "time": "11:00"}, "уже прошло"),
        ({"repeat": "once", "date": "2026-10-02", "time": "26:00"}, "Не понял время"),
    ],
)
async def test_an_unusable_schedule_gets_a_question_and_stores_nothing(
    service: ReminderService,
    storage: ReminderStorage,
    fields: dict[str, object],
    expected: str,
) -> None:
    reply = await service.create(create(text="Врач", **fields), user_id=OWNER, now=NOW)

    assert expected in reply
    assert await storage.list_reminders(owner_id=OWNER) == []


async def test_the_number_of_reminders_is_capped(
    service: ReminderService, storage: ReminderStorage
) -> None:
    for _ in range(MAX_REMINDERS):
        await service.create(create(text="x", repeat="daily"), user_id=OWNER, now=NOW)

    reply = await service.create(
        create(text="лишнее", repeat="daily"), user_id=OWNER, now=NOW
    )

    assert "предел" in reply
    assert len(await storage.list_reminders(owner_id=OWNER)) == MAX_REMINDERS


async def test_overview_lists_reminders_and_the_calendar_watch(
    service: ReminderService,
) -> None:
    assert "Напоминаний пока нет" in await service.overview(user_id=OWNER)

    await service.create(
        create(text="Таблетки", repeat="daily", time="8:30"), user_id=OWNER, now=NOW
    )
    await service.watch(
        WatchCalendarAction(type=ActionType.WATCH_CALENDAR, lead_minutes=90),
        user_id=OWNER,
        now=NOW,
    )

    overview = await service.overview(user_id=OWNER)

    assert "#1 Таблетки — каждый день в 08:30" in overview
    assert "предупреждаю о событиях за 90 мин" in overview


async def test_delete_by_words_from_the_text(
    service: ReminderService, storage: ReminderStorage
) -> None:
    await service.create(
        create(text="Купить таблетки", repeat="daily"), user_id=OWNER, now=NOW
    )

    reply = await service.delete(delete("ТАБЛЕТКИ"), user_id=OWNER)

    assert "Удалил напоминание: Купить таблетки" in reply
    assert await storage.list_reminders(owner_id=OWNER) == []


async def test_delete_by_number(
    service: ReminderService, storage: ReminderStorage
) -> None:
    await service.create(create(text="a", repeat="daily"), user_id=OWNER, now=NOW)
    await service.create(create(text="b", repeat="daily"), user_id=OWNER, now=NOW)

    await service.delete(delete("#2"), user_id=OWNER)

    assert [r.text for r in await storage.list_reminders(owner_id=OWNER)] == ["a"]


async def test_an_ambiguous_delete_asks_instead_of_guessing(
    service: ReminderService, storage: ReminderStorage
) -> None:
    await service.create(
        create(text="Врач в среду", repeat="daily"), user_id=OWNER, now=NOW
    )
    await service.create(
        create(text="Врач в пятницу", repeat="daily"), user_id=OWNER, now=NOW
    )

    reply = await service.delete(delete("врач"), user_id=OWNER)

    assert "Уточни" in reply
    assert "#1" in reply and "#2" in reply
    assert len(await storage.list_reminders(owner_id=OWNER)) == 2


async def test_delete_of_an_unknown_reminder_says_so(
    service: ReminderService,
) -> None:
    assert "не найдено" in await service.delete(delete("зубы"), user_id=OWNER)


async def test_watch_can_be_retuned_without_losing_the_other_setting(
    service: ReminderService, storage: ReminderStorage
) -> None:
    await service.watch(
        WatchCalendarAction(
            type=ActionType.WATCH_CALENDAR, lead_minutes=120, interval_minutes=30
        ),
        user_id=OWNER,
        now=NOW,
    )

    await service.watch(
        WatchCalendarAction(type=ActionType.WATCH_CALENDAR, lead_minutes=45),
        user_id=OWNER,
        now=NOW,
    )

    watch = await storage.get_watch(owner_id=OWNER)
    assert watch is not None
    assert (watch.lead_minutes, watch.interval_minutes) == (45, 30)


async def test_watch_can_be_turned_off(
    service: ReminderService, storage: ReminderStorage
) -> None:
    await service.watch(
        WatchCalendarAction(type=ActionType.WATCH_CALENDAR), user_id=OWNER, now=NOW
    )

    reply = await service.watch(
        WatchCalendarAction(type=ActionType.WATCH_CALENDAR, enabled=False),
        user_id=OWNER,
        now=NOW,
    )

    assert "выключено" in reply
    assert await storage.due_watches(now=NOW) == []


async def test_watch_refuses_absurd_values(service: ReminderService) -> None:
    reply = await service.watch(
        WatchCalendarAction(type=ActionType.WATCH_CALENDAR, lead_minutes=1),
        user_id=OWNER,
        now=NOW,
    )

    assert "5–1440" in reply
