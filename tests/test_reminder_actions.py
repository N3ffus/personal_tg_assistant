import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from src.application.services.action_executor import ActionExecutor
from src.application.services.reminders import ReminderService
from src.domain.assistant.models import AssistantDecision
from src.infrastructure.reminders.storage import ReminderStorage
from src.infrastructure.telegram import handlers as common_handlers
from src.main import _run_services

pytestmark = pytest.mark.asyncio

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))


@pytest_asyncio.fixture
async def service(tmp_path: Path) -> ReminderService:
    storage = ReminderStorage(
        database_path=str(tmp_path / "assistant.db"),
        encryption_key=Fernet.generate_key().decode(),
    )
    await storage.initialize()
    return ReminderService(store=storage, timezone="Europe/Moscow")


def executor_with(reminders: ReminderService | None) -> ActionExecutor:
    return ActionExecutor(
        calendar=SimpleNamespace(),
        pending_operations=SimpleNamespace(),
        task_tracker=SimpleNamespace(),
        reminders=reminders,
    )


def decision(*actions: dict[str, Any]) -> AssistantDecision:
    return AssistantDecision.model_validate({"actions": list(actions)})


async def test_the_model_can_set_list_and_cancel_reminders_in_one_conversation(
    service: ReminderService,
) -> None:
    executor = executor_with(service)

    created = await executor.execute_many(
        decision(
            {
                "type": "create_reminder",
                "text": "Завтра день рождения у Маши",
                "repeat": "yearly",
                "date": "2026-05-14",
                "days_before": 1,
            },
            {
                "type": "create_reminder",
                "text": "Выпить таблетки",
                "repeat": "weekly",
                "weekdays": [0, 1, 2, 3, 4],
                "time": "08:30",
            },
        ).actions,
        user_id=42,
        now=NOW,
    )
    listed = await executor.execute_many(
        decision({"type": "list_reminders"}).actions, user_id=42, now=NOW
    )
    cancelled = await executor.execute_many(
        decision({"type": "delete_reminder", "query": "таблетки"}).actions,
        user_id=42,
        now=NOW,
    )
    after = await executor.execute_many(
        decision({"type": "list_reminders"}).actions, user_id=42, now=NOW
    )

    assert isinstance(created, str) and created.count("⏰ Напомню") == 2
    assert isinstance(listed, str)
    assert "#1 Завтра день рождения у Маши — каждый год, за день до 14 мая" in listed
    assert "#2 Выпить таблетки — еженедельно (пн, вт, ср, чт, пт) в 08:30" in listed
    assert cancelled == "🗑 Удалил напоминание: Выпить таблетки"
    assert isinstance(after, str) and "Выпить таблетки" not in after


async def test_calendar_watch_is_configured_through_an_action(
    service: ReminderService,
) -> None:
    executor = executor_with(service)

    reply = await executor.execute_many(
        decision(
            {"type": "watch_calendar", "lead_minutes": 120, "interval_minutes": 30}
        ).actions,
        user_id=42,
        now=NOW,
    )

    assert reply == (
        "👀 Слежу за календарём: проверяю каждые 30 мин и предупреждаю о "
        "событиях за 120 мин до начала."
    )


async def test_without_a_reminder_service_the_user_is_told() -> None:
    reply = await executor_with(None).execute_many(
        decision({"type": "list_reminders"}).actions, user_id=42, now=NOW
    )

    assert reply == "⚠️ Напоминания сейчас недоступны."


async def test_reminders_command_shows_the_overview_to_the_owner_only(
    service: ReminderService,
) -> None:
    router = common_handlers.create_router(
        process_message=SimpleNamespace(),  # type: ignore[arg-type]
        timezone="Europe/Moscow",
        allowed_user_id=42,
        reminders=service,
    )
    handler = next(
        h.callback
        for h in router.message.handlers
        if h.callback.__name__ == "reminders_handler"
    )

    class Message:
        def __init__(self, user_id: int) -> None:
            self.from_user = SimpleNamespace(id=user_id)
            self.answer = AsyncMock()

    owner, stranger = Message(42), Message(99)
    await handler(owner)
    await handler(stranger)

    assert owner.answer.await_args is not None
    assert "Напоминаний пока нет" in owner.answer.await_args.args[0]
    stranger.answer.assert_not_awaited()


async def test_the_scheduler_runs_beside_polling_and_stops_with_it() -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Scheduler:
        async def run(self) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    class Dispatcher:
        async def start_polling(self, bot: object, **kwargs: object) -> None:
            await started.wait()

    class Server:
        should_exit = False

        async def serve(self) -> None:
            await asyncio.Event().wait()

    await _run_services(
        dispatcher=Dispatcher(),  # type: ignore[arg-type]
        bot=object(),  # type: ignore[arg-type]
        server=Server(),  # type: ignore[arg-type]
        scheduler=Scheduler(),
    )

    assert stopped.is_set()
