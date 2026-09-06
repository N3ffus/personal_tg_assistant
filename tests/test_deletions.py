import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import aiosqlite
import pytest
from cryptography.fernet import Fernet

from src.application.ports.calendar import CalendarError, CalendarEventNotFoundError
from src.application.ports.tasks import (
    TaskDeletionUncertainError,
    TaskNotFoundError,
    TaskTrackerError,
)
from src.application.services.action_executor import ActionExecutor
from src.application.services.deletions import DeletionService
from src.application.services.explicit_commands import parse_bulk_deletion
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.domain.assistant.deletions import DeletionResource, DeletionTarget
from src.domain.assistant.models import AssistantDecision
from src.domain.assistant.replies import AssistantReply
from src.infrastructure.calendar.storage import CalendarStorage


async def dependencies(
    tmp_path: Path,
) -> tuple[DeletionService, SimpleNamespace, SimpleNamespace, CalendarStorage]:
    storage = CalendarStorage(
        database_path=str(tmp_path / "calendar.db"),
        encryption_key=Fernet.generate_key().decode(),
    )
    await storage.initialize()
    tasks = SimpleNamespace(
        find_tasks=AsyncMock(
            return_value=[
                DeletionTarget(
                    id="issue-1", title="Купить молоко", label="ENG-1: Купить молоко"
                )
            ]
        ),
        delete_task=AsyncMock(),
    )
    calendar = SimpleNamespace(
        find_events=AsyncMock(return_value=[]), delete_event=AsyncMock()
    )
    service = DeletionService(calendar=calendar, task_tracker=tasks, storage=storage)
    return service, tasks, calendar, storage


@pytest.mark.asyncio
async def test_bulk_deletion_requires_button_and_uses_snapshot_once(
    tmp_path: Path,
) -> None:
    service, tasks, _, _ = await dependencies(tmp_path)
    reply = await service.prepare(user_id=42, resource="linear", title=None)
    assert isinstance(reply, AssistantReply)
    assert "1" in reply.confirmations[0].text
    tasks.delete_task.assert_not_awaited()
    operation_id = reply.confirmations[0].operation_id
    tasks.find_tasks.return_value.append(
        DeletionTarget(id="new-issue", title="Новая", label="ENG-2: Новая")
    )
    result = await service.resolve(user_id=42, operation_id=operation_id, confirm=True)
    assert "Удалено: 1" in result
    tasks.delete_task.assert_awaited_once_with(task_id="issue-1")
    assert "устарело" in await service.resolve(
        user_id=42, operation_id=operation_id, confirm=True
    )
    assert tasks.delete_task.await_count == 1


@pytest.mark.asyncio
async def test_foreign_user_cannot_consume_confirmation_and_cancel_prevents_delete(
    tmp_path: Path,
) -> None:
    service, tasks, _, _ = await dependencies(tmp_path)
    reply = await service.prepare(user_id=42, resource="linear", title="ENG-1")
    assert isinstance(reply, AssistantReply)
    operation_id = reply.confirmations[0].operation_id
    assert "устарело" in await service.resolve(
        user_id=99, operation_id=operation_id, confirm=True
    )
    assert "отменено" in await service.resolve(
        user_id=42, operation_id=operation_id, confirm=False
    )
    assert "устарело" in await service.resolve(
        user_id=42, operation_id=operation_id, confirm=True
    )
    tasks.delete_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_event_titles_offer_individual_confirmations(
    tmp_path: Path,
) -> None:
    service, _, calendar, _ = await dependencies(tmp_path)
    calendar.find_events.return_value = [
        DeletionTarget(
            id="event-1", title="Встреча", label="06.09.2026 10:00 — Встреча"
        ),
        DeletionTarget(
            id="event-2", title="Встреча", label="07.09.2026 11:00 — Встреча"
        ),
    ]
    reply = await service.prepare(user_id=42, resource="calendar", title="Встреча")
    assert isinstance(reply, AssistantReply)
    assert len(reply.confirmations) == 2
    await service.resolve(
        user_id=42, operation_id=reply.confirmations[1].operation_id, confirm=True
    )
    calendar.delete_event.assert_awaited_once_with(user_id=42, event_id="event-2")


@pytest.mark.asyncio
async def test_empty_search_does_not_create_confirmation(tmp_path: Path) -> None:
    service, _, _, _ = await dependencies(tmp_path)
    reply = await service.prepare(user_id=42, resource="calendar", title=None)
    assert isinstance(reply, str)
    assert "не найдено" in reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Удали все задачи с Linear",
        "Удали все задачи в Linear",
        "Удали все задачи с Linear все",
        "Удали все задачи с Linear подтверждаю",
        "Пожалуйста, удали все задачи из Linear!",
        "Удали все события в календаре",
        "Удали все события из Google календаря",
    ],
)
async def test_reported_bulk_commands_return_buttons_without_llm_or_deletion(
    tmp_path: Path, text: str
) -> None:
    _, tasks, calendar, storage = await dependencies(tmp_path)
    calendar.find_events.return_value = [
        DeletionTarget(id="event-1", title="Встреча", label="Встреча")
    ]
    llm = SimpleNamespace(
        parse_message=AsyncMock(
            side_effect=AssertionError("LLM must not route an explicit bulk command")
        )
    )
    use_case = ProcessMessageUseCase(
        llm=llm,
        action_executor=ActionExecutor(
            calendar=calendar, task_tracker=tasks, pending_operations=storage
        ),
    )
    reply = await use_case.execute(
        text=text, user_id=42, now=datetime.now(UTC), timezone="Europe/Moscow"
    )
    assert isinstance(reply, AssistantReply)
    assert len(reply.confirmations) == 1
    tasks.delete_task.assert_not_awaited()
    calendar.delete_event.assert_not_awaited()
    llm.parse_message.assert_not_awaited()
    await use_case.resolve_deletion(
        user_id=42, operation_id=reply.confirmations[0].operation_id, confirm=True
    )
    if "задачи" in text:
        tasks.find_tasks.assert_awaited_once_with(title=None)
        calendar.find_events.assert_not_awaited()
        tasks.delete_task.assert_awaited_once_with(task_id="issue-1")
    else:
        calendar.find_events.assert_awaited_once_with(user_id=42, title=None)
        tasks.find_tasks.assert_not_awaited()
        calendar.delete_event.assert_awaited_once_with(user_id=42, event_id="event-1")


@pytest.mark.parametrize(
    "text",
    [
        "Не удали все задачи с Linear",
        "Удали все задачи с Linear кроме ENG-1",
        "Удали все задачи с Linear и создай событие",
        "Удали все события в календаре завтра",
        "Удали все выполненные задачи с Linear",
        "Как удалить все задачи в Linear?",
        "подтверждаю",
    ],
)
def test_bulk_shortcut_does_not_expand_qualified_or_negative_requests(
    text: str,
) -> None:
    assert parse_bulk_deletion(text) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["linear", "calendar"])
async def test_empty_and_blank_queries_are_safe(
    tmp_path: Path, resource: DeletionResource
) -> None:
    service, tasks, calendar, _ = await dependencies(tmp_path)
    tasks.find_tasks.return_value = []
    empty_reply = await service.prepare(user_id=42, resource=resource, title=None)
    assert isinstance(empty_reply, str)
    assert "не найдено" in empty_reply
    tasks.find_tasks.reset_mock()
    calendar.find_events.reset_mock()
    blank_reply = await service.prepare(user_id=42, resource=resource, title=" ")
    assert isinstance(blank_reply, str)
    assert "Укажите" in blank_reply
    tasks.find_tasks.assert_not_awaited()
    calendar.find_events.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_confirmation_consumed_once(tmp_path: Path) -> None:
    service, tasks, _, _ = await dependencies(tmp_path)
    reply = await service.prepare(user_id=42, resource="linear", title=None)
    assert isinstance(reply, AssistantReply)
    results = await asyncio.gather(
        *(
            service.resolve(
                user_id=42,
                operation_id=reply.confirmations[0].operation_id,
                confirm=True,
            )
            for _ in range(2)
        )
    )
    assert sum("Удалено: 1" in result for result in results) == 1
    tasks.delete_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_expired_confirmation_and_invalid_payload_do_not_delete(
    tmp_path: Path,
) -> None:
    service, tasks, _, storage = await dependencies(tmp_path)
    reply = await service.prepare(user_id=42, resource="linear", title=None)
    assert isinstance(reply, AssistantReply)
    async with aiosqlite.connect(str(tmp_path / "calendar.db")) as db:
        await db.execute(
            "UPDATE pending_operations SET expires_at = ?",
            ((datetime.now(UTC) - timedelta(minutes=1)).isoformat(),),
        )
        await db.commit()
    assert "устарело" in await service.resolve(
        user_id=42, operation_id=reply.confirmations[0].operation_id, confirm=True
    )
    for kind in ("delete_confirm", "select"):
        operation_id = await storage.create_operation(
            user_id=42, kind=kind, payload={"resource": "linear", "targets": []}
        )
        assert "недействительно" in await service.resolve(
            user_id=42, operation_id=operation_id, confirm=True
        )
    tasks.delete_task.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [TaskTrackerError(), TaskDeletionUncertainError(), RuntimeError("unexpected")],
)
async def test_partial_failure_is_reported_and_does_not_repeat_success(
    tmp_path: Path, error: Exception
) -> None:
    service, tasks, _, _ = await dependencies(tmp_path)
    tasks.find_tasks.return_value = [
        DeletionTarget(id=f"issue-{i}", title=f"Задача {i}", label=f"ENG-{i}")
        for i in range(3)
    ]
    tasks.delete_task.side_effect = [None, error, None]
    reply = await service.prepare(user_id=42, resource="linear", title=None)
    assert isinstance(reply, AssistantReply)
    result = await service.resolve(
        user_id=42, operation_id=reply.confirmations[0].operation_id, confirm=True
    )
    assert "Удалено: 2 из 3" in result
    assert "Не подтверждено: 1" in result
    assert "ENG-1" in result
    assert tasks.delete_task.await_args_list == [
        call(task_id=f"issue-{i}") for i in range(3)
    ]


@pytest.mark.asyncio
async def test_bulk_calendar_missing_events_are_reported(tmp_path: Path) -> None:
    service, _, calendar, _ = await dependencies(tmp_path)
    calendar.find_events.return_value = [
        DeletionTarget(id=f"event-{i}", title="Встреча", label=f"Встреча {i}")
        for i in range(3)
    ]
    calendar.delete_event.side_effect = [
        None,
        CalendarEventNotFoundError(),
        CalendarError(),
    ]
    reply = await service.prepare(user_id=42, resource="calendar", title=None)
    assert isinstance(reply, AssistantReply)
    result = await service.resolve(
        user_id=42, operation_id=reply.confirmations[0].operation_id, confirm=True
    )
    assert "Удалено: 1 из 3" in result
    assert "не найдено: 1" in result
    assert "Не подтверждено: 1" in result


@pytest.mark.asyncio
async def test_bulk_linear_missing_tasks_are_reported(tmp_path: Path) -> None:
    service, tasks, _, _ = await dependencies(tmp_path)
    tasks.find_tasks.return_value = [
        DeletionTarget(id=f"issue-{i}", title="Задача", label=f"ENG-{i}")
        for i in range(3)
    ]
    tasks.delete_task.side_effect = [
        None,
        TaskNotFoundError(),
        TaskTrackerError(),
    ]
    reply = await service.prepare(user_id=42, resource="linear", title=None)
    assert isinstance(reply, AssistantReply)
    result = await service.resolve(
        user_id=42, operation_id=reply.confirmations[0].operation_id, confirm=True
    )
    assert "Удалено: 1 из 3" in result
    assert "не найдено: 1" in result
    assert "Не подтверждено: 1" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["delete_task", "delete_event", "delete_all_tasks", "delete_all_events"]
)
async def test_executor_returns_confirmations_for_llm_deletion_actions(
    tmp_path: Path, kind: str
) -> None:
    _, tasks, calendar, storage = await dependencies(tmp_path)
    calendar.find_events.return_value = [
        DeletionTarget(id="event", title="Встреча", label="Встреча")
    ]
    action: dict[str, object] = {"type": kind}
    if "all" not in kind:
        action["title"] = "Встреча"
    executor = ActionExecutor(
        calendar=calendar, task_tracker=tasks, pending_operations=storage
    )
    reply = await executor.execute_many(
        AssistantDecision.model_validate({"actions": [action]}).actions,
        user_id=42,
        now=datetime.now(UTC),
    )
    assert isinstance(reply, AssistantReply)
    assert len(reply.confirmations) == 1
    calendar.delete_event.assert_not_awaited()
    tasks.delete_task.assert_not_awaited()
