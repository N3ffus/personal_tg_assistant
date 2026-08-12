from datetime import datetime

import pytest

from src.application.services.action_executor import ActionExecutor
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    ChatAction,
    CreateEventAction,
    CreateTaskAction,
    SaveNoteAction,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (ChatAction(type=ActionType.CHAT, text="Привет"), "Привет"),
        (
            CreateTaskAction(type=ActionType.CREATE_TASK, title="Купить продукты"),
            "✅ Понял, нужно создать задачу:\nКупить продукты",
        ),
        (
            CreateEventAction(
                type=ActionType.CREATE_EVENT,
                title="Стоматолог",
                starts_at=datetime(2026, 8, 13, 15, 0),
            ),
            "📅 Понял, нужно создать событие:\nСтоматолог\nВремя: 13.08.2026 15:00",
        ),
        (
            SaveNoteAction(type=ActionType.SAVE_NOTE, text="Люблю Python"),
            "📝 Понял, нужно сохранить заметку:\nЛюблю Python",
        ),
    ],
)
async def test_execute_returns_action_response(action: object, expected: str) -> None:
    result = await ActionExecutor().execute(action)  # type: ignore[arg-type]

    assert result == expected
