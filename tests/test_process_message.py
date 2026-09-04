from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.process_message import ProcessMessageUseCase
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import AssistantDecision, ChatAction


@pytest.mark.asyncio
async def test_process_message_parses_then_executes_decision() -> None:
    now = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
    action = ChatAction(type=ActionType.CHAT, text="Ответ")
    llm = SimpleNamespace(
        parse_message=AsyncMock(return_value=AssistantDecision(action=action))
    )
    action_executor = SimpleNamespace(execute=AsyncMock(return_value="Ответ"))
    use_case = ProcessMessageUseCase(
        llm=llm,
        action_executor=action_executor,  # type: ignore[arg-type]
    )

    result = await use_case.execute(
        text="Вопрос",
        now=now,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert result == "Ответ"
    llm.parse_message.assert_awaited_once_with(
        text="Вопрос",
        now=now,
        timezone="Europe/Moscow",
    )
    action_executor.execute.assert_awaited_once_with(
        action,
        user_id=42,
        now=now,
    )


@pytest.mark.asyncio
async def test_process_message_does_not_execute_when_llm_fails() -> None:
    error = RuntimeError("LLM unavailable")
    llm = SimpleNamespace(parse_message=AsyncMock(side_effect=error))
    action_executor = SimpleNamespace(execute=AsyncMock())
    use_case = ProcessMessageUseCase(
        llm=llm,
        action_executor=action_executor,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError) as exc:
        await use_case.execute(
            text="Вопрос",
            now=datetime(2026, 9, 3, tzinfo=UTC),
            timezone="Europe/Moscow",
            user_id=42,
        )

    assert exc.value is error
    action_executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_message_propagates_executor_failure() -> None:
    action = ChatAction(type=ActionType.CHAT, text="Ответ")
    llm = SimpleNamespace(
        parse_message=AsyncMock(return_value=AssistantDecision(action=action))
    )
    error = RuntimeError("calendar unavailable")
    action_executor = SimpleNamespace(execute=AsyncMock(side_effect=error))
    use_case = ProcessMessageUseCase(
        llm=llm,
        action_executor=action_executor,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError) as exc:
        await use_case.execute(
            text="Создай событие",
            now=datetime(2026, 9, 3, tzinfo=UTC),
            timezone="Europe/Moscow",
            user_id=42,
        )

    assert exc.value is error
