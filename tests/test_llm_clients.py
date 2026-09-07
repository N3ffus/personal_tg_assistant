import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    AssistantDecision,
    ChatAction,
)
from src.infrastructure.llm.gonkagate import GonkaGateLLMClient
from src.infrastructure.llm.openai import OpenAILLMClient

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone(timedelta(hours=3)))


@pytest.mark.parametrize(
    "actions",
    [
        [],
        [{"type": "create_task", "title": f"Task {index}"} for index in range(11)],
    ],
)
def test_assistant_decision_limits_batch_size(actions: list[object]) -> None:
    with pytest.raises(ValidationError):
        AssistantDecision.model_validate({"actions": actions})


@pytest.mark.asyncio
async def test_openai_client_requests_structured_decision_with_user_context() -> None:
    client = OpenAILLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    expected = AssistantDecision(
        actions=[ChatAction(type=ActionType.CHAT, text="Привет!")]
    )
    parse = AsyncMock(return_value=SimpleNamespace(output_parsed=expected))
    client._client.responses.parse = parse  # type: ignore[method-assign]

    result = await client.parse_message(
        text="Привет",
        now=NOW,
        timezone="Europe/Moscow",
    )

    assert result is expected
    parse.assert_awaited_once()
    assert parse.await_args is not None
    kwargs = parse.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["input"] == "Привет"
    assert kwargs["text_format"] is AssistantDecision
    assert f"Текущее время: {NOW.isoformat()}" in kwargs["instructions"]
    assert "Timezone пользователя: Europe/Moscow" in kwargs["instructions"]
    assert "list_events" in kwargs["instructions"]
    assert "list_tasks" in kwargs["instructions"]
    assert "update_event" in kwargs["instructions"]
    assert "delete_event" in kwargs["instructions"]
    assert "в массиве actions" in kwargs["instructions"]
    assert "Поботать LLM-ки" in kwargs["instructions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_name", ["openai", "gonkagate"])
async def test_recalled_knowledge_turns_the_call_into_an_answer_turn(
    client_name: str,
) -> None:
    """A repeated search would loop and bury the answer under a raw fact dump."""
    expected = AssistantDecision(actions=[ChatAction(type=ActionType.CHAT, text="Ок")])
    if client_name == "openai":
        client: OpenAILLMClient | GonkaGateLLMClient = OpenAILLMClient(
            api_key="test-key", base_url="https://example.test/v1", model="test-model"
        )
        call = AsyncMock(return_value=SimpleNamespace(output_parsed=expected))
        client._client.responses.parse = call  # type: ignore[method-assign]
    else:
        client = GonkaGateLLMClient(
            api_key="test-key", base_url="https://example.test/v1", model="test-model"
        )
        call = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=expected.model_dump_json())
                    )
                ]
            )
        )
        client._client.chat.completions.create = call  # type: ignore[method-assign]

    await client.parse_message(
        text="Сколько мне лет",
        now=NOW,
        timezone="Europe/Moscow",
        knowledge='[{"fact": "Работает с Python"}]',
    )

    assert call.await_args is not None
    kwargs = call.await_args.kwargs
    instructions = (
        kwargs["instructions"]
        if client_name == "openai"
        else kwargs["messages"][0]["content"]
    )
    assert "Повторный search_knowledge невозможен" in instructions


@pytest.mark.asyncio
@pytest.mark.parametrize("client_name", ["openai", "gonkagate"])
async def test_the_answer_turn_instruction_is_absent_without_recall(
    client_name: str,
) -> None:
    expected = AssistantDecision(actions=[ChatAction(type=ActionType.CHAT, text="Ок")])
    if client_name == "openai":
        client: OpenAILLMClient | GonkaGateLLMClient = OpenAILLMClient(
            api_key="test-key", base_url="https://example.test/v1", model="test-model"
        )
        call = AsyncMock(return_value=SimpleNamespace(output_parsed=expected))
        client._client.responses.parse = call  # type: ignore[method-assign]
    else:
        client = GonkaGateLLMClient(
            api_key="test-key", base_url="https://example.test/v1", model="test-model"
        )
        call = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=expected.model_dump_json())
                    )
                ]
            )
        )
        client._client.chat.completions.create = call  # type: ignore[method-assign]

    await client.parse_message(text="Привет", now=NOW, timezone="Europe/Moscow")

    assert call.await_args is not None
    kwargs = call.await_args.kwargs
    instructions = (
        kwargs["instructions"]
        if client_name == "openai"
        else kwargs["messages"][0]["content"]
    )
    assert "Повторный search_knowledge невозможен" not in instructions


@pytest.mark.asyncio
async def test_openai_client_rejects_missing_structured_output() -> None:
    client = OpenAILLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    client._client.responses.parse = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(output_parsed=None)
    )

    with pytest.raises(RuntimeError, match="no parsed output"):
        await client.parse_message(
            text="Привет",
            now=NOW,
            timezone="Europe/Moscow",
        )


@pytest.mark.asyncio
async def test_openai_client_close_closes_sdk_client() -> None:
    client = OpenAILLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    close = AsyncMock()
    client._client.close = close  # type: ignore[method-assign]

    await client.close()

    close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_gonkagate_client_includes_json_contract_and_user_context() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"actions":[{"type":"list_events"}]}'
                    )
                )
            ]
        )
    )
    client._client.chat.completions.create = create  # type: ignore[method-assign]

    decision = await client.parse_message(
        text="Что у меня в календаре?",
        now=NOW,
        timezone="Europe/Moscow",
    )

    assert decision.actions[0].type is ActionType.LIST_EVENTS
    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][1] == {
        "role": "user",
        "content": "Что у меня в календаре?",
    }
    instructions = kwargs["messages"][0]["content"]
    assert f"Текущее время: {NOW.isoformat()}" in instructions
    assert "Timezone пользователя: Europe/Moscow" in instructions
    assert '"type":"update_event"' in instructions
    assert '"type":"list_tasks"' in instructions
    assert '"type":"delete_event"' in instructions
    assert "Корневой объект всегда содержит массив actions" in instructions
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_gonkagate_client_rejects_empty_message_content() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    client._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
    )

    with pytest.raises(RuntimeError, match="empty response"):
        await client.parse_message(
            text="Привет",
            now=NOW,
            timezone="Europe/Moscow",
        )


@pytest.mark.asyncio
async def test_gonkagate_client_rejects_malformed_json() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    client._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
        )
    )

    with pytest.raises(json.JSONDecodeError):
        await client.parse_message(
            text="Привет",
            now=NOW,
            timezone="Europe/Moscow",
        )


@pytest.mark.asyncio
async def test_gonkagate_client_validates_action_schema() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    client._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"actions":[{"type":"create_event","title":"Meet"}]}'
                    )
                )
            ]
        )
    )

    with pytest.raises(ValidationError):
        await client.parse_message(
            text="Создай встречу",
            now=NOW,
            timezone="Europe/Moscow",
        )


@pytest.mark.asyncio
async def test_gonkagate_client_close_closes_sdk_client() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    close = AsyncMock()
    client._client.close = close  # type: ignore[method-assign]

    await client.close()

    close.assert_awaited_once_with()
