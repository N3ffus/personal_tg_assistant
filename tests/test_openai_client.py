from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import CreateTaskAction
from src.infrastructure.llm.gonkagate import GonkaGateLLMClient


@pytest.mark.asyncio
async def test_parse_message_validates_json_chat_completion() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"actions":[{"type":"create_task",'
                            '"title":"Купить продукты"}]}'
                        )
                    )
                )
            ]
        )
    )
    client._client.chat.completions.create = create  # type: ignore[method-assign]

    decision = await client.parse_message(
        text="Добавь задачу купить продукты",
        now=datetime(2026, 8, 12, 12, 0),
        timezone="Europe/Moscow",
    )

    assert decision.actions[0].type is ActionType.CREATE_TASK
    assert decision.actions[0].title == "Купить продукты"
    assert create.await_args is not None
    assert create.await_args.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_parse_message_rejects_empty_completion() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )
    client._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(choices=[])
    )

    with pytest.raises(RuntimeError, match="no choices"):
        await client.parse_message(
            text="Привет",
            now=datetime(2026, 8, 12, 12, 0),
            timezone="Europe/Moscow",
        )


@pytest.mark.asyncio
async def test_parse_message_supports_flat_action_response() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )
    client._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"action":"chat","text":"Привет!"}'
                    )
                )
            ]
        )
    )

    decision = await client.parse_message(
        text="Привет",
        now=datetime(2026, 8, 12, 12, 0),
        timezone="Europe/Moscow",
    )

    assert decision.actions[0].type is ActionType.CHAT
    assert decision.actions[0].text == "Привет!"


@pytest.mark.asyncio
async def test_parse_message_supports_legacy_nested_action_response() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )
    client._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"action":{"type":"chat","text":"Привет!"}}'
                    )
                )
            ]
        )
    )

    decision = await client.parse_message(
        text="Привет",
        now=datetime(2026, 8, 12, 12, 0),
        timezone="Europe/Moscow",
    )

    assert decision.actions[0].type is ActionType.CHAT
    assert decision.actions[0].text == "Привет!"


@pytest.mark.asyncio
async def test_parse_message_supports_multiple_actions() -> None:
    client = GonkaGateLLMClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )
    client._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"actions":['
                            '{"type":"create_task","title":"Посмотреть фильм"},'
                            '{"type":"create_task","title":"Поботать LLM-ки"},'
                            '{"type":"create_task","title":"Отдохнуть"}'
                            "]}"
                        )
                    )
                )
            ]
        )
    )

    decision = await client.parse_message(
        text="Завтра посмотреть фильм, поботать LLM-ки и отдохнуть",
        now=datetime(2026, 9, 4, 15, 39),
        timezone="Europe/Moscow",
    )

    assert all(isinstance(action, CreateTaskAction) for action in decision.actions)
    assert [
        action.title
        for action in decision.actions
        if isinstance(action, CreateTaskAction)
    ] == [
        "Посмотреть фильм",
        "Поботать LLM-ки",
        "Отдохнуть",
    ]
