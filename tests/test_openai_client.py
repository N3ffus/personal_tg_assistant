from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.domain.assistant.enums import ActionType
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
                            '{"action":{"type":"create_task",'
                            '"title":"Купить продукты"}}'
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

    assert decision.action.type is ActionType.CREATE_TASK
    assert decision.action.title == "Купить продукты"
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

    assert decision.action.type is ActionType.CHAT
    assert decision.action.text == "Привет!"
