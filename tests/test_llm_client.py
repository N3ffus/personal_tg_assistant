import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import AssistantDecision, CreateTaskAction
from src.infrastructure.llm.client import REQUEST_TIMEOUT, ChatLLMClient

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


def completion(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recalled", [True, False])
async def test_recalled_knowledge_turns_the_call_into_an_answer_turn(
    recalled: bool,
) -> None:
    """A repeated search would loop and bury the answer under a raw fact dump."""
    client = ChatLLMClient(
        api_key="test-key", base_url="https://example.test/v1", model="test-model"
    )
    call = AsyncMock(
        return_value=completion('{"actions":[{"type":"chat","text":"Ок"}]}')
    )
    client._client.chat.completions.create = call  # type: ignore[method-assign]

    await client.parse_message(
        text="Сколько мне лет",
        now=NOW,
        timezone="Europe/Moscow",
        knowledge='[{"fact": "Работает с Python"}]' if recalled else "",
    )

    assert call.await_args is not None
    instructions = call.await_args.kwargs["messages"][0]["content"]
    assert ("Повторный search_knowledge невозможен" in instructions) is recalled
    assert "Поботать LLM-ки" in instructions


@pytest.mark.asyncio
async def test_a_field_hoisted_out_of_an_action_does_not_lose_the_reply() -> None:
    """The model put a recommendation's `limit` beside actions, not inside."""
    client = ChatLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    client._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "actions": [
                                    {
                                        "type": "recommend_films",
                                        "candidates": [
                                            {"title": "Дюна", "reason": "Эпос"}
                                        ],
                                    }
                                ],
                                "limit": 3,
                            }
                        )
                    )
                )
            ]
        )
    )

    decision = await client.parse_message(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
    )

    assert decision.actions[0].type is ActionType.RECOMMEND_FILMS


@pytest.mark.asyncio
async def test_client_includes_json_contract_and_user_context() -> None:
    client = ChatLLMClient(
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
    # A changing timestamp must not invalidate the large static schema prefix.
    assert instructions.index('"$defs"') < instructions.index("Текущее время:")


@pytest.mark.asyncio
async def test_client_rejects_empty_message_content() -> None:
    client = ChatLLMClient(
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
async def test_client_rejects_malformed_json() -> None:
    client = ChatLLMClient(
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
    assert client._client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_client_validates_action_schema() -> None:
    client = ChatLLMClient(
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
async def test_client_close_closes_sdk_client() -> None:
    client = ChatLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    close = AsyncMock()
    client._client.close = close  # type: ignore[method-assign]

    await client.close()

    close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_client_asks_again_when_actions_come_back_empty() -> None:
    """An empty actions array used to cost the user the whole reply."""
    client = ChatLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    replies = ['{"actions":[]}', '{"actions":[{"type":"chat","text":"Джек"}]}']
    create = AsyncMock(
        side_effect=[
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
            )
            for reply in replies
        ]
    )
    client._client.chat.completions.create = create  # type: ignore[method-assign]

    decision = await client.parse_message(
        text="Как зовут мою собаку?", now=NOW, timezone="Europe/Moscow"
    )

    assert [action.type for action in decision.actions] == [ActionType.CHAT]
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_parse_message_validates_json_chat_completion() -> None:
    client = ChatLLMClient(
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
    client = ChatLLMClient(
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
    client = ChatLLMClient(
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
    client = ChatLLMClient(
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
    client = ChatLLMClient(
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broken",
    [
        "not-json",
        '{"actions":[{"type":"chat","text":"Палермо-клаудиное (нельзя орех — исключ)]}',
        '{"actions":[{"type":"create_event","title":"Meet"}]}',
        None,
    ],
)
async def test_a_reply_that_cannot_become_a_decision_is_asked_again(
    broken: str | None,
) -> None:
    """Truncated JSON or a schema slip used to cost the user the whole reply."""
    client = ChatLLMClient(
        api_key="test-key", base_url="https://example.test/v1", model="test-model"
    )
    create = AsyncMock(
        side_effect=[
            completion(broken),
            completion('{"actions":[{"type":"chat","text":"Готово"}]}'),
        ]
    )
    client._client.chat.completions.create = create  # type: ignore[method-assign]

    decision = await client.parse_message(
        text="Привет", now=NOW, timezone="Europe/Moscow"
    )

    assert [action.type for action in decision.actions] == [ActionType.CHAT]
    assert create.await_count == 2


def test_client_bounds_how_long_a_provider_may_hang() -> None:
    """The SDK default of 600 s x 3 tries left a message unanswered for minutes."""
    client = ChatLLMClient(
        api_key="test-key", base_url="https://example.test/v1", model="test-model"
    )

    assert client._client.timeout == REQUEST_TIMEOUT
    assert REQUEST_TIMEOUT.read == 90
    assert client._client.max_retries == 1
