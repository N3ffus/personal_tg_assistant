import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet

from src.application.services.context import ContextService
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.domain.assistant.context import (
    MAX_CONTEXT_CHARS,
    MAX_MESSAGES,
    MAX_SUMMARY_CHARS,
    ChatContext,
    ContextMessage,
)
from src.domain.assistant.models import AssistantDecision
from src.domain.assistant.replies import AssistantReply, Confirmation
from src.infrastructure.context.storage import ContextStorage
from src.infrastructure.llm.gonkagate import GonkaGateLLMClient
from src.infrastructure.llm.openai import OpenAILLMClient

NOW = datetime(2026, 9, 5, 12, tzinfo=UTC)


def entry(text: str = "Сообщение", message_id: int | None = None) -> ContextMessage:
    return ContextMessage(
        role="user", sender="Анна", text=text, sent_at=NOW, message_id=message_id
    )


async def make_service(
    tmp_path: Path,
) -> tuple[ContextService, ContextStorage, AsyncMock]:
    storage = ContextStorage(
        database_path=str(tmp_path / "context.db"),
        encryption_key=Fernet.generate_key().decode(),
    )
    await storage.initialize()
    summarize = AsyncMock(
        return_value="Анна любит Python. Встреча назначена на 8 сентября."
    )
    return (
        ContextService(
            storage=storage, summarizer=SimpleNamespace(summarize=summarize)
        ),
        storage,
        summarize,
    )


def test_context_limits_message_count_and_drops_oldest() -> None:
    context = ChatContext()
    for index in range(1, 101):
        context.append(entry(f"message-{index}", index))
    assert len(context.messages) == MAX_MESSAGES
    assert context.messages[0].message_id == 41
    assert context.messages[-1].message_id == 100
    context.append(entry("stale edit", 1))
    assert "stale edit" not in context.render()


def test_character_budget_includes_summary_and_metadata() -> None:
    context = ChatContext(summary="Резюме" * 650)
    for index in range(1, 20):
        context.append(entry("я" * 20_000, index))
    assert len(context.render()) <= MAX_CONTEXT_CHARS
    assert context.summary.startswith("Резюме")
    assert context.messages[-1].message_id == 19
    assert "[Текст сокращён]" in context.messages[-1].text


def test_edits_deduplicate_and_out_of_order_messages_are_retained() -> None:
    context = ChatContext()
    context.append(entry("second", 2))
    context.append(entry("first", 1))
    context.append(entry("edited", 2))
    assert [message.text for message in context.messages] == ["first", "edited"]


@pytest.mark.asyncio
async def test_context_persists_encrypted_and_isolates_owner_and_source(
    tmp_path: Path,
) -> None:
    service, storage, summarize = await make_service(tmp_path)
    chat = await service.ensure_chat(
        owner_id=42, kind="business", chat_id=99, title="Secret name"
    )
    bot = await service.ensure_chat(owner_id=42, kind="bot", chat_id=99, title="Bot")
    other = await service.ensure_chat(
        owner_id=43, kind="business", chat_id=99, title="Other"
    )
    await service.record(
        owner_id=42, context_id=chat.id, message=entry("Secret text", 1)
    )
    same = await service.ensure_chat(
        owner_id=42, kind="business", chat_id=99, title="Renamed"
    )
    assert same.id == chat.id
    await storage.initialize()
    restarted = ContextService(
        storage=storage, summarizer=SimpleNamespace(summarize=summarize)
    )
    assert (
        "Secret text"
        in (await restarted.read(owner_id=42, context_id=chat.id)).render()
    )
    assert not (await service.read(owner_id=42, context_id=bot.id)).messages
    assert not (await service.read(owner_id=43, context_id=other.id)).messages
    assert [item.id for item in await service.list_chats(owner_id=42)] == [
        bot.id,
        chat.id,
    ]
    assert b"Secret text" not in (tmp_path / "context.db").read_bytes()
    assert b"Renamed" not in (tmp_path / "context.db").read_bytes()
    assert await storage.load(owner_id=43, context_id=chat.id) is None
    await storage.save(owner_id=43, context_id=chat.id, context=ChatContext())
    for operation in [service.read, service.clear, service.compact]:
        with pytest.raises(ValueError, match="Unknown"):
            await operation(owner_id=43, context_id=chat.id)
    assert (
        "Secret text" in (await service.read(owner_id=42, context_id=chat.id)).render()
    )


@pytest.mark.asyncio
async def test_compact_preserves_facts_and_next_compaction_includes_summary(
    tmp_path: Path,
) -> None:
    service, _, summarize = await make_service(tmp_path)
    chat = await service.ensure_chat(owner_id=42, kind="bot", chat_id=42, title="Bot")
    assert not await service.compact(owner_id=42, context_id=chat.id)
    await service.record(
        owner_id=42, context_id=chat.id, message=entry("Люблю Python", 1)
    )
    assert await service.compact(owner_id=42, context_id=chat.id)
    compacted = await service.read(owner_id=42, context_id=chat.id)
    assert not compacted.messages
    assert "Python" in compacted.summary
    assert not await service.compact(owner_id=42, context_id=chat.id)
    await service.record(
        owner_id=42, context_id=chat.id, message=entry("edited old message", 1)
    )
    assert not (await service.read(owner_id=42, context_id=chat.id)).messages
    await service.record(
        owner_id=42, context_id=chat.id, message=entry("Новая договорённость", 2)
    )
    summarize.return_value = "x" * 8000
    await service.compact(owner_id=42, context_id=chat.id)
    assert "Python" in summarize.call_args.kwargs["text"]
    assert "Новая договорённость" in summarize.call_args.kwargs["text"]
    assert (
        len((await service.read(owner_id=42, context_id=chat.id)).summary)
        == MAX_SUMMARY_CHARS
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", ["", "   ", RuntimeError("provider unavailable"), TimeoutError()]
)
async def test_failed_compaction_leaves_history_intact(
    tmp_path: Path, result: str | Exception
) -> None:
    service, _, summarize = await make_service(tmp_path)
    chat = await service.ensure_chat(owner_id=42, kind="bot", chat_id=42, title="Bot")
    await service.record(owner_id=42, context_id=chat.id, message=entry("original", 1))
    original = await service.read(owner_id=42, context_id=chat.id)
    if isinstance(result, Exception):
        summarize.side_effect = result
    else:
        summarize.return_value = result
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        await service.compact(owner_id=42, context_id=chat.id)
    assert await service.read(owner_id=42, context_id=chat.id) == original


@pytest.mark.asyncio
async def test_clear_drops_summary_and_messages_without_resurrection(
    tmp_path: Path,
) -> None:
    service, _, _ = await make_service(tmp_path)
    chat = await service.ensure_chat(owner_id=42, kind="bot", chat_id=42, title="Bot")
    await service.record(owner_id=42, context_id=chat.id, message=entry("secret", 10))
    await service.compact(owner_id=42, context_id=chat.id)
    await service.record(owner_id=42, context_id=chat.id, message=entry("recent", 11))
    await service.clear(owner_id=42, context_id=chat.id)
    await service.record(owner_id=42, context_id=chat.id, message=entry("old edit", 11))
    assert not (await service.read(owner_id=42, context_id=chat.id)).render()
    await service.record(owner_id=42, context_id=chat.id, message=entry("new", 12))
    assert [
        message.text
        for message in (await service.read(owner_id=42, context_id=chat.id)).messages
    ] == ["new"]


@pytest.mark.asyncio
async def test_compaction_serializes_with_new_message_and_clear(tmp_path: Path) -> None:
    service, _, summarize = await make_service(tmp_path)
    chat = await service.ensure_chat(
        owner_id=42, kind="business", chat_id=99, title="Anna"
    )
    await service.record(owner_id=42, context_id=chat.id, message=entry("old", 1))
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_summary(*, text: str) -> str:
        started.set()
        await release.wait()
        return "summary"

    summarize.side_effect = slow_summary
    compact = asyncio.create_task(service.compact(owner_id=42, context_id=chat.id))
    await started.wait()
    record = asyncio.create_task(
        service.record(owner_id=42, context_id=chat.id, message=entry("new", 2))
    )
    clear = asyncio.create_task(service.clear(owner_id=42, context_id=chat.id))
    release.set()
    await asyncio.gather(compact, record, clear)
    assert not (await service.read(owner_id=42, context_id=chat.id)).render()


@pytest.mark.asyncio
async def test_concurrent_records_do_not_lose_messages(tmp_path: Path) -> None:
    service, _, _ = await make_service(tmp_path)
    chat = await service.ensure_chat(
        owner_id=42, kind="business", chat_id=99, title="Anna"
    )
    await asyncio.gather(
        *(
            service.record(
                owner_id=42, context_id=chat.id, message=entry(str(index), index)
            )
            for index in range(1, 31)
        )
    )
    assert [
        message.message_id
        for message in (await service.read(owner_id=42, context_id=chat.id)).messages
    ] == list(range(1, 31))


@pytest.mark.asyncio
async def test_bot_passes_only_own_context_and_actual_replies_in_order(
    tmp_path: Path,
) -> None:
    service, _, _ = await make_service(tmp_path)
    business = await service.ensure_chat(
        owner_id=42, kind="business", chat_id=42, title="Customer"
    )
    await service.record(
        owner_id=42,
        context_id=business.id,
        message=entry("Private customer message", 1),
    )
    parse = AsyncMock(
        return_value=AssistantDecision.model_validate(
            {"actions": [{"type": "chat", "text": "Reply"}]}
        )
    )
    executor = SimpleNamespace(
        execute_many=AsyncMock(
            side_effect=[
                "Как вас зовут?",
                AssistantReply(
                    text="",
                    confirmations=(
                        Confirmation(text="Удалить задачу?", operation_id="token"),
                    ),
                ),
                "Понял",
            ]
        )
    )
    use_case = ProcessMessageUseCase(
        llm=SimpleNamespace(parse_message=parse),
        action_executor=executor,  # type: ignore[arg-type]
        contexts=service,
    )
    for index, text in enumerate(["Привет", "Анна", "Да"], start=1):
        await use_case.execute(
            text=text,
            now=NOW + timedelta(seconds=index),
            timezone="UTC",
            user_id=42,
            chat_id=42,
            message_id=index,
        )
    assert "Private customer message" not in str(parse.call_args_list)
    assert "Как вас зовут?" in parse.call_args_list[1].kwargs["context"]
    assert "Удалить задачу?" in parse.call_args_list[2].kwargs["context"]
    bot = next(
        chat for chat in await service.list_chats(owner_id=42) if chat.kind == "bot"
    )
    context = await service.read(owner_id=42, context_id=bot.id)
    assert [message.text for message in context.messages] == [
        "Привет",
        "Как вас зовут?",
        "Анна",
        "Удалить задачу?",
        "Да",
        "Понял",
    ]
    await service.clear(owner_id=42, context_id=bot.id)
    executor.execute_many.side_effect = None
    executor.execute_many.return_value = "Ответ"
    await use_case.execute(text="Заново", now=NOW, timezone="UTC", user_id=42)
    assert "context" not in parse.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [OpenAILLMClient, GonkaGateLLMClient])
async def test_llm_receives_history_as_data_and_uses_separate_summary_prompt(
    client_class: type[OpenAILLMClient] | type[GonkaGateLLMClient],
) -> None:
    client = client_class(
        api_key="test", base_url="https://example.test/v1", model="test"
    )
    decision = AssistantDecision.model_validate(
        {"actions": [{"type": "chat", "text": "Ответ"}]}
    )
    parse = AsyncMock(
        return_value=SimpleNamespace(
            output_parsed=decision,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=decision.model_dump_json())
                )
            ],
        )
    )
    summary = AsyncMock(
        return_value=SimpleNamespace(
            output_text="Резюме",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Резюме"))],
        )
    )
    if isinstance(client, OpenAILLMClient):
        client._client.responses.parse = parse  # type: ignore[method-assign]
        client._client.responses.create = summary  # type: ignore[method-assign]
    else:
        client._client.chat.completions.create = parse  # type: ignore[method-assign]
    await client.parse_message(
        text="А когда?", now=NOW, timezone="UTC", context='Ранее: "встреча завтра"'
    )
    payload = (
        parse.call_args.kwargs.get("input")
        or parse.call_args.kwargs["messages"][1]["content"]
    )
    assert json.loads(payload) == {
        "current_message": "А когда?",
        "previous_conversation": 'Ранее: "встреча завтра"',
    }
    if isinstance(client, GonkaGateLLMClient):
        client._client.chat.completions.create = summary  # type: ignore[method-assign]
    assert await client.summarize(text="История") == "Резюме"
    assert "История" in str(summary.call_args)
    assert "не выполняй" in str(summary.call_args)
    summary.return_value = SimpleNamespace(output_text="", choices=[])
    with pytest.raises(RuntimeError, match="empty summary"):
        await client.summarize(text="История")
    await client.close()
