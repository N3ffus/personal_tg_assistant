import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from html import escape
from weakref import WeakValueDictionary
from zoneinfo import ZoneInfo

from src.application.ports.business import BusinessStore
from src.application.ports.llm import BusinessDialogLLM
from src.application.services.action_executor import ActionExecutor
from src.application.services.context import ContextService
from src.domain.assistant.business import (
    BusinessDecision,
    BusinessIntent,
    BusinessNote,
    BusinessTask,
)
from src.domain.assistant.context import ContextChat, ContextMessage

logger = logging.getLogger(__name__)
BusinessNotifier = Callable[[str, str], Awaitable[None]]


def _evidence_message(
    message_id: int | None,
    quote: str,
    *,
    messages: dict[int, ContextMessage],
    sources: set[int],
    owner_id: int,
    from_owner: bool,
) -> ContextMessage | None:
    message = messages.get(message_id) if message_id is not None else None
    if (
        message is None
        or message_id not in sources
        or (message.sender_id == owner_id) != from_owner
        or message.is_forwarded
        or not quote.strip()
        or quote not in message.text
    ):
        return None
    return message


def supported_intents(
    decision: BusinessDecision,
    *,
    history: list[ContextMessage],
    owner_id: int,
    cursor: int,
) -> list[BusinessIntent]:
    # Revalidate even objects returned by an alternate LLM implementation.
    decision = BusinessDecision.model_validate(decision.model_dump())
    messages = {
        m.message_id: m
        for m in history
        if m.message_id is not None
        and m.sender_id is not None
        and not m.is_business_bot
    }
    accepted = []
    for intent in decision.actions:
        sources = set(intent.source_message_ids)
        if not sources.issubset(messages) or max(sources) <= cursor:
            continue
        confirmation = _evidence_message(
            intent.owner_confirmation_message_id,
            intent.owner_confirmation_quote,
            messages=messages,
            sources=sources,
            owner_id=owner_id,
            from_owner=True,
        )
        new_confirmation = (
            confirmation is not None and (confirmation.message_id or 0) > cursor
        )
        if isinstance(intent.action, BusinessTask):
            # Authorship alone is not task ownership. Missing/unclear ownership
            # must never fall back to creating a task for the owner.
            if intent.task_assignee != "owner":
                continue
            request = _evidence_message(
                intent.peer_request_message_id,
                intent.peer_request_quote,
                messages=messages,
                sources=sources,
                owner_id=owner_id,
                from_owner=False,
            )
            if intent.task_basis == "peer_request":
                if (
                    request is None
                    or (request.message_id or 0) <= cursor
                    or intent.owner_confirmation_message_id is not None
                    or intent.owner_confirmation_quote
                ):
                    continue
            elif intent.task_basis == "owner_commitment":
                if (
                    not new_confirmation
                    or intent.peer_request_message_id is not None
                    or intent.peer_request_quote
                ):
                    continue
            elif intent.task_basis == "owner_acceptance":
                if (
                    not new_confirmation
                    or confirmation is None
                    or request is None
                    or (request.message_id or 0) >= (confirmation.message_id or 0)
                    or (
                        confirmation.reply_to_message_id is not None
                        and confirmation.reply_to_message_id != request.message_id
                    )
                ):
                    continue
            else:
                continue
        elif not isinstance(intent.action, BusinessNote) and not new_confirmation:
            continue
        accepted.append(intent)
    return accepted


class ProcessBusinessDialog:
    def __init__(
        self,
        *,
        contexts: ContextService,
        storage: BusinessStore,
        llm: BusinessDialogLLM,
        executor: ActionExecutor,
        owner_id: int,
        timezone: str,
        notify: BusinessNotifier,
    ) -> None:
        self._contexts = contexts
        self._storage = storage
        self._llm = llm
        self._executor = executor
        self._owner_id = owner_id
        self._timezone = timezone
        self._notify = notify
        self._locks: WeakValueDictionary[int, asyncio.Lock] = WeakValueDictionary()

    async def process(self, chat: ContextChat) -> None:
        if chat.owner_id != self._owner_id or chat.kind != "business":
            return
        lock = self._locks.setdefault(chat.id, asyncio.Lock())
        async with lock:
            await self._deliver(chat)
            context = await self._contexts.read(
                owner_id=self._owner_id, context_id=chat.id
            )
            cursor = await self._storage.cursor(
                owner_id=self._owner_id, context_id=chat.id
            )
            history = [
                m
                for m in context.messages
                if m.sender_id is not None
                and not m.is_business_bot
                and m.message_id is not None
            ]
            if not history or max(m.message_id or 0 for m in history) <= cursor:
                return
            last_id = max(m.message_id or 0 for m in history)
            decision = await asyncio.wait_for(
                self._llm.parse_business_dialog(
                    history=history,
                    interlocutor=chat.title,
                    owner_id=self._owner_id,
                    last_processed_message_id=cursor,
                    now=datetime.now(ZoneInfo(self._timezone)),
                    timezone=self._timezone,
                ),
                timeout=90,
            )
            # Clear/compact during the LLM call must not resurrect discarded messages.
            current = await self._contexts.read(
                owner_id=self._owner_id, context_id=chat.id
            )
            history = [m for m in history if m in current.messages]
            intents = supported_intents(
                decision, history=history, owner_id=self._owner_id, cursor=cursor
            )
            logger.info(
                "Business dialog analyzed: context=%s through=%s proposed=%s accepted=%s",
                chat.id,
                last_id,
                len(decision.actions),
                len(intents),
            )
            await self._storage.prepare(
                owner_id=self._owner_id,
                context_id=chat.id,
                expected_cursor=cursor,
                last_message_id=last_id,
                intents=intents,
            )
            await self._deliver(chat)

    async def _deliver(self, chat: ContextChat) -> None:
        for item in await self._storage.outstanding(
            owner_id=self._owner_id, context_id=chat.id
        ):
            result = item.result
            if item.status != "completed":
                result = "⚠️ Выполнение было прервано. Проверьте Linear/календарь перед ручным повтором."
                if item.status == "pending":
                    if not await self._storage.claim(
                        owner_id=self._owner_id, action_id=item.id
                    ):
                        continue
                    try:
                        result = await asyncio.wait_for(
                            self._executor.execute_business(
                                item.action, user_id=self._owner_id
                            ),
                            timeout=60,
                        )
                    except Exception as error:
                        logger.warning(
                            "Business action failed: %s", type(error).__name__
                        )
                        label = (
                            item.action.text
                            if isinstance(item.action, BusinessNote)
                            else item.action.title
                        )
                        result = (
                            f"⚠️ Не удалось подтвердить выполнение: {escape(label)}. "
                            "Проверьте Linear/календарь перед ручным повтором."
                        )
                await self._storage.complete(
                    owner_id=self._owner_id, action_id=item.id, result=result
                )
            if result is not None:
                await self._notify(chat.title, result)
                await self._storage.notified(owner_id=self._owner_id, action_id=item.id)
