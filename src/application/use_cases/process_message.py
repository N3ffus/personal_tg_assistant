import json
import logging
from datetime import datetime

from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.ports.llm import LLMClient
from src.application.services.action_executor import (
    ActionExecutor,
)
from src.application.services.context import ContextService
from src.application.services.explicit_commands import (
    parse_bulk_deletion,
    parse_view_request,
)
from src.application.services.knowledge import KnowledgeService
from src.domain.assistant.context import ContextMessage
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    AssistantAction,
    AssistantDecision,
    ChatAction,
    SearchKnowledgeAction,
)
from src.domain.assistant.replies import AssistantReply, ResultPage
from src.domain.knowledge.models import KnowledgeFact

logger = logging.getLogger(__name__)


class ProcessMessageUseCase:
    def __init__(
        self,
        *,
        llm: LLMClient,
        action_executor: ActionExecutor,
        contexts: ContextService | None = None,
        knowledge: KnowledgeService | None = None,
    ) -> None:
        self._llm = llm
        self._action_executor = action_executor
        self._contexts = contexts
        self._knowledge = knowledge

    async def execute(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
        user_id: int,
        chat_id: int | None = None,
        message_id: int | None = None,
    ) -> str | AssistantReply:
        if self._contexts is None:
            return await self._respond(
                text=text,
                now=now,
                timezone=timezone,
                user_id=user_id,
                message_id=message_id,
            )
        chat = await self._contexts.ensure_chat(
            owner_id=user_id,
            kind="bot",
            chat_id=chat_id if chat_id is not None else user_id,
            title="Чат с ботом",
        )
        async with self._contexts.session(
            owner_id=user_id, context_id=chat.id
        ) as context:
            response = await self._respond(
                text=text,
                now=now,
                timezone=timezone,
                user_id=user_id,
                message_id=message_id,
                context=context.render(),
            )
            context.append(
                ContextMessage(
                    role="user",
                    sender="Вы",
                    text=text,
                    sent_at=now,
                    message_id=message_id,
                )
            )
            reply_text = (
                response
                if isinstance(response, str)
                else "\n\n".join(
                    [
                        response.text,
                        *(confirmation.text for confirmation in response.confirmations),
                        *(page.text for page in response.pages),
                    ]
                ).strip()
            )
            context.append(
                ContextMessage(
                    role="assistant", sender="Бот", text=reply_text, sent_at=now
                )
            )
            return response

    async def _respond(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
        user_id: int,
        message_id: int | None = None,
        context: str = "",
    ) -> str | AssistantReply:
        decision = (
            parse_view_request(text)
            or parse_bulk_deletion(text)
            or await self._llm.parse_message(
                text=text,
                now=now,
                timezone=timezone,
                **({"context": context} if context else {}),
            )
        )
        decision = await self._with_recalled_knowledge(
            decision,
            text=text,
            now=now,
            timezone=timezone,
            user_id=user_id,
            context=context,
        )

        return await self._action_executor.execute_many(
            decision.actions,
            user_id=user_id,
            now=now,
            message_id=message_id,
        )

    async def _with_recalled_knowledge(
        self,
        decision: AssistantDecision,
        *,
        text: str,
        now: datetime,
        timezone: str,
        user_id: int,
        context: str,
    ) -> AssistantDecision:
        """Answer memory questions with the facts the graph actually holds.

        Retrieval is tool-based: only a decision that asked for a search pays for
        the second model call.
        """
        searches = [
            action
            for action in decision.actions
            if isinstance(action, SearchKnowledgeAction)
        ]
        if not searches or self._knowledge is None:
            return decision
        recalled, facts = await self._recall(searches, user_id=user_id)
        informed = await self._llm.parse_message(
            text=text,
            now=now,
            timezone=timezone,
            knowledge=recalled,
            **({"context": context} if context else {}),
        )
        actions: list[AssistantAction] = [
            action
            for action in informed.actions
            # A second search would loop: answer from what memory already returned.
            if not isinstance(action, SearchKnowledgeAction)
        ]
        if not actions:
            return AssistantDecision(
                actions=[
                    ChatAction(
                        type=ActionType.CHAT,
                        text=ActionExecutor.format_facts(facts),
                    )
                ]
            )
        return AssistantDecision(actions=actions)

    async def _recall(
        self, searches: list[SearchKnowledgeAction], *, user_id: int
    ) -> tuple[str, list[KnowledgeFact]]:
        if self._knowledge is None:  # pragma: no cover - guarded by the caller
            return "[]", []
        facts: list[KnowledgeFact] = []
        for search in searches:
            try:
                found = await self._knowledge.search(
                    user_id=user_id, query=search.query, limit=search.limit
                )
            except KnowledgeMemoryError as error:
                logger.warning("knowledge.search.degraded reason=%s", error)
                return (
                    json.dumps(
                        {"error": "knowledge memory is temporarily unavailable"},
                        ensure_ascii=False,
                    ),
                    [],
                )
            facts.extend(fact for fact in found if fact not in facts)
        payload = [fact.as_payload() for fact in facts]
        return json.dumps(payload, ensure_ascii=False), facts

    async def resolve_deletion(
        self, *, user_id: int, operation_id: str, confirm: bool
    ) -> str:
        return await self._action_executor.resolve_deletion(
            user_id=user_id, operation_id=operation_id, confirm=confirm
        )

    async def browse(self, *, user_id: int, data: str) -> ResultPage | str:
        return await self._action_executor.browse(user_id=user_id, data=data)
