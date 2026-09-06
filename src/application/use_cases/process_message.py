from datetime import datetime

from src.application.ports.llm import LLMClient
from src.application.services.action_executor import (
    ActionExecutor,
)
from src.application.services.context import ContextService
from src.application.services.explicit_commands import (
    parse_bulk_deletion,
    parse_view_request,
)
from src.domain.assistant.context import ContextMessage
from src.domain.assistant.replies import AssistantReply


class ProcessMessageUseCase:
    def __init__(
        self,
        *,
        llm: LLMClient,
        action_executor: ActionExecutor,
        contexts: ContextService | None = None,
    ) -> None:
        self._llm = llm
        self._action_executor = action_executor
        self._contexts = contexts

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
                text=text, now=now, timezone=timezone, user_id=user_id
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

        return await self._action_executor.execute_many(
            decision.actions,
            user_id=user_id,
            now=now,
        )

    async def resolve_deletion(
        self, *, user_id: int, operation_id: str, confirm: bool
    ) -> str:
        return await self._action_executor.resolve_deletion(
            user_id=user_id, operation_id=operation_id, confirm=confirm
        )
