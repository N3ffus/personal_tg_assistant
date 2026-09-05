from datetime import datetime

from src.application.ports.llm import LLMClient
from src.application.services.action_executor import (
    ActionExecutor,
)


class ProcessMessageUseCase:
    def __init__(
        self,
        *,
        llm: LLMClient,
        action_executor: ActionExecutor,
    ) -> None:
        self._llm = llm
        self._action_executor = action_executor

    async def execute(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
        user_id: int,
    ) -> str:
        decision = await self._llm.parse_message(
            text=text,
            now=now,
            timezone=timezone,
        )

        return await self._action_executor.execute_many(
            decision.actions,
            user_id=user_id,
            now=now,
        )
