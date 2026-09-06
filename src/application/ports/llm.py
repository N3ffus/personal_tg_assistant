from datetime import datetime
from typing import Protocol

from src.domain.assistant.business import BusinessDecision
from src.domain.assistant.context import ContextMessage
from src.domain.assistant.models import AssistantDecision


class BusinessDialogLLM(Protocol):
    async def parse_business_dialog(
        self,
        *,
        history: list[ContextMessage],
        interlocutor: str,
        owner_id: int,
        last_processed_message_id: int,
        now: datetime,
        timezone: str,
    ) -> BusinessDecision: ...


class LLMClient(Protocol):
    async def parse_message(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
        context: str = "",
    ) -> AssistantDecision: ...
