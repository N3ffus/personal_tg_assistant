from datetime import datetime
from typing import Protocol

from src.domain.assistant.models import AssistantDecision


class LLMClient(Protocol):
    async def parse_message(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
    ) -> AssistantDecision: ...
