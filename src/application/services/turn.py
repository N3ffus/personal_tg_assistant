from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter

from src.application.ports.llm import LLMClient
from src.domain.assistant.models import AssistantDecision


@dataclass(frozen=True, slots=True)
class Turn:
    """One user message being answered, with everything a follow-up LLM call needs."""

    text: str
    now: datetime
    timezone: str
    user_id: int
    context: str = ""
    # Seconds each LLM call of this turn took. A slow reply is then read as
    # "four calls" or "one slow call" instead of being blamed on the graph.
    llm_seconds: list[float] = field(default_factory=list)

    async def ask(
        self,
        llm: LLMClient,
        *,
        knowledge: str = "",
        text: str | None = None,
        with_context: bool = True,
    ) -> AssistantDecision:
        started = perf_counter()
        try:
            return await llm.parse_message(
                text=self.text if text is None else text,
                now=self.now,
                timezone=self.timezone,
                **({"knowledge": knowledge} if knowledge else {}),
                **({"context": self.context} if self.context and with_context else {}),
            )
        finally:
            self.llm_seconds.append(perf_counter() - started)
