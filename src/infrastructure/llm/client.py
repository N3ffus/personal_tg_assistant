import json
import logging
from datetime import datetime
from time import perf_counter
from typing import Any, get_args

from openai import AsyncOpenAI, Timeout
from pydantic import ValidationError

from src.domain.assistant.business import (
    BusinessDecision,
    BusinessEvent,
    BusinessIntent,
    BusinessNote,
    BusinessTask,
)
from src.domain.assistant.context import ContextMessage
from src.domain.assistant.models import AssistantDecision
from src.infrastructure.llm.business import business_input, business_instructions
from src.infrastructure.llm.prompts import (
    SUMMARY_PROMPT,
    decision_instructions,
    message_with_context,
)

logger = logging.getLogger(__name__)

# The fields each business action really has. A model that adds one more must
# not cost the owner the whole decision, so the extras are dropped and logged
# instead of failing ``extra="forbid"`` validation.
BUSINESS_ACTION_FIELDS = {
    get_args(model.model_fields["type"].annotation)[0].value: frozenset(
        model.model_fields
    )
    for model in (BusinessTask, BusinessEvent, BusinessNote)
}
BUSINESS_INTENT_FIELDS = frozenset(BusinessIntent.model_fields)

# The SDK default is 600 s with two retries: a hung provider kept a message
# unanswered for up to half an hour. The slowest real call on production was a
# 36 s answer turn (2026-09-22), so 90 s leaves room without hiding a hang.
# openai bundles its own HTTP client: use its Timeout, not httpx's.
REQUEST_TIMEOUT = Timeout(90, connect=10)
MAX_SDK_RETRIES = 1
# One more try when the model's reply cannot become a decision at all.
DECISION_ATTEMPTS = 2


class EmptyResponseError(RuntimeError):
    """The provider answered, but with no content to parse."""


class ChatLLMClient:
    """Any OpenAI-compatible chat-completions provider in JSON mode."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=REQUEST_TIMEOUT,
            max_retries=MAX_SDK_RETRIES,
        )
        self._model = model

    async def close(self) -> None:
        await self._client.close()

    async def parse_message(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
        context: str = "",
        knowledge: str = "",
    ) -> AssistantDecision:
        instructions = decision_instructions(
            now=now, timezone=timezone, knowledge=knowledge
        )
        message = message_with_context(text, context, knowledge)
        stage = "answer" if knowledge else "decision"
        # An empty actions array (memory eval, 2026-09-17), truncated JSON or a
        # schema slip each cost the user the whole reply. Asking once more is
        # far cheaper than answering nothing; the last failure still surfaces.
        for attempt in range(1, DECISION_ATTEMPTS):
            try:
                decision = normalize_decision(
                    await self._json(instructions, message, stage)
                )
                if decision.get("actions"):
                    return AssistantDecision.model_validate(decision)
                logger.warning("llm.decision.empty attempt=%s", attempt)
            except (json.JSONDecodeError, ValidationError, EmptyResponseError) as error:
                logger.warning(
                    "llm.decision.invalid attempt=%s error=%s",
                    attempt,
                    type(error).__name__,
                )
        decision = normalize_decision(await self._json(instructions, message, stage))
        return AssistantDecision.model_validate(decision)

    async def parse_business_dialog(
        self,
        *,
        history: list[ContextMessage],
        interlocutor: str,
        owner_id: int,
        last_processed_message_id: int,
        now: datetime,
        timezone: str,
    ) -> BusinessDecision:
        instructions = (
            business_instructions(now=now, timezone=timezone)
            + "\nJSON schema: "
            + json.dumps(BusinessDecision.model_json_schema())
        )
        message = business_input(
            history=history,
            interlocutor=interlocutor,
            owner_id=owner_id,
            last_processed_message_id=last_processed_message_id,
        )
        payload = await self._json(instructions, message, "business")
        return BusinessDecision.model_validate(without_business_extras(payload))

    async def summarize(self, *, text: str) -> str:
        content = await self._complete(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": text},
            ],
            stage="summary",
            max_completion_tokens=4096,
        )
        if not content.strip():
            raise RuntimeError("LLM returned an empty summary")
        return content

    async def _json(self, instructions: str, message: str, stage: str) -> Any:
        content = await self._complete(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": message},
            ],
            stage=stage,
            response_format={"type": "json_object"},
        )
        if not content:
            raise EmptyResponseError("LLM returned an empty response")
        return json.loads(content)

    async def _complete(
        self, messages: list[dict[str, str]], *, stage: str, **options: Any
    ) -> str:
        started = perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            **options,
        )
        usage = getattr(response, "usage", None)
        logger.info(
            "llm.%s.completed latency_ms=%.0f prompt_tokens=%s completion_tokens=%s "
            "cached_tokens=%s reasoning_tokens=%s",
            stage,
            (perf_counter() - started) * 1000,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
            ),
            getattr(
                getattr(usage, "completion_tokens_details", None),
                "reasoning_tokens",
                None,
            ),
        )
        if not response.choices:
            raise RuntimeError("LLM returned no choices")
        return response.choices[0].message.content or ""


def normalize_decision(data: Any) -> dict[str, Any]:
    """Coerce the shapes models actually return into ``{"actions": [...]}``."""
    if not isinstance(data, dict):
        return {}
    if "actions" in data:
        if len(data) > 1:
            # A field the model hoisted out of an action (it put a
            # recommendation's `limit` here) must not fail the whole reply:
            # the decision is its actions, and the strays carry no meaning.
            strays = [key for key in data if key != "actions"]
            logger.warning("llm.decision.extra_fields keys=%s", strays)
        return {"actions": data["actions"]}
    # Legacy single-action shapes: {"action": {...}} or {"action": "chat", ...}.
    action = data.get("action")
    if isinstance(action, dict):
        return {"actions": [action]}
    if isinstance(action, str):
        payload = {key: value for key, value in data.items() if key != "action"}
        return {"actions": [{"type": action, **payload}]}
    return data


def without_business_extras(payload: Any) -> Any:
    """Drop fields the business schema does not declare, keeping the rest.

    A single stray key anywhere in the decision otherwise fails validation and
    loses every task the model found. Only known action types are trimmed: an
    unknown or forbidden one keeps its extras and is still rejected, so this
    never widens the executable allowlist.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        return payload
    strays: list[str] = []

    def trimmed(data: Any, allowed: frozenset[str], where: str) -> Any:
        if not isinstance(data, dict):
            return data
        strays.extend(f"{where}.{key}" for key in data if key not in allowed)
        return {key: value for key, value in data.items() if key in allowed}

    intents = []
    for intent in payload["actions"]:
        intent = trimmed(intent, BUSINESS_INTENT_FIELDS, "intent")
        if isinstance(intent, dict) and isinstance(intent.get("action"), dict):
            allowed = BUSINESS_ACTION_FIELDS.get(intent["action"].get("type"))
            if allowed is not None:
                intent = {
                    **intent,
                    "action": trimmed(intent["action"], allowed, "action"),
                }
        intents.append(intent)
    if strays:
        logger.warning("llm.business.extra_fields keys=%s", strays)
    return {**payload, "actions": intents}
