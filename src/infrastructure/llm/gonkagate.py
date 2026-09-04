import json
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

from src.domain.assistant.models import AssistantDecision
from src.infrastructure.llm.openai import SYSTEM_PROMPT


class GonkaGateLLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model

    async def parse_message(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
    ) -> AssistantDecision:
        instructions = (
            f"{SYSTEM_PROMPT}\n\n"
            "Верни только корректный JSON без Markdown и пояснений.\n"
            "Поле action всегда должно быть объектом с полем type.\n"
            'Пример chat: {"action":{"type":"chat","text":"Привет!"}}\n'
            'Пример create_task: {"action":{"type":"create_task",'
            '"title":"Купить продукты"}}\n'
            'Пример create_event: {"action":{"type":"create_event",'
            '"title":"Стоматолог","starts_at":"2026-08-13T15:00:00+03:00"}}\n'
            'Пример list_events: {"action":{"type":"list_events"}}\n'
            'Пример update_event: {"action":{"type":"update_event",'
            '"title":"Стоматолог","starts_at":"2026-08-14T16:00:00+03:00"}}\n'
            'Пример delete_event: {"action":{"type":"delete_event",'
            '"title":"Стоматолог"}}\n'
            'Пример save_note: {"action":{"type":"save_note",'
            '"text":"Люблю Python"}}\n'
            f"Текущее время: {now.isoformat()}\n"
            f"Timezone пользователя: {timezone}"
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )

        if not response.choices:
            raise RuntimeError("LLM returned no choices")

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError("LLM returned an empty response")

        decision_data: Any = json.loads(content)

        if isinstance(decision_data, dict) and isinstance(
            decision_data.get("action"), str
        ):
            action_type = decision_data.pop("action")
            decision_data = {"action": {"type": action_type, **decision_data}}

        return AssistantDecision.model_validate(decision_data)

    async def close(self) -> None:
        await self._client.close()
