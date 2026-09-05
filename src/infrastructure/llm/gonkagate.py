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
            "Корневой объект всегда содержит массив actions.\n"
            "Каждый элемент actions всегда является объектом с полем type.\n"
            'Пример chat: {"actions":[{"type":"chat","text":"Привет!"}]}\n'
            'Пример create_task: {"actions":[{"type":"create_task",'
            '"title":"Купить продукты"}]}\n'
            'Пример create_event: {"actions":[{"type":"create_event",'
            '"title":"Стоматолог","starts_at":"2026-08-13T15:00:00+03:00"}]}\n'
            'Пример list_events: {"actions":[{"type":"list_events"}]}\n'
            'Пример update_event: {"actions":[{"type":"update_event",'
            '"title":"Стоматолог","starts_at":"2026-08-14T16:00:00+03:00"}]}\n'
            'Пример delete_event: {"actions":[{"type":"delete_event",'
            '"title":"Стоматолог"}]}\n'
            'Пример save_note: {"actions":[{"type":"save_note",'
            '"text":"Люблю Python"}]}\n'
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

        decision_data = self._normalize_legacy_decision(decision_data)

        return AssistantDecision.model_validate(decision_data)

    @staticmethod
    def _normalize_legacy_decision(decision_data: Any) -> Any:
        if not isinstance(decision_data, dict) or "actions" in decision_data:
            return decision_data

        action = decision_data.get("action")
        if isinstance(action, dict):
            return {"actions": [action]}
        if isinstance(action, str):
            payload = {
                key: value for key, value in decision_data.items() if key != "action"
            }
            return {"actions": [{"type": action, **payload}]}
        return decision_data

    async def close(self) -> None:
        await self._client.close()
