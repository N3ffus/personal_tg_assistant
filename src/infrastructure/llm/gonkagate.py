import json
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

from src.domain.assistant.business import BusinessDecision
from src.domain.assistant.context import ContextMessage
from src.domain.assistant.models import AssistantDecision
from src.infrastructure.llm.business import business_input, business_instructions
from src.infrastructure.llm.openai import (
    SUMMARY_PROMPT,
    SYSTEM_PROMPT,
    message_with_context,
)


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
        context: str = "",
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
            'Пример list_tasks: {"actions":[{"type":"list_tasks"}]}\n'
            'Пример update_event: {"actions":[{"type":"update_event",'
            '"title":"Стоматолог","starts_at":"2026-08-14T16:00:00+03:00"}]}\n'
            'Пример delete_event: {"actions":[{"type":"delete_event",'
            '"title":"Стоматолог"}]}\n'
            'Пример delete_task: {"actions":[{"type":"delete_task","title":"ENG-42"}]}\n'
            'Пример delete_all_tasks: {"actions":[{"type":"delete_all_tasks"}]}\n'
            'Пример delete_all_events: {"actions":[{"type":"delete_all_events"}]}\n'
            'Пример save_note: {"actions":[{"type":"save_note",'
            '"text":"Люблю Python"}]}\n'
            f"Текущее время: {now.isoformat()}\n"
            f"Timezone пользователя: {timezone}"
            "\nПеред ответом проверь ВСЕ ограничения запроса: период, текст, статус, "
            "сортировку. Явная сортировка ОБЯЗАТЕЛЬНО задаёт sort_by: "
            "например, «по названию» — title, «сначала ближайший срок» — due asc.\n"
            "JSON schema ответа:\n"
            + json.dumps(AssistantDecision.model_json_schema(), ensure_ascii=False)
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": message_with_context(text, context)},
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
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": business_instructions(now=now, timezone=timezone)
                    + "\nJSON schema: "
                    + json.dumps(BusinessDecision.model_json_schema()),
                },
                {
                    "role": "user",
                    "content": business_input(
                        history=history,
                        interlocutor=interlocutor,
                        owner_id=owner_id,
                        last_processed_message_id=last_processed_message_id,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("LLM returned no business decision")
        return BusinessDecision.model_validate_json(response.choices[0].message.content)

    async def summarize(self, *, text: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": text},
            ],
            max_completion_tokens=4096,
        )
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("LLM returned an empty summary")
        return response.choices[0].message.content
