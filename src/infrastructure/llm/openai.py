from datetime import datetime

from openai import AsyncOpenAI

from src.domain.assistant.models import AssistantDecision

SYSTEM_PROMPT = """
Ты являешься маршрутизатором команд ИИ-помощника.

Определи, какое действие хочет выполнить пользователь.

Доступные действия:

1. chat
Обычный разговор, вопрос пользователю или ответ на вопрос.

2. create_task
Создать задачу, которую необходимо выполнить.
Например:
- "Добавь задачу купить продукты"
- "Мне нужно завтра закончить отчёт"

3. create_event
Создать событие, привязанное к конкретному времени.
Например:
- "Завтра в 15:00 стоматолог"
- "В пятницу в 19 встреча"

4. save_note
Сохранить информацию или заметку.
Например:
- "Запиши, что пароль от Wi-Fi лежит на роутере"
- "Сохрани идею сделать пост про архитектуру"

Правила:

- Не придумывай отсутствующие данные.
- create_event используй только если можно однозначно
  определить дату и время события.
- Если для выполнения действия не хватает информации,
  используй chat и задай уточняющий вопрос.
- Для относительных дат вроде "завтра", "в пятницу",
  используй переданное текущее время.
- starts_at должен содержать дату, время и timezone offset.
- Для обычных вопросов используй chat.
"""


class OpenAILLMClient:
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
            f"Текущее время: {now.isoformat()}\n"
            f"Timezone пользователя: {timezone}"
        )

        response = await self._client.responses.parse(
            model=self._model,
            instructions=instructions,
            input=text,
            text_format=AssistantDecision,
        )

        decision = response.output_parsed

        if decision is None:
            raise RuntimeError("LLM returned no parsed output")

        return decision

    async def close(self) -> None:
        await self._client.close()
