import asyncio
import logging

from aiogram import Bot, Dispatcher

from src.application.services.action_executor import (
    ActionExecutor,
)
from src.application.use_cases.process_message import (
    ProcessMessageUseCase,
)
from src.config import Settings
from src.infrastructure.llm.gonkagate import (
    GonkaGateLLMClient,
)
from src.infrastructure.telegram.handlers import (
    create_router,
)


async def main() -> None:
    settings = Settings()  # type: ignore[call-arg]

    llm = GonkaGateLLMClient(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )

    action_executor = ActionExecutor()

    process_message = ProcessMessageUseCase(
        llm=llm,
        action_executor=action_executor,
    )

    router = create_router(
        process_message=process_message,
        timezone=settings.app_timezone,
    )

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    bot = Bot(
        token=settings.telegram_bot_token.get_secret_value(),
    )

    try:
        await dispatcher.start_polling(bot)

    finally:
        await llm.close()
        await bot.session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
    )

    asyncio.run(main())
