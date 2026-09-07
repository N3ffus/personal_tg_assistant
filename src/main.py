import asyncio
import logging

import uvicorn
from aiogram import Bot, Dispatcher

from src.application.services.action_executor import (
    ActionExecutor,
)
from src.application.services.context import ContextService
from src.application.services.knowledge import KnowledgeService
from src.application.use_cases.process_business_dialog import ProcessBusinessDialog
from src.application.use_cases.process_message import (
    ProcessMessageUseCase,
)
from src.config import Settings
from src.infrastructure.calendar.google import GoogleCalendarClient
from src.infrastructure.calendar.oauth import GoogleOAuthService, create_oauth_app
from src.infrastructure.calendar.storage import CalendarStorage
from src.infrastructure.context.business_storage import BusinessStorage
from src.infrastructure.context.storage import ContextStorage
from src.infrastructure.knowledge.factory import create_knowledge_service
from src.infrastructure.llm.gonkagate import (
    GonkaGateLLMClient,
)
from src.infrastructure.tasks.linear import LinearTaskClient
from src.infrastructure.telegram.business import create_business_router
from src.infrastructure.telegram.business_worker import (
    BusinessDialogWorker,
    OwnerBusinessNotifier,
)
from src.infrastructure.telegram.calendar import create_calendar_router
from src.infrastructure.telegram.commands import configure_commands
from src.infrastructure.telegram.context import create_context_router
from src.infrastructure.telegram.handlers import (
    create_router,
)

POLLING_CONCURRENCY_LIMIT = 32


async def _run_services(
    *, dispatcher: Dispatcher, bot: Bot, server: uvicorn.Server
) -> None:
    polling = asyncio.create_task(
        dispatcher.start_polling(
            bot,
            handle_signals=False,
            close_bot_session=False,
            tasks_concurrency_limit=POLLING_CONCURRENCY_LIMIT,
        ),
        name="telegram-polling",
    )
    serving = asyncio.create_task(server.serve(), name="oauth-http-server")
    services = {polling, serving}

    try:
        done, _ = await asyncio.wait(
            services,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            task.result()
    finally:
        server.should_exit = True
        for task in services:
            if not task.done():
                task.cancel()
        await asyncio.gather(*services, return_exceptions=True)


async def main() -> None:
    settings = Settings()  # type: ignore[call-arg]

    llm = GonkaGateLLMClient(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    task_tracker: LinearTaskClient | None = None
    bot: Bot | None = None
    business_worker: BusinessDialogWorker | None = None
    knowledge: KnowledgeService | None = None
    try:
        storage = CalendarStorage(
            database_path=settings.database_path,
            encryption_key=settings.google_token_encryption_key.get_secret_value(),
        )
        await storage.initialize()
        context_storage = ContextStorage(
            database_path=settings.database_path,
            encryption_key=settings.google_token_encryption_key.get_secret_value(),
        )
        await context_storage.initialize()
        contexts = ContextService(storage=context_storage, summarizer=llm)
        calendar = GoogleCalendarClient(storage=storage)
        oauth = GoogleOAuthService(
            storage=storage,
            client_id=settings.google_oauth_client_id,
            client_secret=settings.google_oauth_client_secret.get_secret_value(),
            redirect_uri=settings.google_oauth_redirect_uri,
        )
        task_tracker = LinearTaskClient(
            api_key=settings.linear_api_key.get_secret_value(),
            team_id=settings.linear_team_id,
        )

        # One Graphiti client per process; a missing Neo4j only disables memory.
        knowledge = await create_knowledge_service(settings)

        action_executor = ActionExecutor(
            calendar=calendar,
            pending_operations=storage,
            task_tracker=task_tracker,
            knowledge=knowledge,
        )

        bot = Bot(token=settings.telegram_bot_token.get_secret_value())
        business_storage = BusinessStorage(
            database_path=settings.database_path,
            encryption_key=settings.google_token_encryption_key.get_secret_value(),
        )
        await business_storage.initialize()
        business_worker = BusinessDialogWorker(
            processor=ProcessBusinessDialog(
                contexts=contexts,
                storage=business_storage,
                llm=llm,
                executor=action_executor,
                owner_id=settings.telegram_allowed_user_id,
                timezone=settings.app_timezone,
                notify=OwnerBusinessNotifier(
                    bot=bot, owner_id=settings.telegram_allowed_user_id
                ),
            )
        )

        process_message = ProcessMessageUseCase(
            llm=llm,
            action_executor=action_executor,
            contexts=contexts,
            knowledge=knowledge,
        )

        dispatcher = Dispatcher()
        dispatcher.include_router(
            create_business_router(
                allowed_user_id=settings.telegram_allowed_user_id,
                contexts=contexts,
                extractor=business_worker,
            )
        )
        dispatcher.include_router(
            create_calendar_router(
                timezone=settings.app_timezone,
                allowed_user_id=settings.telegram_allowed_user_id,
                calendar=calendar,
                oauth=oauth,
                storage=storage,
            )
        )
        dispatcher.include_router(
            create_context_router(
                contexts=contexts, allowed_user_id=settings.telegram_allowed_user_id
            )
        )
        dispatcher.include_router(
            create_router(
                process_message=process_message,
                timezone=settings.app_timezone,
                allowed_user_id=settings.telegram_allowed_user_id,
                contexts=contexts,
            )
        )

        await configure_commands(bot)
        await business_worker.resume(
            contexts=contexts, owner_id=settings.telegram_allowed_user_id
        )

        server = uvicorn.Server(
            uvicorn.Config(
                create_oauth_app(oauth=oauth, knowledge=knowledge),
                host=settings.http_host,
                port=settings.http_port,
                log_level="info",
                access_log=False,
            )
        )
        await _run_services(dispatcher=dispatcher, bot=bot, server=server)
    finally:
        if business_worker is not None:
            await business_worker.close()
        try:
            if knowledge is not None:
                await knowledge.close()
        finally:
            try:
                if task_tracker is not None:
                    await task_tracker.close()
            finally:
                try:
                    await llm.close()
                finally:
                    if bot is not None:
                        await bot.session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
    )

    asyncio.run(main())
