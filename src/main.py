import asyncio
import logging
from contextlib import AsyncExitStack

import uvicorn
from aiogram import Bot, Dispatcher

from src.application.services.action_executor import ActionExecutor
from src.application.services.context import ContextService
from src.application.use_cases.process_business_dialog import ProcessBusinessDialog
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.config import Settings
from src.infrastructure.calendar.google import GoogleCalendarClient
from src.infrastructure.calendar.oauth import GoogleOAuthService, create_oauth_app
from src.infrastructure.calendar.storage import CalendarStorage
from src.infrastructure.context.business_storage import BusinessStorage
from src.infrastructure.context.storage import ContextStorage
from src.infrastructure.films.tmdb import TMDBFilmDirectory
from src.infrastructure.knowledge.factory import create_knowledge_service
from src.infrastructure.llm.client import ChatLLMClient
from src.infrastructure.tasks.linear import LinearTaskClient
from src.infrastructure.telegram.business import create_business_router
from src.infrastructure.telegram.business_worker import (
    BusinessDialogWorker,
    OwnerBusinessNotifier,
)
from src.infrastructure.telegram.calendar import create_calendar_router
from src.infrastructure.telegram.commands import configure_commands
from src.infrastructure.telegram.context import create_context_router
from src.infrastructure.telegram.handlers import create_router

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
    encryption_key = settings.google_token_encryption_key.get_secret_value()
    owner_id = settings.telegram_allowed_user_id

    # Resources close in reverse order of opening, even when startup fails.
    async with AsyncExitStack() as resources:
        llm = ChatLLMClient(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
        resources.push_async_callback(llm.close)
        films = None
        if settings.tmdb_api_key.get_secret_value():
            films = TMDBFilmDirectory(api_key=settings.tmdb_api_key.get_secret_value())
            resources.push_async_callback(films.close)

        storage = CalendarStorage(
            database_path=settings.database_path, encryption_key=encryption_key
        )
        await storage.initialize()
        context_storage = ContextStorage(
            database_path=settings.database_path, encryption_key=encryption_key
        )
        await context_storage.initialize()
        business_storage = BusinessStorage(
            database_path=settings.database_path, encryption_key=encryption_key
        )
        await business_storage.initialize()

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
        resources.push_async_callback(task_tracker.close)

        # One Graphiti client per process; a missing Neo4j only disables memory.
        knowledge = await create_knowledge_service(settings)
        if knowledge is not None:
            resources.push_async_callback(knowledge.close)

        action_executor = ActionExecutor(
            calendar=calendar,
            pending_operations=storage,
            task_tracker=task_tracker,
            knowledge=knowledge,
        )
        bot = Bot(token=settings.telegram_bot_token.get_secret_value())
        resources.push_async_callback(bot.session.close)
        business_worker = BusinessDialogWorker(
            processor=ProcessBusinessDialog(
                contexts=contexts,
                storage=business_storage,
                llm=llm,
                executor=action_executor,
                owner_id=owner_id,
                timezone=settings.app_timezone,
                notify=OwnerBusinessNotifier(bot=bot, owner_id=owner_id),
            )
        )
        resources.push_async_callback(business_worker.close)
        process_message = ProcessMessageUseCase(
            llm=llm,
            action_executor=action_executor,
            contexts=contexts,
            knowledge=knowledge,
            films=films,
        )

        dispatcher = Dispatcher()
        dispatcher.include_routers(
            create_business_router(
                allowed_user_id=owner_id, contexts=contexts, extractor=business_worker
            ),
            create_calendar_router(
                timezone=settings.app_timezone,
                allowed_user_id=owner_id,
                calendar=calendar,
                oauth=oauth,
                storage=storage,
            ),
            create_context_router(contexts=contexts, allowed_user_id=owner_id),
            create_router(
                process_message=process_message,
                timezone=settings.app_timezone,
                allowed_user_id=owner_id,
                contexts=contexts,
            ),
        )

        await configure_commands(bot)
        await business_worker.resume(contexts=contexts, owner_id=owner_id)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
