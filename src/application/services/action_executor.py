import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, tzinfo
from html import escape

from src.application.ports.calendar import (
    CalendarClient,
    CalendarError,
    CalendarNotConnectedError,
    PendingOperationStore,
)
from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.ports.tasks import (
    TaskCreationUncertainError,
    TaskTrackerClient,
    TaskTrackerError,
)
from src.application.services.deletions import DeletionService
from src.application.services.knowledge import KnowledgeService
from src.application.services.retrieval import RetrievalService
from src.domain.assistant.business import (
    BusinessAction,
    BusinessEvent,
    BusinessNote,
    BusinessTask,
)
from src.domain.assistant.models import (
    AssistantAction,
    ChatAction,
    CreateEventAction,
    CreateTaskAction,
    DeleteAllEventsAction,
    DeleteAllTasksAction,
    DeleteEventAction,
    DeleteTaskAction,
    ListEventsAction,
    ListTasksAction,
    RememberKnowledgeAction,
    SaveNoteAction,
    SearchKnowledgeAction,
    UpdateEventAction,
)
from src.domain.assistant.replies import AssistantReply, Confirmation, ResultPage
from src.domain.assistant.retrieval import EventQuery, RetrievalLimitError, TaskQuery
from src.domain.calendar.models import CalendarEvent
from src.domain.knowledge.models import KnowledgeFact, KnowledgeSourceType
from src.domain.tasks.models import Task

logger = logging.getLogger(__name__)

MEMORY_UNAVAILABLE = "⚠️ Долговременная память сейчас недоступна, попробуйте позже."


class ActionExecutor:
    def __init__(
        self,
        *,
        calendar: CalendarClient,
        pending_operations: PendingOperationStore,
        task_tracker: TaskTrackerClient,
        knowledge: KnowledgeService | None = None,
    ) -> None:
        self._calendar = calendar
        self._pending_operations = pending_operations
        self._task_tracker = task_tracker
        self._knowledge = knowledge
        self._retrieval = RetrievalService(calendar=calendar, task_tracker=task_tracker)
        self._deletions = DeletionService(
            calendar=calendar, task_tracker=task_tracker, storage=pending_operations
        )

    async def execute_business(
        self,
        action: BusinessAction,
        *,
        user_id: int,
        now: datetime | None = None,
        source_id: str | None = None,
    ) -> str:
        # Deliberately separate from execute(): business input cannot reach destructive actions.
        if isinstance(action, BusinessTask):
            task = await self._task_tracker.create_task(title=action.title)
            return (
                f'✅ Создана задача в Linear: <a href="{escape(task.url, quote=True)}">'
                f"{escape(task.identifier)}</a> — {escape(task.title)}"
            )
        if isinstance(action, BusinessEvent):
            event = await self._calendar.create_event(
                user_id=user_id,
                title=action.title,
                starts_at=action.starts_at,
            )
            return (
                f"📅 Запланировано событие: {escape(event.title)} "
                f"({event.starts_at:%d.%m.%Y %H:%M %z})"
            )
        if isinstance(action, BusinessNote):
            # The caller commits the encrypted note before delivering this notification.
            await self._remember_quietly(
                content=action.text,
                user_id=user_id,
                now=now,
                source_id=source_id,
            )
            return f"📝 Сохранена заметка: {escape(action.text)}"
        raise ValueError("Forbidden business action")

    async def _remember_quietly(
        self,
        *,
        content: str,
        user_id: int,
        now: datetime | None,
        source_id: str | None,
    ) -> None:
        """Feed a stored note into memory without ever failing its delivery."""
        if self._knowledge is None or now is None or source_id is None:
            return
        try:
            await self._remember(
                content=content,
                user_id=user_id,
                now=now,
                message_id=None,
                source_type=KnowledgeSourceType.BUSINESS_NOTE,
                source_id=source_id,
            )
        except KnowledgeMemoryError as error:
            logger.warning("knowledge.remember.skipped reason=%s", error)

    async def resolve_deletion(
        self, *, user_id: int, operation_id: str, confirm: bool
    ) -> str:
        return await self._deletions.resolve(
            user_id=user_id, operation_id=operation_id, confirm=confirm
        )

    async def execute_many(
        self,
        actions: Sequence[AssistantAction],
        *,
        user_id: int,
        now: datetime,
        message_id: int | None = None,
    ) -> str | AssistantReply:
        if not actions:
            raise ValueError("At least one action is required")

        responses: list[str] = []
        confirmations: list[Confirmation] = []
        pages: list[ResultPage] = []
        for action in actions:
            try:
                response = await self.execute(
                    action, user_id=user_id, now=now, message_id=message_id
                )
            except (RetrievalLimitError, TimeoutError):
                if not isinstance(action, (ListTasksAction, ListEventsAction)):
                    raise
                response = "Выборка слишком большая или поиск занял слишком много времени. Уточните период, слово, проект или статус и повторите запрос."
            except TaskCreationUncertainError:
                if not isinstance(action, CreateTaskAction):
                    raise
                response = (
                    f"⚠️ Linear мог создать задачу «{action.title}». "
                    "Проверьте список задач перед повтором."
                )
            except TaskTrackerError:
                if isinstance(action, ListTasksAction):
                    response = "❌ Не удалось получить задачи Linear. Проверьте доступ и попробуйте позже."
                elif isinstance(action, (DeleteTaskAction, DeleteAllTasksAction)):
                    response = "❌ Не удалось получить задачи Linear для удаления. Проверьте доступ и попробуйте позже."
                elif isinstance(action, CreateTaskAction):
                    response = (
                        f"❌ Не удалось создать задачу «{action.title}» в Linear."
                    )
                else:
                    raise
            except KnowledgeMemoryError:
                if not isinstance(
                    action,
                    (RememberKnowledgeAction, SearchKnowledgeAction, SaveNoteAction),
                ):
                    raise
                response = MEMORY_UNAVAILABLE
            except CalendarNotConnectedError:
                if not self._is_calendar_action(action):
                    raise
                response = (
                    f"❌ {self._calendar_action_label(action)}: подключите календарь "
                    "командой /calendar_connect."
                )
            except CalendarError:
                if not self._is_calendar_action(action):
                    raise
                response = (
                    f"❌ {self._calendar_action_label(action)}: календарь недоступен."
                )
            if isinstance(response, AssistantReply):
                if response.text:
                    responses.append(response.text)
                confirmations.extend(response.confirmations)
                pages.extend(response.pages)
            else:
                responses.append(response)

        text = "\n\n".join(responses)
        return (
            AssistantReply(
                text=text, confirmations=tuple(confirmations), pages=tuple(pages)
            )
            if confirmations or pages
            else text
        )

    async def execute(
        self,
        action: AssistantAction,
        *,
        user_id: int,
        now: datetime,
        message_id: int | None = None,
    ) -> str | AssistantReply:
        if isinstance(action, ChatAction):
            return action.text

        if isinstance(action, CreateTaskAction):
            task = await self._task_tracker.create_task(title=action.title)
            return f"✅ Создана задача {task.identifier}: {task.title}\n{task.url}"

        if isinstance(action, ListTasksAction):
            query = TaskQuery.model_validate(action.model_dump(exclude={"type"}))
            page = await self._retrieval.open(query=query, user_id=user_id, now=now)
            return AssistantReply(text="", confirmations=(), pages=(page,))

        if isinstance(action, (DeleteTaskAction, DeleteAllTasksAction)):
            return await self._deletions.prepare(
                user_id=user_id,
                resource="linear",
                title=action.title if isinstance(action, DeleteTaskAction) else None,
            )

        if isinstance(action, CreateEventAction):
            event = await self._calendar.create_event(
                user_id=user_id,
                title=action.title,
                starts_at=action.starts_at,
            )
            return self._event_result("Создано событие", event)

        if isinstance(action, ListEventsAction):
            event_query = EventQuery.model_validate(action.model_dump(exclude={"type"}))
            page = await self._retrieval.open(
                query=event_query, user_id=user_id, now=now
            )
            return AssistantReply(text="", confirmations=(), pages=(page,))

        if isinstance(action, UpdateEventAction):
            targets = await self._calendar.find_events(
                user_id=user_id, title=action.event_title
            )
            if not targets:
                return (
                    f"Событие «{action.event_title}» не найдено. "
                    "Уточните его текущее название."
                )
            if len(targets) > 1:
                # Editing is not confirmable like deletion: never guess the event.
                listing = "\n".join(f"• {target.label}" for target in targets)
                return (
                    "Найдено несколько подходящих событий. Уточните, какое изменить:\n"
                    f"{listing}"
                )
            event = await self._calendar.update_event(
                user_id=user_id,
                event_id=targets[0].id,
                title=action.title,
                starts_at=action.starts_at,
            )
            return self._event_result("Событие изменено", event)

        if isinstance(action, (DeleteEventAction, DeleteAllEventsAction)):
            return await self._deletions.prepare(
                user_id=user_id,
                resource="calendar",
                title=action.title if isinstance(action, DeleteEventAction) else None,
            )

        if isinstance(action, SaveNoteAction):
            stored = await self._remember(
                content=action.text,
                user_id=user_id,
                now=now,
                message_id=message_id,
                source_type=KnowledgeSourceType.NOTE,
            )
            prefix = "Заметка сохранена" if stored else "Понял, нужно сохранить заметку"
            return f"📝 {prefix}:\n{action.text}"

        if isinstance(action, RememberKnowledgeAction):
            if self._knowledge is None:
                return "⚠️ Долговременная память отключена, сохранить факт не могу."
            await self._remember(
                content=action.content,
                user_id=user_id,
                now=now,
                message_id=message_id,
            )
            return f"🧠 Запомнил: {action.content}"

        if isinstance(action, SearchKnowledgeAction):
            if self._knowledge is None:
                return "⚠️ Долговременная память отключена, поиск по ней недоступен."
            facts = await self._knowledge.search(
                user_id=user_id, query=action.query, limit=action.limit
            )
            return self.format_facts(facts)

        raise ValueError(f"Unsupported action: {type(action)!r}")

    async def _remember(
        self,
        *,
        content: str,
        user_id: int,
        now: datetime,
        message_id: int | None,
        source_type: KnowledgeSourceType = KnowledgeSourceType.TELEGRAM_MESSAGE,
        source_id: str | None = None,
    ) -> bool:
        """Persist user-authored knowledge; returns whether memory accepted it."""
        if self._knowledge is None:
            return False
        await self._knowledge.remember(
            user_id=user_id,
            content=content,
            source_id=source_id or self._source_id(message_id=message_id, now=now),
            source_type=source_type,
            # The message time is the moment the user stated the fact.
            reference_time=now,
        )
        return True

    @staticmethod
    def _source_id(*, message_id: int | None, now: datetime) -> str:
        return str(message_id) if message_id is not None else f"{now:%Y%m%dT%H%M%S%z}"

    @staticmethod
    def format_facts(facts: list[KnowledgeFact]) -> str:
        if not facts:
            return "🧠 В долговременной памяти ничего не нашлось по этому запросу."
        lines = ["🧠 Нашёл в памяти:"]
        for fact in facts:
            line = f"• {fact.fact}"
            if fact.valid_from:
                line += f" (с {fact.valid_from:%d.%m.%Y}"
                line += (
                    f", до {fact.valid_until:%d.%m.%Y})" if fact.valid_until else ")"
                )
            elif fact.valid_until:
                line += f" (до {fact.valid_until:%d.%m.%Y})"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def format_tasks(tasks: list[Task]) -> str:
        if not tasks:
            return "В настроенной команде Linear нет неархивных задач."
        lines = [f"📋 Задачи Linear — {len(tasks)} (без архивных):"]
        for task in tasks:
            lines.append(
                f"• {task.identifier}: {task.title} — {task.status}\n{task.url}"
            )
        return "\n".join(lines)

    async def browse(self, *, user_id: int, data: str) -> ResultPage | str:
        return await self._retrieval.navigate(user_id=user_id, data=data)

    @staticmethod
    def format_events(
        events: list[CalendarEvent], *, timezone: tzinfo | None = None
    ) -> str:
        if not events:
            return "Ближайших событий не найдено."
        lines = ["📅 Ближайшие события Google Calendar (до 10):"]
        for event in events:
            if event.all_day:
                when = f"{event.starts_at:%d.%m.%Y}"
                last_day = event.ends_at - timedelta(days=1)
                if last_day.date() > event.starts_at.date():
                    when += f"–{last_day:%d.%m.%Y}"
                when += " (весь день)"
            else:
                start = (
                    event.starts_at.astimezone(timezone)
                    if timezone
                    else event.starts_at
                )
                when = f"{start:%d.%m.%Y %H:%M %Z}"
            lines.append(f"• {when} — {event.title}")
            if event.html_link:
                lines.append(event.html_link)
        return "\n".join(lines)

    @staticmethod
    def _event_result(verb: str, event: CalendarEvent) -> str:
        result = f"✅ {verb}: {event.title}\nВремя: {event.starts_at:%d.%m.%Y %H:%M}"
        return f"{result}\n{event.html_link}" if event.html_link else result

    @staticmethod
    def _is_calendar_action(action: AssistantAction) -> bool:
        return isinstance(
            action,
            (
                CreateEventAction,
                ListEventsAction,
                UpdateEventAction,
                DeleteEventAction,
                DeleteAllEventsAction,
            ),
        )

    @staticmethod
    def _calendar_action_label(action: AssistantAction) -> str:
        if isinstance(action, CreateEventAction):
            return f"Не удалось создать событие «{action.title}»"
        if isinstance(action, UpdateEventAction):
            return f"Не удалось изменить событие «{action.title}»"
        if isinstance(action, DeleteEventAction):
            return f"Не удалось удалить событие «{action.title}»"
        if isinstance(action, DeleteAllEventsAction):
            return "Не удалось подготовить удаление событий"
        if isinstance(action, ListEventsAction):
            return "Не удалось получить события"
        raise ValueError(f"Not a calendar action: {type(action)!r}")
