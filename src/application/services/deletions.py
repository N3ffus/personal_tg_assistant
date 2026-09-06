import logging

from pydantic import ValidationError

from src.application.ports.calendar import (
    CalendarClient,
    CalendarError,
    CalendarEventNotFoundError,
    PendingOperationStore,
)
from src.application.ports.tasks import (
    TaskDeletionUncertainError,
    TaskNotFoundError,
    TaskTrackerClient,
    TaskTrackerError,
)
from src.domain.assistant.deletions import (
    DeletionPlan,
    DeletionResource,
    DeletionTarget,
)
from src.domain.assistant.replies import AssistantReply, Confirmation

logger = logging.getLogger(__name__)
DELETION_KIND = "delete_confirm"
INVALID_CONFIRMATION = (
    "Подтверждение недействительно или устарело. Повторите запрос удаления."
)


class DeletionService:
    def __init__(
        self,
        *,
        calendar: CalendarClient,
        task_tracker: TaskTrackerClient,
        storage: PendingOperationStore,
    ) -> None:
        self._calendar = calendar
        self._task_tracker = task_tracker
        self._storage = storage

    async def prepare(
        self, *, user_id: int, resource: DeletionResource, title: str | None
    ) -> str | AssistantReply:
        # A blank search must never become a request to delete everything.
        if title is not None and not title.strip():
            return "Укажите название или идентификатор для удаления."
        targets = (
            await self._task_tracker.find_tasks(title=title)
            if resource == "linear"
            else await self._calendar.find_events(user_id=user_id, title=title)
        )
        targets = list({target.id: target for target in targets}.values())
        if not targets:
            return (
                "Задач в Linear не найдено."
                if resource == "linear"
                else "Событий в календаре не найдено."
            )
        ambiguous = title is not None and len(targets) > 1
        groups = [[target] for target in targets] if ambiguous else [targets]
        confirmations = []
        for group in groups:
            plan = DeletionPlan(resource=resource, targets=group)
            operation_id = await self._storage.create_operation(
                user_id=user_id, kind=DELETION_KIND, payload=plan.model_dump()
            )
            confirmations.append(
                Confirmation(
                    text=self._preview(
                        resource=resource, targets=group, all_items=title is None
                    ),
                    operation_id=operation_id,
                )
            )
        return AssistantReply(
            text="Найдено несколько совпадений. Выберите нужное для удаления."
            if ambiguous
            else "",
            confirmations=tuple(confirmations),
        )

    @staticmethod
    def _preview(
        *, resource: DeletionResource, targets: list[DeletionTarget], all_items: bool
    ) -> str:
        scope = (
            "задачи настроенной команды Linear (включая архивные)"
            if resource == "linear"
            else "события основного Google Calendar (включая прошедшие и повторяющиеся серии)"
        )
        heading = (
            f"Удалить все {scope}?"
            if all_items
            else (
                "Удалить задачу в Linear?"
                if resource == "linear"
                else "Удалить событие в Google Calendar?"
            )
        )
        # The complete snapshot is shown, split into Telegram messages by the adapter.
        listing = "\n".join(f"• {target.label}" for target in targets)
        return f"{heading}\nКоличество: {len(targets)}\n\n{listing}\n\nПодтвердите кнопкой ниже."

    async def resolve(self, *, user_id: int, operation_id: str, confirm: bool) -> str:
        # Atomic DELETE ... RETURNING in storage makes both buttons single-use.
        operation = await self._storage.consume_operation(
            operation_id=operation_id, user_id=user_id
        )
        if operation is None or operation[0] != DELETION_KIND:
            return INVALID_CONFIRMATION
        try:
            plan = DeletionPlan.model_validate(operation[1])
        except ValidationError:
            return INVALID_CONFIRMATION
        if not confirm:
            return "Удаление отменено."
        deleted = 0
        missing = 0
        failures: list[str] = []
        for target in plan.targets:
            try:
                if plan.resource == "linear":
                    await self._task_tracker.delete_task(task_id=target.id)
                else:
                    await self._calendar.delete_event(
                        user_id=user_id, event_id=target.id
                    )
            except (CalendarEventNotFoundError, TaskNotFoundError):
                missing += 1
            except TaskDeletionUncertainError:
                failures.append(
                    f"• {target.label}: результат неизвестен, проверьте Linear."
                )
            except (TaskTrackerError, CalendarError):
                failures.append(
                    f"• {target.label}: не удалось подтвердить удаление, проверьте сервис."
                )
            except Exception:
                logger.exception(
                    "Unexpected deletion failure for resource %s", plan.resource
                )
                failures.append(
                    f"• {target.label}: результат неизвестен, проверьте сервис."
                )
            else:
                deleted += 1
        service = "Linear" if plan.resource == "linear" else "Google Calendar"
        result = f"{service}. Удалено: {deleted} из {len(plan.targets)}."
        if missing:
            result += f" Уже удалено или не найдено: {missing}."
        if failures:
            result += f"\nНе подтверждено: {len(failures)}.\n" + "\n".join(failures)
            result += "\nПроверьте список и повторите запрос для оставшихся объектов."
        return result
