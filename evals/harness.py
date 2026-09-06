from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from evals.scenarios import Scenario
from src.application.ports.calendar import CalendarError, CalendarNotConnectedError
from src.application.ports.llm import LLMClient
from src.application.ports.tasks import TaskCreationUncertainError, TaskTrackerError
from src.application.services.action_executor import ActionExecutor
from src.application.services.explicit_commands import (
    parse_bulk_deletion,
    parse_view_request,
)
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.domain.assistant.deletions import DeletionTarget
from src.domain.assistant.models import AssistantDecision
from src.domain.assistant.replies import AssistantReply
from src.domain.assistant.retrieval import EventQuery, TaskQuery
from src.domain.calendar.models import CalendarEvent
from src.domain.tasks.models import CreatedTask, Task


@dataclass
class RecordedCall:
    name: str
    arguments: dict[str, object]
    output: object = None
    error: str | None = None


@dataclass
class RecordingIntegrations:
    """In-memory ports: capture actual executor calls, including failed attempts."""

    scenario: Scenario
    calls: list[RecordedCall] = field(default_factory=list)
    task_count: int = 0
    selection_consumed: bool = False

    async def find_tasks(self, *, title: str | None) -> list[DeletionTarget]:
        targets = [
            DeletionTarget(
                id="eval-task",
                title=title or "Тестовая задача",
                label=f"EVAL-1: {title or 'Тестовая задача'}",
            )
        ]
        self.calls.append(
            RecordedCall(
                "linear.find_tasks", {"title": title}, [t.model_dump() for t in targets]
            )
        )
        return targets

    async def delete_task(self, *, task_id: str) -> None:
        raise AssertionError("Natural-language deletion must require confirmation")

    async def find_events(
        self, *, user_id: int, title: str | None
    ) -> list[DeletionTarget]:
        targets = [
            DeletionTarget(
                id="eval-event",
                title=title or "Встреча",
                label=f"06.09.2026 12:00 — {title or 'Встреча'}",
            )
        ]
        self.calls.append(
            RecordedCall(
                "calendar.find_events",
                {"user_id": user_id, "title": title},
                [t.model_dump() for t in targets],
            )
        )
        return targets

    async def create_operation(
        self, *, user_id: int, kind: str, payload: dict[str, object]
    ) -> str:
        assert user_id == self.scenario.user_id
        assert kind == "delete_confirm"
        return "eval-confirmation"

    async def consume_operation(
        self, *, operation_id: str, user_id: int
    ) -> tuple[str, dict[str, object]] | None:
        raise AssertionError("Natural-language input must not consume confirmations")

    async def list_tasks(self, *, query: TaskQuery | None = None) -> list[Task]:
        self.calls.append(RecordedCall("linear.list_tasks", {}))
        if self.scenario.failure == "linear_error":
            self.calls[-1].error = "TaskTrackerError"
            raise TaskTrackerError()
        tasks = [
            Task(
                identifier="EVAL-1",
                title="Подготовить отчёт",
                status="In Progress",
                url="https://linear.example/issue/EVAL-1",
            )
        ]
        self.calls[-1].output = [asdict(task) for task in tasks]
        return tasks

    async def create_task(self, *, title: str) -> CreatedTask:
        self.calls.append(RecordedCall("linear.create_task", {"title": title}))
        self.task_count += 1
        if self.task_count == 1:
            if self.scenario.failure == "linear_uncertain":
                self.calls[-1].error = "TaskCreationUncertainError"
                raise TaskCreationUncertainError()
            if self.scenario.failure == "linear_error":
                self.calls[-1].error = "TaskTrackerError"
                raise TaskTrackerError()
        task = CreatedTask(
            identifier=f"EVAL-{self.task_count}",
            title=title,
            url=f"https://linear.example/issue/EVAL-{self.task_count}",
        )
        self.calls[-1].output = asdict(task)
        return task

    def _check_calendar(self) -> None:
        if self.scenario.failure == "calendar_disconnected":
            self.calls[-1].error = "CalendarNotConnectedError"
            raise CalendarNotConnectedError()
        if self.scenario.failure == "calendar_error":
            self.calls[-1].error = "CalendarError"
            raise CalendarError()

    @staticmethod
    def _event_data(event: CalendarEvent) -> dict[str, object]:
        return {
            "event_id": event.event_id,
            "title": event.title,
            "starts_at": event.starts_at.isoformat(),
            "html_link": event.html_link,
        }

    @staticmethod
    def _event(title: str, starts_at: datetime, event_id: str) -> CalendarEvent:
        return CalendarEvent(
            event_id=event_id,
            title=title,
            starts_at=starts_at,
            ends_at=starts_at + timedelta(hours=1),
            html_link=f"https://calendar.example/event/{event_id}",
        )

    async def create_event(
        self, *, user_id: int, title: str, starts_at: datetime
    ) -> CalendarEvent:
        self.calls.append(
            RecordedCall(
                "calendar.create_event",
                {
                    "user_id": user_id,
                    "title": title,
                    "starts_at": starts_at.isoformat(),
                },
            )
        )
        self._check_calendar()
        event = self._event(title, starts_at, "created-event")
        self.calls[-1].output = self._event_data(event)
        return event

    async def list_events(
        self, *, user_id: int, now: datetime, query: EventQuery | None = None
    ) -> list[CalendarEvent]:
        self.calls.append(
            RecordedCall(
                "calendar.list_events", {"user_id": user_id, "now": now.isoformat()}
            )
        )
        self._check_calendar()
        if self.scenario.empty_calendar:
            self.calls[-1].output = []
            return []
        events = [
            self._event("Встреча команды", now + timedelta(hours=2), "listed-event")
        ]
        self.calls[-1].output = [self._event_data(event) for event in events]
        return events

    async def update_event(
        self, *, user_id: int, event_id: str, title: str, starts_at: datetime
    ) -> CalendarEvent:
        self.calls.append(
            RecordedCall(
                "calendar.update_event",
                {
                    "user_id": user_id,
                    "event_id": event_id,
                    "title": title,
                    "starts_at": starts_at.isoformat(),
                },
            )
        )
        self._check_calendar()
        event = self._event(title, starts_at, event_id)
        self.calls[-1].output = self._event_data(event)
        return event

    async def delete_event(self, *, user_id: int, event_id: str) -> None:
        self.calls.append(
            RecordedCall(
                "calendar.delete_event", {"user_id": user_id, "event_id": event_id}
            )
        )
        raise AssertionError("Natural-language deletion must require confirmation")

    async def disconnect(self, *, user_id: int) -> None:
        raise AssertionError("A prompt must not disconnect a calendar")

    async def consume_latest_operation(
        self, *, user_id: int, kind: str
    ) -> dict[str, object] | None:
        assert user_id == self.scenario.user_id
        assert kind == "update"
        if self.scenario.selected_event_id is None or self.selection_consumed:
            return None
        self.selection_consumed = True
        return {"event_id": self.scenario.selected_event_id}


@dataclass
class RecordingLLM:
    inner: LLMClient
    decision: AssistantDecision | None = None

    async def parse_message(
        self, *, text: str, now: datetime, timezone: str, context: str = ""
    ) -> AssistantDecision:
        self.decision = await self.inner.parse_message(
            text=text, now=now, timezone=timezone, context=context
        )
        return self.decision


@dataclass
class EvaluationRun:
    decision: AssistantDecision
    calls: list[RecordedCall]
    reply: str


async def run_scenario(scenario: Scenario, llm: LLMClient) -> EvaluationRun:
    integrations = RecordingIntegrations(scenario)
    recording_llm = RecordingLLM(llm)
    use_case = ProcessMessageUseCase(
        llm=recording_llm,
        action_executor=ActionExecutor(
            calendar=integrations,
            pending_operations=integrations,
            task_tracker=integrations,
        ),
    )
    reply = await use_case.execute(
        text=scenario.prompt,
        now=scenario.now,
        timezone=scenario.timezone,
        user_id=scenario.user_id,
    )
    decision = (
        recording_llm.decision
        or parse_view_request(scenario.prompt)
        or parse_bulk_deletion(scenario.prompt)
    )
    assert decision is not None
    if isinstance(reply, AssistantReply):
        reply = "\n\n".join(
            [
                reply.text,
                *(c.text for c in reply.confirmations),
                *(p.text for p in reply.pages),
            ]
        )
    return EvaluationRun(decision, integrations.calls, reply)
