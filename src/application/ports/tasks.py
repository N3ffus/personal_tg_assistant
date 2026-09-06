from typing import Protocol

from src.domain.assistant.deletions import DeletionTarget
from src.domain.assistant.retrieval import TaskQuery
from src.domain.tasks.models import CreatedTask, Task


class TaskTrackerError(Exception):
    """Expected task tracker integration failure."""


class TaskCreationUncertainError(TaskTrackerError):
    """The request may have succeeded despite a transport failure."""


class TaskDeletionUncertainError(TaskTrackerError):
    """The deletion may have succeeded despite a transport failure."""


class TaskNotFoundError(TaskTrackerError):
    """The task was not found or was already deleted."""


class TaskTrackerClient(Protocol):
    async def create_task(self, *, title: str) -> CreatedTask: ...

    async def list_tasks(self, *, query: TaskQuery | None = None) -> list[Task]: ...

    async def find_tasks(self, *, title: str | None) -> list[DeletionTarget]: ...

    async def delete_task(self, *, task_id: str) -> None: ...
