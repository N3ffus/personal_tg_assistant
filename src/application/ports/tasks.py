from typing import Protocol

from src.domain.tasks.models import CreatedTask


class TaskTrackerError(Exception):
    """Expected task tracker integration failure."""


class TaskCreationUncertainError(TaskTrackerError):
    """The request may have succeeded despite a transport failure."""


class TaskTrackerClient(Protocol):
    async def create_task(self, *, title: str) -> CreatedTask: ...
