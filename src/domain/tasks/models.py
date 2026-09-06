from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class CreatedTask:
    identifier: str
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class Task:
    identifier: str
    title: str
    url: str
    status: str
    description: str = ""
    status_type: str = ""
    priority: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    due_date: date | None = None
    project: str = ""
    assignee: str = ""
    labels: tuple[str, ...] = ()
