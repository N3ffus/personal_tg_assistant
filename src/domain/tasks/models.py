from dataclasses import dataclass


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
