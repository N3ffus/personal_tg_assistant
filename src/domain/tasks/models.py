from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CreatedTask:
    identifier: str
    title: str
    url: str
