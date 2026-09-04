from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    event_id: str
    title: str
    starts_at: datetime
    ends_at: datetime
    html_link: str | None
