from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

SearchText = Annotated[str, Field(min_length=1, max_length=160)]
MAX_RETRIEVAL_ITEMS = 5000
MAX_RETRIEVAL_PAGES = 100


class RetrievalLimitError(Exception):
    """A complete result would exceed the bounded retrieval budget."""


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: SearchText | None = None
    search_terms: list[SearchText] = Field(default_factory=list, max_length=8)
    date_from: AwareDatetime | None = None
    date_to: AwareDatetime | None = None
    direction: Literal["asc", "desc"] = "asc"
    page_size: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.date_from and self.date_to and self.date_from >= self.date_to:
            raise ValueError("date_to must be later than date_from (exclusive end)")
        return self

    def matches_text(self, *values: str) -> bool:
        text = "\n".join(values).casefold()
        return (self.query is None or self.query.casefold() in text) and (
            not self.search_terms
            or any(term.casefold() in text for term in self.search_terms)
        )


class TaskQuery(SearchQuery):
    status: SearchText | None = None
    status_group: Literal["all", "open", "completed", "canceled"] = "all"
    project: SearchText | None = None
    assignee: SearchText | None = None
    label: SearchText | None = None
    priority: int | None = Field(default=None, ge=0, le=4)
    include_archived: bool = False
    date_field: Literal["created", "updated", "due", "completed"] = "created"
    sort_by: Literal[
        "created", "updated", "due", "completed", "priority", "title", "status"
    ] = "created"
    direction: Literal["asc", "desc"] = "desc"


class EventQuery(SearchQuery):
    sort_by: Literal["start", "updated", "title"] = "start"
    all_day: bool | None = None

    def with_default_range(self, now: datetime) -> Self:
        from datetime import timedelta

        # Recurring events may be infinite: every expanded search needs an end.
        start = self.date_from or (
            self.date_to - timedelta(days=365) if self.date_to else now
        )
        end = self.date_to or start + timedelta(days=365)
        return self.model_copy(update={"date_from": start, "date_to": end})
