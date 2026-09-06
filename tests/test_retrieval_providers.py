import json
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from src.application.ports.calendar import CalendarError
from src.application.ports.tasks import TaskTrackerError
from src.domain.assistant.retrieval import EventQuery, RetrievalLimitError, TaskQuery
from src.infrastructure.calendar import google as google_module
from src.infrastructure.calendar.google import GoogleCalendarClient
from src.infrastructure.tasks import linear as linear_module
from src.infrastructure.tasks.linear import LinearTaskClient, task_filter

START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2027, 1, 1, tzinfo=UTC)


def issue(number: int) -> dict[str, Any]:
    return {
        "identifier": f"APP-{number}",
        "title": "Отчёт",
        "url": f"https://linear.test/{number}",
        "state": {"name": "Done", "type": "completed"},
        "description": "Годовой отчёт",
        "priority": 2,
        "createdAt": "2026-01-01T09:00:00Z",
        "updatedAt": "2026-09-06T09:00:00Z",
        "completedAt": "2026-09-06T09:00:00Z",
        "dueDate": "2026-09-07",
        "project": {"name": "Работа"},
        "assignee": {"name": "Andrey"},
    }


def issue_page(
    items: list[dict[str, Any]], cursor: str | None = None
) -> dict[str, Any]:
    return {
        "data": {
            "issues": {
                "nodes": items,
                "pageInfo": {"hasNextPage": cursor is not None, "endCursor": cursor},
            }
        }
    }


@pytest.mark.asyncio
async def test_linear_all_filters_and_archive_are_preserved_on_every_cursor() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json=issue_page([issue(2)])
            if payload["variables"]["after"]
            else issue_page([issue(1)], "next"),
        )

    query = TaskQuery(
        query="отчёт",
        search_terms=["год", "месяц"],
        status="Done",
        status_group="completed",
        date_field="completed",
        date_from=START,
        date_to=END,
        project="Работа",
        assignee="Andrey",
        label="Finance",
        priority=2,
        include_archived=True,
    )
    client = LinearTaskClient(
        api_key="test", team_id="team-id", transport=httpx.MockTransport(handler)
    )
    try:
        tasks = await client.list_tasks(query=query)
    finally:
        await client.close()
    assert len(tasks) == 2
    assert tasks[0].created_at == datetime(2026, 1, 1, 9, tzinfo=UTC)
    assert tasks[0].due_date == date(2026, 9, 7)
    assert tasks[0].priority == 2 and tasks[0].project == "Работа"
    assert tasks[0].assignee == "Andrey" and tasks[0].status_type == "completed"
    filters = {
        "and": [
            {"team": {"id": {"eq": "team-id"}}},
            {
                "or": [
                    {"title": {"containsIgnoreCase": "отчёт"}},
                    {"description": {"containsIgnoreCase": "отчёт"}},
                ]
            },
            {
                "or": [
                    {field: {"containsIgnoreCase": term}}
                    for term in ("год", "месяц")
                    for field in ("title", "description")
                ]
            },
            {"state": {"name": {"eqIgnoreCase": "Done"}}},
            {"state": {"type": {"in": ["completed"]}}},
            {"project": {"name": {"containsIgnoreCase": "Работа"}}},
            {"assignee": {"name": {"containsIgnoreCase": "Andrey"}}},
            {"labels": {"name": {"eqIgnoreCase": "Finance"}}},
            {"priority": {"eq": 2}},
            {"completedAt": {"gte": START.isoformat(), "lt": END.isoformat()}},
        ]
    }
    assert [r["variables"]["after"] for r in requests] == [None, "next"]
    assert all(
        r["variables"]["filter"] == filters
        and r["variables"]["includeArchived"] is True
        for r in requests
    )


def test_linear_due_dates_use_local_dates_and_open_types() -> None:
    query = TaskQuery(
        status_group="open",
        date_field="due",
        date_from=datetime.fromisoformat("2026-09-07T00:00:00+03:00"),
        date_to=END,
    )
    filters = task_filter("team", query)["and"]
    assert isinstance(filters, list)
    assert {"dueDate": {"gte": "2026-09-07", "lt": "2027-01-01"}} in filters
    assert {
        "state": {"type": {"in": ["triage", "backlog", "unstarted", "started"]}}
    } in filters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {"dueDate": "not-date"},
        {"priority": 9},
        {"createdAt": "2026-09-01"},
        {"updatedAt": 42},
        {"completedAt": "invalid"},
        {"project": "invalid"},
    ],
)
async def test_linear_invalid_metadata_never_produces_a_partial_list(
    bad: dict[str, object],
) -> None:
    payload = issue(1) | bad
    client = LinearTaskClient(
        api_key="test",
        team_id="team",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=issue_page([payload]))
        ),
    )
    try:
        with pytest.raises(TaskTrackerError):
            await client.list_tasks(query=TaskQuery())
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["MAX_RETRIEVAL_ITEMS", "MAX_RETRIEVAL_PAGES"])
async def test_linear_limits_never_silently_truncate(
    monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    monkeypatch.setattr(linear_module, bound, 1)
    client = LinearTaskClient(
        api_key="test",
        team_id="team",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=issue_page([issue(1), issue(2)], "next"))
        ),
    )
    try:
        with pytest.raises(RetrievalLimitError):
            await client.list_tasks(query=TaskQuery())
    finally:
        await client.close()


def calendar_event(number: int, **extra: object) -> dict[str, Any]:
    return {
        "id": str(number),
        "summary": "План",
        "start": {"dateTime": "2026-09-06T12:00:00+03:00"},
        "end": {"dateTime": "2026-09-06T13:00:00+03:00"},
        "updated": "2026-09-01T12:00:00Z",
        **extra,
    }


def calendar_client(pages: list[object]) -> tuple[GoogleCalendarClient, Mock]:
    events = Mock()
    events.list.side_effect = [Mock(execute=Mock(return_value=page)) for page in pages]
    service = Mock(events=Mock(return_value=events))
    client = GoogleCalendarClient(storage=Mock())

    async def execute(*, user_id: int, operation: Any) -> Any:
        assert user_id == 42
        return operation(service)

    client._execute = AsyncMock(side_effect=execute)  # type: ignore[method-assign]
    return client, events.list


@pytest.mark.asyncio
async def test_calendar_search_scans_all_pages_and_literal_text_in_description_location() -> (
    None
):
    client, request = calendar_client(
        [
            {"items": [calendar_event(1)], "nextPageToken": "next"},
            {
                "items": [
                    calendar_event(2, description="План РЕМОНТА кухни"),
                    calendar_event(3, location="Магазин ремонта"),
                    {"status": "cancelled"},
                ]
            },
        ]
    )
    events = await client.list_events(
        user_id=42,
        now=START,
        query=EventQuery(query="ремонт", date_from=START, date_to=END),
    )
    assert [e.event_id for e in events] == ["2", "3"]
    assert events[0].updated_at == datetime(2026, 9, 1, 12, tzinfo=UTC)
    assert [r.kwargs.get("pageToken") for r in request.call_args_list] == [None, "next"]
    assert all(
        r.kwargs["timeMin"] == START.isoformat()
        and r.kwargs["timeMax"] == END.isoformat()
        and r.kwargs["singleEvents"] is True
        and r.kwargs["showDeleted"] is False
        for r in request.call_args_list
    )


@pytest.mark.asyncio
async def test_calendar_all_day_and_theme_filters() -> None:
    client, _ = calendar_client(
        [
            {
                "items": [
                    calendar_event(1, summary="Обои"),
                    calendar_event(
                        2,
                        summary="Электрик",
                        start={"date": "2026-09-06"},
                        end={"date": "2026-09-07"},
                    ),
                    calendar_event(3),
                ]
            }
        ]
    )
    events = await client.list_events(
        user_id=42,
        now=START,
        query=EventQuery(search_terms=["обои", "электрик"], all_day=True),
    )
    assert [e.event_id for e in events] == ["2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pages",
    [
        [{"items": [], "nextPageToken": "x"}, {"items": [], "nextPageToken": "x"}],
        [{"items": [], "nextPageToken": ""}],
        [{"items": {}}],
        [[]],
        [{"items": [None]}],
        [{"items": [calendar_event(1, updated="2026-01-01")]}],
    ],
)
async def test_calendar_malformed_pages_are_errors(pages: list[object]) -> None:
    client, _ = calendar_client(pages)
    with pytest.raises(CalendarError):
        await client.list_events(user_id=42, now=START, query=EventQuery())


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["MAX_RETRIEVAL_ITEMS", "MAX_RETRIEVAL_PAGES"])
async def test_calendar_scan_limits_do_not_claim_no_matches(
    monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    monkeypatch.setattr(google_module, bound, 1)
    client, _ = calendar_client(
        [{"items": [calendar_event(1), calendar_event(2)], "nextPageToken": "next"}]
    )
    with pytest.raises(RetrievalLimitError):
        await client.list_events(
            user_id=42, now=START, query=EventQuery(query="missing")
        )
