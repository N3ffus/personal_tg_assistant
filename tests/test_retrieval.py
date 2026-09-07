import re
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from html import unescape
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from src.application.services import retrieval as module
from src.application.services.retrieval import RetrievalService, linked
from src.domain.assistant.models import AssistantDecision
from src.domain.assistant.replies import ResultPage
from src.domain.assistant.retrieval import EventQuery, TaskQuery
from src.domain.calendar.models import CalendarEvent
from src.domain.tasks.models import Task

NOW = datetime(2026, 9, 6, 12, tzinfo=ZoneInfo("Europe/Moscow"))


def task(number: int, **changes: Any) -> Task:
    return replace(
        Task(
            f"APP-{number}",
            f"Задача {number}",
            f"https://linear.test/{number}",
            "Todo",
            created_at=NOW + timedelta(minutes=number),
        ),
        **changes,
    )


def event(number: int, **changes: Any) -> CalendarEvent:
    return replace(
        CalendarEvent(
            str(number),
            f"Событие {number}",
            NOW + timedelta(days=number),
            NOW + timedelta(days=number, hours=1),
            f"https://calendar.test/{number}",
        ),
        **changes,
    )


def dependencies(
    *, tasks: list[Task] | None = None, events: list[CalendarEvent] | None = None
) -> tuple[RetrievalService, SimpleNamespace]:
    client = SimpleNamespace(
        list_tasks=AsyncMock(return_value=tasks or []),
        list_events=AsyncMock(return_value=events or []),
        create_operation=AsyncMock(return_value="selection"),
        delete_event=AsyncMock(),
    )
    return RetrievalService(
        calendar=client, task_tracker=client, storage=client
    ), client


def callback(page: ResultPage, command: str, value: int | None = None) -> str:
    return next(
        b.callback_data
        for row in page.buttons
        for b in row
        if f":{command}:" in b.callback_data
        and (value is None or b.callback_data.endswith(f":{value}"))
    )


async def navigate(
    service: RetrievalService, page: ResultPage, command: str, value: int | None = None
) -> ResultPage:
    result = await service.navigate(
        user_id=42, data=callback(page, command, value), now=NOW
    )
    assert isinstance(result, ResultPage)
    return result


@pytest.mark.asyncio
async def test_pagination_has_ten_items_stable_back_next_and_no_extra_requests() -> (
    None
):
    service, client = dependencies(tasks=[task(i) for i in range(1, 24)])
    first = await service.open(query=TaskQuery(), user_id=42, now=NOW)
    assert first.text.count("<a href=") == 10
    assert "1–10 из 23" in first.text and "1/3" in first.text
    assert "APP-23:" in first.text and "APP-13:" not in first.text
    second = await navigate(service, first, "page", 1)
    assert "11–20 из 23" in second.text and "APP-13:" in second.text
    third = await navigate(service, second, "page", 2)
    assert third.text.count("<a href=") == 3 and "21–23 из 23" in third.text
    assert not any(":page:3" in b.callback_data for row in third.buttons for b in row)
    back = await navigate(service, second, "page", 0)
    assert back == first
    client.list_tasks.assert_awaited_once_with(query=TaskQuery())
    client.create_operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_keeps_full_filter_and_replaces_snapshot_only_after_success() -> (
    None
):
    service, client = dependencies(tasks=[task(1)])
    query = TaskQuery(
        query="отчёт",
        search_terms=["год", "месяц"],
        project="Аналитика",
        status_group="open",
        sort_by="due",
        direction="asc",
        page_size=5,
        date_from=NOW - timedelta(days=365),
        date_to=NOW,
    )
    first = await service.open(query=query, user_id=42, now=NOW)
    client.list_tasks.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError):
        await navigate(service, first, "refresh")
    assert await navigate(service, first, "page", 0) == first
    client.list_tasks.side_effect = None
    client.list_tasks.return_value = [task(2)]
    refreshed = await navigate(service, first, "refresh")
    assert "APP-2:" in refreshed.text and "APP-1:" not in refreshed.text
    assert client.list_tasks.call_args.kwargs["query"] == query
    assert await navigate(service, first, "page", 0) == first


@pytest.mark.asyncio
async def test_sort_direction_size_and_status_preserve_search() -> None:
    service, client = dependencies(tasks=[task(1, title="Z"), task(2, title="A")])
    page = await service.open(
        query=TaskQuery(query="текст", sort_by="title", direction="asc"),
        user_id=42,
        now=NOW,
    )
    assert page.text.index("APP-2") < page.text.index("APP-1")
    descending = await navigate(service, page, "direction")
    assert descending.text.index("APP-1") < descending.text.index("APP-2")
    resized = await navigate(service, descending, "size")
    assert any(b.text == "По 20" for row in resized.buttons for b in row)
    menu = await navigate(service, resized, "sort")
    sorted_page = await navigate(service, menu, "order", 6)
    assert "по статусу" in sorted_page.text
    client.list_tasks.assert_awaited_once()
    status = await navigate(service, sorted_page, "status")
    assert "Открытые" in status.text
    assert client.list_tasks.call_args.kwargs["query"].query == "текст"
    assert client.list_tasks.call_args.kwargs["query"].status_group == "open"


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["asc", "desc"])
@pytest.mark.parametrize(
    "sort_by", ["due", "priority", "created", "updated", "completed"]
)
async def test_missing_sort_values_always_last(sort_by: str, direction: str) -> None:
    service, _ = dependencies(
        tasks=[
            task(
                1,
                due_date=date(2026, 9, 7),
                priority=1,
                updated_at=NOW,
                completed_at=NOW,
            ),
            task(2, created_at=None),
        ]
    )
    query = TaskQuery.model_validate({"sort_by": sort_by, "direction": direction})
    page = await service.open(query=query, user_id=42, now=NOW)
    assert page.text.index("APP-1") < page.text.index("APP-2")


@pytest.mark.asyncio
async def test_calendar_year_all_day_sort_and_select_use_server_ids() -> None:
    whole_day = event(
        1,
        event_id="x" * 1024,
        title="A",
        starts_at=datetime(2026, 9, 6, tzinfo=UTC),
        ends_at=datetime(2026, 9, 9, tzinfo=UTC),
        all_day=True,
    )
    timed = event(2, title="B", starts_at=NOW, ends_at=NOW + timedelta(hours=1))
    service, client = dependencies(events=[timed, whole_day])
    first = await service.open(query=EventQuery(), user_id=42, now=NOW)
    assert first.text.index(">A</a>") < first.text.index(">B</a>")
    assert "06.09.2026–08.09.2026 · весь день" in first.text
    assert "06.09.2026 12:00 MSK" in first.text
    query = client.list_events.call_args.kwargs["query"]
    assert query.date_from == NOW and query.date_to == NOW + timedelta(days=365)
    assert all(
        len(b.callback_data.encode()) <= 64 for row in first.buttons for b in row
    )
    assert not any(":edit:0" in b.callback_data for row in first.buttons for b in row)
    reversed_page = await navigate(service, first, "direction")
    # An older message still selects the original item after sorting elsewhere.
    confirmation = await navigate(service, first, "delete", 0)
    assert confirmation.buttons[0][0].callback_data == "calyes:selection"
    client.create_operation.assert_awaited_once_with(
        user_id=42, kind="delete", payload={"event_id": "x" * 1024}
    )
    client.delete_event.assert_not_awaited()
    result = await service.navigate(
        user_id=42, data=callback(reversed_page, "edit", 0), now=NOW
    )
    assert isinstance(result, str) and "новое название" in result
    assert client.create_operation.call_args.kwargs["payload"] == {
        "event_id": timed.event_id
    }


@pytest.mark.asyncio
async def test_calendar_presets_keep_text_and_empty_results_offer_refresh() -> None:
    service, client = dependencies()
    first = await service.open(
        query=EventQuery(query="Python", all_day=False), user_id=42, now=NOW
    )
    assert "ничего не найдено" in first.text
    assert "0–0 из 0" in first.text
    week = await navigate(service, first, "days", 7)
    query = client.list_events.call_args.kwargs["query"]
    assert query.query == "Python" and query.all_day is False
    assert query.date_from == NOW.replace(hour=0)
    assert query.date_to == NOW.replace(hour=0) + timedelta(days=7)
    assert "Python" in week.text


@pytest.mark.asyncio
async def test_foreign_forged_expired_and_out_of_range_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Mock(return_value=10)
    monkeypatch.setattr(module, "monotonic", clock)
    service, client = dependencies(tasks=[task(1)])
    first = await service.open(query=TaskQuery(), user_id=42, now=NOW)
    data = callback(first, "refresh")
    assert isinstance(await service.navigate(user_id=99, data=data, now=NOW), str)
    for invalid in (
        "browse:nope:page:0",
        "broken",
        data.rsplit(":", 1)[0] + ":-1",
        data.rsplit(":", 1)[0] + ":１２",
    ):
        assert isinstance(
            await service.navigate(user_id=42, data=invalid, now=NOW), str
        )
    prefix = data.rsplit(":", 2)[0]
    for command in ("unknown:0", "delete:99", "edit:0", "days:7"):
        assert isinstance(
            await service.navigate(user_id=42, data=f"{prefix}:{command}", now=NOW), str
        )
    last = await service.navigate(user_id=42, data=f"{prefix}:page:99999", now=NOW)
    assert isinstance(last, ResultPage) and "1/1" in last.text
    clock.return_value = 10 + module.SESSION_TTL
    assert "устарел" in str(await service.navigate(user_id=42, data=data, now=NOW))
    client.list_tasks.assert_awaited_once()
    client.create_operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "MAX_SESSIONS", 2)
    service, _ = dependencies()
    pages = [
        await service.open(query=TaskQuery(), user_id=42, now=NOW) for _ in range(3)
    ]
    assert "устарел" in str(
        await service.navigate(user_id=42, data=callback(pages[0], "refresh"), now=NOW)
    )


@pytest.mark.asyncio
async def test_telegram_html_and_size_limits_with_twenty_long_emoji_rows() -> None:
    service, _ = dependencies(
        tasks=[
            task(i, title='<script>"&😀' * 200, status="😀" * 100) for i in range(20)
        ]
    )
    page = await service.open(
        query=TaskQuery(page_size=20, query="😀" * 160, search_terms=["😀" * 160] * 8),
        user_id=42,
        now=NOW,
    )
    assert "<script>" not in page.text
    visible = unescape(re.sub(r"<[^>]*>", "", page.text))
    assert len(visible.encode("utf-16-le")) // 2 <= 4096
    assert page.text.count("<a href=") == 20
    assert all(len(b.callback_data.encode()) <= 64 for row in page.buttons for b in row)


@pytest.mark.parametrize(
    "url",
    [
        None,
        "javascript:alert(1)",
        "file:///secret",
        "https://[bad",
        "https://x/" + "a" * 3000,
    ],
)
def test_unsafe_links_are_plain_text(url: str | None) -> None:
    assert linked("<hello>", url) == "&lt;hello&gt;"


@pytest.mark.parametrize(
    "payload",
    [
        {"page_size": 0},
        {"page_size": 21},
        {"query": "  "},
        {"priority": 5},
        {"date_from": "2026-01-01T00:00:00"},
        {"sort_by": "random"},
        {"date_from": "2026-01-01T00:00:00Z", "date_to": "2025-01-01T00:00:00Z"},
        {"search_terms": ["x"] * 9},
        {"unexpected": True},
    ],
)
def test_invalid_llm_filters_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AssistantDecision.model_validate(
            {"actions": [{"type": "list_tasks", **payload}]}
        )


def test_date_defaults_text_and_theme_semantics() -> None:
    end = datetime(2026, 1, 1, tzinfo=UTC)
    bounded = EventQuery(date_to=end).with_default_range(NOW)
    assert bounded.date_from == end - timedelta(days=365)
    assert bounded.date_to == end
    assert EventQuery(date_from=end).with_default_range(NOW).date_to == end + timedelta(
        days=365
    )
    query = TaskQuery(query="РЕМОНТ", search_terms=["обои", "электрик"])
    assert query.matches_text("Ремонт квартиры", "Нужен электрик")
    assert not query.matches_text("Ремонт машины")
    assert not query.matches_text("электрик")


@pytest.mark.asyncio
async def test_calendar_empty_titles_sort_with_named_events() -> None:
    service, _ = dependencies(events=[event(1, title="Planning"), event(2, title="")])
    page = await service.open(query=EventQuery(sort_by="title"), user_id=42, now=NOW)
    assert "Без названия" in page.text
    assert page.text.index("https://calendar.test/2") < page.text.index(
        "https://calendar.test/1"
    )
