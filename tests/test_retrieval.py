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
    )
    return RetrievalService(calendar=client, task_tracker=client), client


def callback(page: ResultPage, value: int) -> str:
    return next(
        b.callback_data
        for row in page.buttons
        for b in row
        if b.callback_data.endswith(f":page:{value}")
    )


async def navigate(
    service: RetrievalService, page: ResultPage, value: int
) -> ResultPage:
    result = await service.navigate(user_id=42, data=callback(page, value))
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
    second = await navigate(service, first, 1)
    assert "11–20 из 23" in second.text and "APP-13:" in second.text
    third = await navigate(service, second, 2)
    assert third.text.count("<a href=") == 3 and "21–23 из 23" in third.text
    assert not any(":page:3" in b.callback_data for row in third.buttons for b in row)
    back = await navigate(service, second, 0)
    assert back == first
    client.list_tasks.assert_awaited_once_with(query=TaskQuery())


@pytest.mark.asyncio
async def test_keyboard_offers_pagination_only_and_never_refetches() -> None:
    service, client = dependencies(tasks=[task(i) for i in range(1, 45)])
    query = TaskQuery(
        query="отчёт",
        search_terms=["год", "месяц"],
        project="Аналитика",
        status_group="open",
        sort_by="due",
        direction="asc",
        page_size=10,
        date_from=NOW - timedelta(days=365),
        date_to=NOW,
    )
    first = await service.open(query=query, user_id=42, now=NOW)
    assert [
        b.callback_data.rsplit(":", 2)[1] for row in first.buttons for b in row
    ] == ["page"] * 4
    assert [b.text for row in first.buttons for b in row] == [
        "1/5",
        "Далее »",
        "« Первая",
        "Последняя »",
    ]
    # Sorting, filters, refresh, editing and deletion are the LLM's job now.
    last = await navigate(service, first, 4)
    assert [b.text for row in last.buttons for b in row] == [
        "« Назад",
        "5/5",
        "« Первая",
        "Последняя »",
    ]
    assert "задайте сообщением" in first.text
    client.list_tasks.assert_awaited_once_with(query=query)


@pytest.mark.asyncio
async def test_single_page_result_carries_no_keyboard() -> None:
    service, _ = dependencies(tasks=[task(1)])
    page = await service.open(query=TaskQuery(), user_id=42, now=NOW)
    assert page.buttons == ()


@pytest.mark.asyncio
async def test_sorting_requested_by_the_model_is_applied_to_the_whole_result() -> None:
    service, _ = dependencies(tasks=[task(1, title="Z"), task(2, title="A")])
    ascending = await service.open(
        query=TaskQuery(sort_by="title", direction="asc"), user_id=42, now=NOW
    )
    assert ascending.text.index("APP-2") < ascending.text.index("APP-1")
    assert "Сортировка: по названию ↑" in ascending.text
    descending = await service.open(
        query=TaskQuery(sort_by="title", direction="desc"), user_id=42, now=NOW
    )
    assert descending.text.index("APP-1") < descending.text.index("APP-2")
    assert "Сортировка: по названию ↓" in descending.text


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
async def test_calendar_year_and_all_day_events_render_without_action_buttons() -> None:
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
    # A 1024-byte event id never has to fit into 64 bytes of callback data.
    assert first.buttons == ()
    for command in ("edit", "delete"):
        stale = await service.navigate(user_id=42, data=f"browse:token:{command}:0")
        assert isinstance(stale, str)


@pytest.mark.asyncio
async def test_empty_results_keep_the_filters_and_drop_the_keyboard() -> None:
    service, client = dependencies()
    page = await service.open(
        query=EventQuery(query="Python", all_day=False), user_id=42, now=NOW
    )
    assert "ничего не найдено" in page.text
    assert "0–0 из 0" in page.text
    assert "Python" in page.text and page.buttons == ()
    query = client.list_events.call_args.kwargs["query"]
    assert query.query == "Python" and query.all_day is False


@pytest.mark.asyncio
async def test_foreign_forged_expired_and_out_of_range_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Mock(return_value=10)
    monkeypatch.setattr(module, "monotonic", clock)
    service, client = dependencies(tasks=[task(1), task(2)])
    first = await service.open(query=TaskQuery(page_size=1), user_id=42, now=NOW)
    data = callback(first, 1)
    assert isinstance(await service.navigate(user_id=99, data=data), str)
    for invalid in (
        "browse:nope:page:0",
        "broken",
        data.rsplit(":", 1)[0] + ":-1",
        data.rsplit(":", 1)[0] + ":１２",
    ):
        assert isinstance(await service.navigate(user_id=42, data=invalid), str)
    prefix = data.rsplit(":", 2)[0]
    # Every command the keyboard used to send is now rejected outright.
    for command in ("unknown:0", "refresh:0", "sort:0", "delete:1", "edit:0", "days:7"):
        assert isinstance(
            await service.navigate(user_id=42, data=f"{prefix}:{command}"), str
        )
    last = await service.navigate(user_id=42, data=f"{prefix}:page:99999")
    assert isinstance(last, ResultPage) and "2/2" in last.text
    clock.return_value = 10 + module.SESSION_TTL
    assert "устарел" in str(await service.navigate(user_id=42, data=data))
    client.list_tasks.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "MAX_SESSIONS", 2)
    service, _ = dependencies(tasks=[task(1), task(2)])
    pages = [
        await service.open(query=TaskQuery(page_size=1), user_id=42, now=NOW)
        for _ in range(3)
    ]
    assert "устарел" in str(
        await service.navigate(user_id=42, data=callback(pages[0], 1))
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
