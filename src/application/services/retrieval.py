import asyncio
import secrets
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from html import escape
from time import monotonic
from urllib.parse import urlsplit

from src.application.ports.calendar import CalendarClient, PendingOperationStore
from src.application.ports.tasks import TaskTrackerClient
from src.domain.assistant.replies import ReplyButton, ResultPage
from src.domain.assistant.retrieval import EventQuery, TaskQuery
from src.domain.calendar.models import CalendarEvent
from src.domain.tasks.models import Task

SESSION_TTL = 3600
MAX_SESSIONS = 16
SORT_LABELS = {
    "created": "созданию",
    "updated": "обновлению",
    "due": "сроку",
    "completed": "завершению",
    "priority": "приоритету",
    "title": "названию",
    "status": "статусу",
    "start": "дате события",
}
STATUS_LABELS = {
    "all": "Все",
    "open": "Открытые",
    "completed": "Выполненные",
    "canceled": "Отменённые",
}
PRIORITIES = {0: "Без приоритета", 1: "Срочно", 2: "Высокий", 3: "Обычный", 4: "Низкий"}


def clip(text: str, limit: int) -> str:
    """Bound visible Telegram text in UTF-16 units, without splitting emoji."""
    text = " ".join(text.split())
    encoded = text.encode("utf-16-le")
    return (
        text
        if len(encoded) <= limit * 2
        else encoded[: (limit - 1) * 2].decode("utf-16-le", errors="ignore") + "…"
    )


def linked(text: str, url: str | None) -> str:
    label = escape(text)
    if url and len(url) <= 2048:
        try:
            parsed = urlsplit(url)
            if parsed.scheme in {"https", "http"} and parsed.netloc:
                return f'<a href="{escape(url, quote=True)}">{label}</a>'
        except ValueError:
            pass
    return label


def event_when(event: CalendarEvent, now: datetime) -> str:
    if event.all_day:
        end = event.ends_at - timedelta(days=1)
        when = f"{event.starts_at:%d.%m.%Y}"
        if end.date() > event.starts_at.date():
            when += f"–{end:%d.%m.%Y}"
        return when + " · весь день"
    return f"{event.starts_at.astimezone(now.tzinfo):%d.%m.%Y %H:%M %Z}"


@dataclass
class BrowseSession:
    user_id: int
    query: TaskQuery | EventQuery
    items: Sequence[Task | CalendarEvent]
    now: datetime
    expires_at: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RetrievalService:
    def __init__(
        self,
        *,
        calendar: CalendarClient,
        task_tracker: TaskTrackerClient,
        storage: PendingOperationStore,
    ) -> None:
        self._calendar = calendar
        self._tasks = task_tracker
        self._storage = storage
        self._sessions: OrderedDict[str, BrowseSession] = OrderedDict()

    async def open(
        self, *, query: TaskQuery | EventQuery, user_id: int, now: datetime
    ) -> ResultPage:
        if isinstance(query, EventQuery):
            query = query.with_default_range(now)
        items = await self._fetch(query, user_id, now)
        session = BrowseSession(user_id, query, items, now, monotonic() + SESSION_TTL)
        self._sort(session)
        return self._remember(session)

    def _remember(self, session: BrowseSession) -> ResultPage:
        self._purge()
        while len(self._sessions) >= MAX_SESSIONS:
            self._sessions.popitem(last=False)
        token = secrets.token_urlsafe(12)
        self._sessions[token] = session
        return self._render(token, session, 0)

    async def _fetch(
        self, query: TaskQuery | EventQuery, user_id: int, now: datetime
    ) -> Sequence[Task | CalendarEvent]:
        # Also cap total wall time across provider pages, not just each HTTP request.
        async with asyncio.timeout(60):
            if isinstance(query, TaskQuery):
                return await self._tasks.list_tasks(query=query)
            return await self._calendar.list_events(
                user_id=user_id, now=now, query=query
            )

    def _purge(self) -> None:
        for token in list(self._sessions):
            if self._sessions[token].expires_at <= monotonic():
                del self._sessions[token]

    async def navigate(
        self, *, user_id: int, data: str, now: datetime
    ) -> ResultPage | str:
        self._purge()
        parts = data.split(":")
        if len(parts) != 4 or parts[0] != "browse":
            return "Кнопка недействительна."
        _, token, command, value = parts
        session = self._sessions.get(token)
        if session is None or session.user_id != user_id:
            return "Список устарел. Повторите запрос или откройте /tasks, /calendar, /agenda."
        if not value.isascii() or not value.isdigit() or len(value) > 6:
            return "Кнопка недействительна."
        position = int(value)
        async with session.lock:
            if command in {"edit", "delete"}:
                return await self._select(session, command, position)
            if command == "page":
                return self._render(token, session, position)
            query = session.query
            if command in {"sort", "order"}:
                keys = (
                    list(SORT_LABELS)
                    if isinstance(query, TaskQuery)
                    else ["start", "title", "updated"]
                )
                if isinstance(query, TaskQuery):
                    keys.remove("start")
                if command == "sort":
                    options = tuple(
                        ReplyButton(
                            "По " + SORT_LABELS[key], f"browse:{token}:order:{index}"
                        )
                        for index, key in enumerate(keys)
                    )
                    rows = tuple(options[i : i + 2] for i in range(0, len(options), 2))
                    return replace(
                        self._render(token, session, position),
                        buttons=rows
                        + (
                            (
                                ReplyButton(
                                    "← К списку", f"browse:{token}:page:{position}"
                                ),
                            ),
                        ),
                    )
                if position >= len(keys):
                    return "Кнопка недействительна."
                query = query.model_copy(update={"sort_by": keys[position]})
            elif command == "direction":
                query = query.model_copy(
                    update={"direction": "desc" if query.direction == "asc" else "asc"}
                )
            elif command == "size":
                size = {5: 10, 10: 20, 20: 5}.get(query.page_size, 10)
                query = query.model_copy(update={"page_size": size})
            elif command == "status" and isinstance(query, TaskQuery):
                groups = list(STATUS_LABELS)
                query = query.model_copy(
                    update={
                        "status": None,
                        "status_group": groups[
                            (groups.index(query.status_group) + 1) % len(groups)
                        ],
                    }
                )
            elif (
                command == "days"
                and isinstance(query, EventQuery)
                and position in {1, 7, 30, 365}
            ):
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.model_copy(
                    update={
                        "date_from": start,
                        "date_to": start + timedelta(days=position),
                    }
                )
            elif command != "refresh":
                return "Кнопка недействительна."
            items = session.items
            if command in {"refresh", "status", "days"}:
                items = await self._fetch(query, user_id, now)
            # Failed refreshes leave the previous query and snapshot intact.
            updated = replace(
                session,
                query=query,
                items=list(items),
                lock=asyncio.Lock(),
                now=now if command in {"refresh", "status", "days"} else session.now,
            )
            self._sort(updated)
            return self._remember(updated)

    async def _select(
        self, session: BrowseSession, command: str, position: int
    ) -> ResultPage | str:
        if position >= len(session.items):
            return "Кнопка недействительна."
        item = session.items[position]
        if not isinstance(item, CalendarEvent) or (command == "edit" and item.all_day):
            return "Это действие недоступно."
        operation_id = await self._storage.create_operation(
            user_id=session.user_id,
            kind="update" if command == "edit" else "delete",
            payload={"event_id": item.event_id},
        )
        if command == "edit":
            return f"Выбрано событие «{clip(item.title, 160)}». Напишите новое название и время события."
        return ResultPage(
            text=f"Удалить событие «{escape(clip(item.title, 160))}» ({event_when(item, session.now)})?",
            buttons=(
                (
                    ReplyButton("Да, удалить", f"calyes:{operation_id}"),
                    ReplyButton("Отмена", f"calno:{operation_id}"),
                ),
            ),
        )

    @staticmethod
    def _sort(session: BrowseSession) -> None:
        key = session.query.sort_by

        def value(item: Task | CalendarEvent) -> str | float | None:
            if key == "title":
                return item.title.casefold()
            if isinstance(item, CalendarEvent):
                dt = item.updated_at if key == "updated" else item.starts_at
                if key == "start" and item.all_day:
                    dt = item.starts_at.replace(tzinfo=session.now.tzinfo)
                return dt.timestamp() if dt else None
            if key == "status":
                return item.status.casefold()
            if key == "priority":
                return float(item.priority) if item.priority else None
            if key == "due":
                return float(item.due_date.toordinal()) if item.due_date else None
            dt = {
                "created": item.created_at,
                "updated": item.updated_at,
                "completed": item.completed_at,
            }[key]
            return dt.timestamp() if dt else None

        # Stable IDs break ties; missing dates/priorities always come last.
        ordered = sorted(
            session.items,
            key=lambda item: (
                item.identifier if isinstance(item, Task) else item.event_id
            ),
        )
        known = [item for item in ordered if value(item) is not None]
        missing = [item for item in ordered if value(item) is None]

        def sort_value(item: Task | CalendarEvent) -> str | float:
            result = value(item)
            assert result is not None
            return result

        known.sort(key=sort_value, reverse=session.query.direction == "desc")
        session.items = tuple(known + missing)

    @staticmethod
    def _description(query: TaskQuery | EventQuery) -> str:
        filters = []
        if query.query:
            filters.append(f"Текст: «{query.query}»")
        if query.search_terms:
            filters.append("Тема (любое слово): " + ", ".join(query.search_terms))
        if query.date_from or query.date_to:
            start = (
                query.date_from.strftime("%d.%m.%Y %H:%M %z")
                if query.date_from
                else "начала"
            )
            end = query.date_to.strftime("%d.%m.%Y %H:%M %z") if query.date_to else "∞"
            field_name = (
                f"по {SORT_LABELS[query.date_field]}"
                if isinstance(query, TaskQuery)
                else "по времени события"
            )
            filters.append(f"Период {field_name}: {start} — {end} (конец не включён)")
        elif isinstance(query, TaskQuery):
            filters.append("Период: всё время")
        if isinstance(query, TaskQuery):
            filters.append(
                f"Статус: {query.status or STATUS_LABELS[query.status_group]} · {'включая архив' if query.include_archived else 'без архивных'}"
            )
            for label, text in (
                ("Проект", query.project),
                ("Исполнитель", query.assignee),
                ("Метка", query.label),
            ):
                if text:
                    filters.append(f"{label}: {text}")
            if query.priority is not None:
                filters.append(f"Приоритет: {PRIORITIES[query.priority]}")
        elif query.all_day is not None:
            filters.append(
                "Только на весь день" if query.all_day else "Только со временем"
            )
        return escape(clip(" · ".join(filters), 650))

    def _render(self, token: str, session: BrowseSession, page: int) -> ResultPage:
        query = session.query
        count = len(session.items)
        pages = max(1, (count + query.page_size - 1) // query.page_size)
        page = min(max(0, page), pages - 1)
        start = page * query.page_size
        selected = session.items[start : start + query.page_size]
        title = (
            "📋 Задачи Linear"
            if isinstance(query, TaskQuery)
            else "📅 События Google Calendar"
        )
        lines = [f"<b>{title}</b> · найдено {count}", self._description(query)]
        direction = "↑" if query.direction == "asc" else "↓"
        lines.append(f"Сортировка: по {SORT_LABELS[query.sort_by]} {direction}")
        if not selected:
            lines.append(
                "По этим фильтрам ничего не найдено. Измените фильтры или период."
            )
        for number, item in enumerate(selected, start + 1):
            url: str | None
            if isinstance(item, Task):
                label = clip(f"{item.identifier}: {item.title}", 85)
                details = item.status
                if item.due_date:
                    details += f" · до {item.due_date:%d.%m.%Y}"
                if item.priority:
                    details += " · " + PRIORITIES[item.priority]
                url = item.url
            else:
                label = clip(item.title, 85) or "Без названия"
                details = event_when(item, session.now)
                url = item.html_link
            lines.append(
                f"{number}. {linked(label, url)}\n   {escape(clip(details, 50))}"
            )
        lines.append(
            f"Страница {page + 1}/{pages} · {start + 1 if count else 0}–{start + len(selected)} из {count}"
        )
        lines.append(
            f"Данные на {session.now:%d.%m %H:%M}. Фильтры можно задать сообщением."
        )

        def button(label: str, command: str, value: int = 0) -> ReplyButton:
            return ReplyButton(label, f"browse:{token}:{command}:{value}")

        rows = []
        navigation = []
        if page:
            navigation.append(button("« Назад", "page", page - 1))
        navigation.append(button(f"{page + 1}/{pages}", "page", page))
        if page + 1 < pages:
            navigation.append(button("Далее »", "page", page + 1))
        rows.append(tuple(navigation))
        if pages > 3:
            rows.append(
                (
                    button("« Первая", "page", 0),
                    button("Последняя »", "page", pages - 1),
                )
            )
        rows.append(
            (button("↻ Обновить", "refresh"), button(f"По {query.page_size}", "size"))
        )
        rows.append(
            (
                button(f"По {SORT_LABELS[query.sort_by]} ▾", "sort", page),
                button(direction + " Порядок", "direction"),
            )
        )
        if isinstance(query, TaskQuery):
            rows.append(
                (
                    button(
                        "Статус: "
                        + (
                            clip(query.status, 25)
                            if query.status
                            else STATUS_LABELS[query.status_group]
                        ),
                        "status",
                    ),
                )
            )
        else:
            rows.append(
                tuple(
                    button(label, "days", days)
                    for label, days in (
                        ("Сегодня", 1),
                        ("7 дней", 7),
                        ("30 дней", 30),
                        ("Год", 365),
                    )
                )
            )
            for position, item in enumerate(selected, start):
                if isinstance(item, CalendarEvent):
                    actions = []
                    if not item.all_day:
                        actions.append(button(f"✏️ {position + 1}", "edit", position))
                    actions.append(button(f"🗑 {position + 1}", "delete", position))
                    rows.append(tuple(actions))
        return ResultPage("\n\n".join(lines), tuple(rows))
