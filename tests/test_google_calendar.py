import json
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from urllib.request import Request as UrlRequest

import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from src.application.ports.calendar import (
    CalendarAuthorizationError,
    CalendarError,
    CalendarEventNotFoundError,
    CalendarNotConnectedError,
)
from src.domain.calendar.models import CalendarEvent
from src.infrastructure.calendar import google as google_module
from src.infrastructure.calendar.google import GoogleCalendarClient


class FakeStorage:
    def __init__(self, credentials: dict[str, object] | None) -> None:
        self.credentials = credentials
        self.saved: list[tuple[int, dict[str, object], str | None]] = []
        self.deleted: list[int] = []

    async def load_credentials(self, *, user_id: int) -> dict[str, object] | None:
        return self.credentials

    async def save_credentials(
        self,
        *,
        user_id: int,
        credentials: dict[str, object],
    ) -> None:
        self.saved.append((user_id, credentials, None))

    async def delete_connection(self, *, user_id: int) -> None:
        self.deleted.append(user_id)


class FakeCredentials:
    def __init__(
        self,
        *,
        valid: bool = True,
        expired: bool = False,
        refresh_token: str | None = "refresh-token",
    ) -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refresh_calls = 0

    def refresh(self, request: object) -> None:
        self.refresh_calls += 1
        self.valid = True
        self.expired = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "token": "refreshed-token",
                "refresh_token": self.refresh_token,
                "expiry": "2026-09-03T12:00:00Z",
            }
        )


class FakeRequest:
    def __init__(self, *, result: object | None = None) -> None:
        self._result = {} if result is None else result
        self.error: Exception | None = None

    def execute(self) -> object:
        if self.error is not None:
            raise self.error
        return self._result


class FakeEventsResource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.requests = {
            "insert": FakeRequest(result=google_event()),
            "list": FakeRequest(result={"items": [google_event()]}),
            "patch": FakeRequest(result=google_event(title="Updated")),
            "delete": FakeRequest(),
        }

    def _request(self, name: str, kwargs: dict[str, Any]) -> FakeRequest:
        self.calls.append((name, kwargs))
        return self.requests[name]

    def insert(self, **kwargs: Any) -> FakeRequest:
        return self._request("insert", kwargs)

    def list(self, **kwargs: Any) -> FakeRequest:
        return self._request("list", kwargs)

    def patch(self, **kwargs: Any) -> FakeRequest:
        return self._request("patch", kwargs)

    def delete(self, **kwargs: Any) -> FakeRequest:
        return self._request("delete", kwargs)


class FakeCalendarService:
    def __init__(self, events: FakeEventsResource) -> None:
        self._events = events

    def events(self) -> FakeEventsResource:
        return self._events


def google_event(
    *,
    event_id: str = "event-1",
    title: str | None = "Planning",
    starts_at: str = "2026-09-03T10:00:00+03:00",
    ends_at: str = "2026-09-03T11:00:00+03:00",
    html_link: str | None = "https://calendar.google.com/event-1",
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "id": event_id,
        "start": {"dateTime": starts_at},
        "end": {"dateTime": ends_at},
    }
    if title is not None:
        event["summary"] = title
    if html_link is not None:
        event["htmlLink"] = html_link
    return event


def connected_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    credentials: FakeCredentials | None = None,
    events: FakeEventsResource | None = None,
) -> tuple[GoogleCalendarClient, FakeStorage, FakeCredentials, FakeEventsResource]:
    storage = FakeStorage({"token": "stored-token"})
    fake_credentials = credentials or FakeCredentials()
    events_resource = events or FakeEventsResource()
    service = FakeCalendarService(events_resource)

    credentials_factory = Mock(return_value=fake_credentials)
    build = Mock(return_value=service)
    monkeypatch.setattr(
        google_module,
        "Credentials",
        SimpleNamespace(from_authorized_user_info=credentials_factory),
    )
    monkeypatch.setattr(google_module, "build", build)

    client = GoogleCalendarClient(storage=storage)  # type: ignore[arg-type]
    return client, storage, fake_credentials, events_resource


@pytest.mark.asyncio
async def test_create_event_uses_primary_calendar_and_one_hour_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, events = connected_client(monkeypatch)
    starts_at = datetime(2026, 9, 3, 10, 0, tzinfo=timezone(timedelta(hours=3)))

    event = await client.create_event(
        user_id=42,
        title="Planning",
        starts_at=starts_at,
    )

    assert event == CalendarEvent(
        event_id="event-1",
        title="Planning",
        starts_at=starts_at,
        ends_at=starts_at + timedelta(hours=1),
        html_link="https://calendar.google.com/event-1",
    )
    assert events.calls == [
        (
            "insert",
            {
                "calendarId": "primary",
                "body": {
                    "summary": "Planning",
                    "start": {"dateTime": starts_at.isoformat()},
                    "end": {"dateTime": (starts_at + timedelta(hours=1)).isoformat()},
                },
            },
        )
    ]


@pytest.mark.asyncio
async def test_list_events_requests_upcoming_ordered_timed_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, events = connected_client(monkeypatch)
    now = datetime(2026, 9, 3, 10, 0, tzinfo=timezone(timedelta(hours=3)))

    result = await client.list_events(user_id=42, now=now)

    assert [event.event_id for event in result] == ["event-1"]
    assert events.calls == [
        (
            "list",
            {
                "calendarId": "primary",
                "timeMin": now.astimezone(UTC).isoformat(),
                "maxResults": 10,
                "singleEvents": True,
                "orderBy": "startTime",
            },
        )
    ]


@pytest.mark.asyncio
async def test_update_event_patches_only_supported_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, events = connected_client(monkeypatch)
    starts_at = datetime(2026, 9, 4, 18, 30, tzinfo=UTC)

    event = await client.update_event(
        user_id=42,
        event_id="event-1",
        title="Updated",
        starts_at=starts_at,
    )

    assert event.title == "Updated"
    assert events.calls == [
        (
            "patch",
            {
                "calendarId": "primary",
                "eventId": "event-1",
                "body": {
                    "summary": "Updated",
                    "start": {"dateTime": starts_at.isoformat()},
                    "end": {"dateTime": (starts_at + timedelta(hours=1)).isoformat()},
                },
            },
        )
    ]


@pytest.mark.asyncio
async def test_delete_event_targets_primary_calendar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, events = connected_client(monkeypatch)

    await client.delete_event(user_id=42, event_id="event-1")

    assert events.calls == [("delete", {"calendarId": "primary", "eventId": "event-1"})]


@pytest.mark.asyncio
async def test_missing_credentials_is_reported_as_not_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage(None)
    client = GoogleCalendarClient(storage=storage)  # type: ignore[arg-type]

    with pytest.raises(CalendarNotConnectedError):
        await client.list_events(user_id=42, now=datetime.now(UTC))


@pytest.mark.asyncio
async def test_expired_credentials_are_refreshed_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = FakeCredentials(valid=False, expired=True)
    client, storage, _, _ = connected_client(
        monkeypatch,
        credentials=credentials,
    )

    await client.list_events(user_id=42, now=datetime(2026, 9, 3, tzinfo=UTC))

    assert credentials.refresh_calls == 1
    assert storage.saved == [
        (
            42,
            {
                "token": "refreshed-token",
                "refresh_token": "refresh-token",
                "expiry": "2026-09-03T12:00:00Z",
            },
            None,
        )
    ]


@pytest.mark.asyncio
async def test_invalid_non_refreshable_credentials_raise_authorization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = FakeCredentials(
        valid=False,
        expired=True,
        refresh_token=None,
    )
    client, _, _, events = connected_client(monkeypatch, credentials=credentials)

    with pytest.raises(CalendarAuthorizationError):
        await client.list_events(user_id=42, now=datetime.now(UTC))

    assert events.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_google_auth_http_errors_are_mapped_to_authorization_error(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    events = FakeEventsResource()
    events.requests["list"].error = HttpError(
        resp=SimpleNamespace(status=status, reason="denied"),
        content=b'{"error": {"message": "denied"}}',
    )
    client, _, _, _ = connected_client(monkeypatch, events=events)

    with pytest.raises(CalendarAuthorizationError):
        await client.list_events(user_id=42, now=datetime.now(UTC))


@pytest.mark.asyncio
async def test_missing_google_event_is_mapped_to_not_found_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = FakeEventsResource()
    source_error = HttpError(
        resp=SimpleNamespace(status=404, reason="not found"),
        content=b'{"error": {"message": "not found"}}',
    )
    events.requests["delete"].error = source_error
    client, _, _, _ = connected_client(monkeypatch, events=events)

    with pytest.raises(CalendarEventNotFoundError) as exc:
        await client.delete_event(user_id=42, event_id="missing")

    assert exc.value.__cause__ is source_error


@pytest.mark.asyncio
async def test_unexpected_google_client_errors_are_mapped_to_calendar_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = FakeEventsResource()
    source_error = RuntimeError("broken response")
    events.requests["insert"].error = source_error
    client, _, _, _ = connected_client(monkeypatch, events=events)

    with pytest.raises(CalendarError, match="Google Calendar request failed") as exc:
        await client.create_event(
            user_id=42,
            title="Planning",
            starts_at=datetime.now(UTC),
        )

    assert exc.value.__cause__ is source_error


@pytest.mark.asyncio
async def test_naive_event_time_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _ = connected_client(monkeypatch)

    with pytest.raises(CalendarError, match="timezone"):
        await client.create_event(
            user_id=42,
            title="Planning",
            starts_at=datetime(2026, 9, 3, 10, 0),
        )


@pytest.mark.asyncio
async def test_all_day_events_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = FakeEventsResource()
    events.requests["list"] = FakeRequest(
        result={
            "items": [
                {
                    "id": "all-day",
                    "summary": "Holiday",
                    "start": {"date": "2026-09-03"},
                    "end": {"date": "2026-09-04"},
                }
            ]
        }
    )
    client, _, _, _ = connected_client(monkeypatch, events=events)

    result = await client.list_events(user_id=42, now=datetime.now(UTC))

    assert result == []


@pytest.mark.asyncio
async def test_event_without_summary_uses_readable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = FakeEventsResource()
    events.requests["list"] = FakeRequest(result={"items": [google_event(title=None)]})
    client, _, _, _ = connected_client(monkeypatch, events=events)

    result = await client.list_events(user_id=42, now=datetime.now(UTC))

    assert result[0].title == "Без названия"


@pytest.mark.asyncio
async def test_list_events_skips_record_without_timed_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = FakeEventsResource()
    events.requests["list"] = FakeRequest(
        result={
            "items": [
                {
                    "id": "broken",
                    "start": {"dateTime": "2026-09-03T10:00:00+03:00"},
                    "end": {"date": "2026-09-04"},
                }
            ]
        }
    )
    client, _, _, _ = connected_client(monkeypatch, events=events)

    result = await client.list_events(user_id=42, now=datetime.now(UTC))

    assert result == []


@pytest.mark.asyncio
async def test_malformed_event_response_is_reported_as_calendar_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = FakeEventsResource()
    events.requests["insert"] = FakeRequest(
        result={
            "id": "all-day",
            "start": {"date": "2026-09-03"},
            "end": {"date": "2026-09-04"},
        }
    )
    client, _, _, _ = connected_client(monkeypatch, events=events)

    with pytest.raises(CalendarError, match="All-day events are not supported"):
        await client.create_event(
            user_id=42,
            title="Holiday",
            starts_at=datetime.now(UTC),
        )


@pytest.mark.asyncio
async def test_rate_limit_http_error_is_mapped_to_generic_calendar_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = FakeEventsResource()
    source_error = HttpError(
        resp=SimpleNamespace(status=429, reason="rate limited"),
        content=b'{"error": {"message": "rate limited"}}',
    )
    events.requests["list"].error = source_error
    client, _, _, _ = connected_client(monkeypatch, events=events)

    with pytest.raises(CalendarError, match="Google Calendar request failed") as exc:
        await client.list_events(user_id=42, now=datetime.now(UTC))

    assert exc.value.__cause__ is source_error


@pytest.mark.asyncio
async def test_refresh_failure_is_mapped_to_authorization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = FakeCredentials(valid=False, expired=True)

    def fail_refresh(request: object) -> None:
        raise RefreshError("refresh rejected")  # type: ignore[no-untyped-call]

    credentials.refresh = fail_refresh  # type: ignore[method-assign]
    client, _, _, _ = connected_client(monkeypatch, credentials=credentials)

    with pytest.raises(CalendarAuthorizationError) as exc:
        await client.list_events(user_id=42, now=datetime.now(UTC))

    assert isinstance(exc.value.__cause__, RefreshError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_error", [TypeError("bad token"), ValueError("bad token")]
)
async def test_malformed_stored_credentials_are_mapped_to_authorization_error(
    monkeypatch: pytest.MonkeyPatch,
    source_error: Exception,
) -> None:
    storage = FakeStorage({"token": "malformed"})
    monkeypatch.setattr(
        google_module,
        "Credentials",
        SimpleNamespace(
            from_authorized_user_info=Mock(side_effect=source_error),
        ),
    )
    client = GoogleCalendarClient(storage=storage)  # type: ignore[arg-type]

    with pytest.raises(CalendarAuthorizationError) as exc:
        await client.list_events(user_id=42, now=datetime.now(UTC))

    assert exc.value.__cause__ is source_error


@pytest.mark.asyncio
async def test_disconnect_revokes_refresh_token_then_deletes_local_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage({"refresh_token": "refresh-token"})
    revoked: list[str] = []
    monkeypatch.setattr(
        GoogleCalendarClient,
        "_revoke",
        staticmethod(revoked.append),
    )
    client = GoogleCalendarClient(storage=storage)  # type: ignore[arg-type]

    await client.disconnect(user_id=42)

    assert revoked == ["refresh-token"]
    assert storage.deleted == [42]


@pytest.mark.asyncio
async def test_disconnect_deletes_local_connection_without_refresh_token() -> None:
    storage = FakeStorage({"token": "access-token"})
    client = GoogleCalendarClient(storage=storage)  # type: ignore[arg-type]

    await client.disconnect(user_id=42)

    assert storage.deleted == [42]


@pytest.mark.asyncio
async def test_disconnect_still_deletes_local_connection_when_revoke_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage({"refresh_token": "refresh-token"})

    def fail_revoke(token: str) -> None:
        raise OSError("network unavailable")

    monkeypatch.setattr(
        GoogleCalendarClient,
        "_revoke",
        staticmethod(fail_revoke),
    )
    client = GoogleCalendarClient(storage=storage)  # type: ignore[arg-type]

    await client.disconnect(user_id=42)

    assert storage.deleted == [42]


@pytest.mark.asyncio
async def test_disconnect_is_idempotent_when_not_connected() -> None:
    storage = FakeStorage(None)
    client = GoogleCalendarClient(storage=storage)  # type: ignore[arg-type]

    await client.disconnect(user_id=42)

    assert storage.deleted == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"items": {}},
        {"items": [None]},
        {"items": [{"id": "broken", "start": "invalid", "end": {}}]},
    ],
)
async def test_malformed_list_payload_is_mapped_to_calendar_error(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    events = FakeEventsResource()
    events.requests["list"] = FakeRequest(result=payload)
    client, _, _, _ = connected_client(monkeypatch, events=events)

    with pytest.raises(CalendarError, match="Malformed Google Calendar"):
        await client.list_events(user_id=42, now=datetime.now(UTC))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {
            "start": {"dateTime": "2026-09-03T10:00:00+03:00"},
            "end": {"dateTime": "2026-09-03T11:00:00+03:00"},
        },
        google_event(starts_at="not-a-date"),
        {**google_event(), "summary": 123},
        {**google_event(), "htmlLink": 123},
    ],
)
async def test_malformed_created_event_is_mapped_to_calendar_error(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    events = FakeEventsResource()
    events.requests["insert"] = FakeRequest(result=payload)
    client, _, _, _ = connected_client(monkeypatch, events=events)

    with pytest.raises(CalendarError, match="Malformed Google Calendar"):
        await client.create_event(
            user_id=42,
            title="Planning",
            starts_at=datetime.now(UTC),
        )


def test_revoke_posts_token_in_body_and_closes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[UrlRequest, float]] = []

    class FakeResponse:
        entered = False
        exited = False
        read_called = False

        def __enter__(self) -> "FakeResponse":
            self.entered = True
            return self

        def __exit__(self, *args: object) -> None:
            self.exited = True

        def read(self) -> bytes:
            self.read_called = True
            return b""

    response = FakeResponse()

    def fake_urlopen(request: UrlRequest, timeout: float) -> FakeResponse:
        captured.append((request, timeout))
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    GoogleCalendarClient._revoke("refresh token")

    request, timeout = captured[0]
    assert request.full_url == "https://oauth2.googleapis.com/revoke"
    assert request.data == b"token=refresh+token"
    assert request.method == "POST"
    assert timeout == 10
    assert response.entered is True
    assert response.read_called is True
    assert response.exited is True
