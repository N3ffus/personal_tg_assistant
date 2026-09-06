import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.application.ports.calendar import (
    CalendarAuthorizationError,
    CalendarError,
    CalendarEventNotFoundError,
    CalendarNotConnectedError,
)
from src.domain.assistant.deletions import DeletionTarget
from src.domain.calendar.models import CalendarEvent
from src.infrastructure.calendar.storage import CalendarStorage

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
logger = logging.getLogger(__name__)


class GoogleCalendarClient:
    def __init__(self, *, storage: CalendarStorage) -> None:
        self._storage = storage

    async def create_event(
        self, *, user_id: int, title: str, starts_at: datetime
    ) -> CalendarEvent:
        body = self._event_body(title=title, starts_at=starts_at)
        result = await self._execute(
            user_id=user_id,
            operation=lambda service: (
                service.events()
                .insert(
                    calendarId="primary",
                    body=body,
                )
                .execute()
            ),
        )
        return self._to_event(result)

    async def list_events(self, *, user_id: int, now: datetime) -> list[CalendarEvent]:
        events: list[CalendarEvent] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, object] = {
                "calendarId": "primary",
                "timeMin": now.astimezone(UTC).isoformat(),
                "maxResults": 10 - len(events),
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if page_token is not None:
                params["pageToken"] = page_token

            def list_page(service: Any, params: dict[str, object] = params) -> Any:
                return service.events().list(**params).execute()

            result = await self._execute(
                user_id=user_id,
                operation=list_page,
            )
            if not isinstance(result, Mapping):
                raise CalendarError("Malformed Google Calendar response")
            items = result.get("items", [])
            if not isinstance(items, list):
                raise CalendarError("Malformed Google Calendar response")
            for item in items:
                if not isinstance(item, Mapping):
                    raise CalendarError("Malformed Google Calendar event")
                if item.get("status") == "cancelled":
                    continue
                events.append(self._to_event(item, allow_all_day=True))
                if len(events) == 10:
                    return events
            page_token = result.get("nextPageToken")
            if page_token is None:
                return events
            if (
                not isinstance(page_token, str)
                or not page_token
                or page_token in seen_tokens
            ):
                raise CalendarError("Malformed Google Calendar pagination")
            seen_tokens.add(page_token)

    async def update_event(
        self, *, user_id: int, event_id: str, title: str, starts_at: datetime
    ) -> CalendarEvent:
        body = self._event_body(title=title, starts_at=starts_at)
        result = await self._execute(
            user_id=user_id,
            operation=lambda service: (
                service.events()
                .patch(
                    calendarId="primary",
                    eventId=event_id,
                    body=body,
                )
                .execute()
            ),
        )
        return self._to_event(result)

    async def find_events(
        self, *, user_id: int, title: str | None
    ) -> list[DeletionTarget]:
        if title is not None and not title.strip():
            raise CalendarError("Event search must not be blank")

        def find(service: Any) -> list[DeletionTarget]:
            targets: dict[str, DeletionTarget] = {}
            page_token: str | None = None
            seen_tokens: set[str] = set()
            while True:
                # Do not expand recurring series: they can have infinitely many
                # instances. A series is one explicit target in the preview.
                result = (
                    service.events()
                    .list(
                        calendarId="primary",
                        maxResults=2500,
                        singleEvents=False,
                        showDeleted=False,
                        pageToken=page_token,
                    )
                    .execute()
                )
                if not isinstance(result, Mapping) or not isinstance(
                    result.get("items", []), list
                ):
                    raise CalendarError("Malformed Google Calendar response")
                for event in result.get("items", []):
                    if not isinstance(event, Mapping):
                        raise CalendarError("Malformed Google Calendar event")
                    if event.get("status") == "cancelled":
                        continue
                    target = self._deletion_target(event)
                    if (
                        title is None
                        or title.strip().casefold() in target.title.casefold()
                    ):
                        targets[target.id] = target
                page_token = result.get("nextPageToken")
                if page_token is None:
                    return list(targets.values())
                if (
                    not isinstance(page_token, str)
                    or not page_token
                    or page_token in seen_tokens
                ):
                    raise CalendarError("Malformed Google Calendar pagination")
                seen_tokens.add(page_token)

        return await self._execute(user_id=user_id, operation=find)  # type: ignore[no-any-return]

    @staticmethod
    def _deletion_target(event: Mapping[str, Any]) -> DeletionTarget:
        event_id = event.get("id")
        title = event.get("summary", "Без названия")
        start = event.get("start")
        if (
            not isinstance(event_id, str)
            or not event_id
            or not isinstance(title, str)
            or not isinstance(start, Mapping)
        ):
            raise CalendarError("Malformed Google Calendar event")
        starts_at = start.get("dateTime", start.get("date"))
        if not isinstance(starts_at, str):
            raise CalendarError("Malformed Google Calendar event time")
        try:
            date = datetime.fromisoformat(starts_at)
        except ValueError as error:
            raise CalendarError("Malformed Google Calendar event time") from error
        when = (
            date.strftime("%d.%m.%Y %H:%M %z")
            if "dateTime" in start
            else date.strftime("%d.%m.%Y (весь день)")
        )
        recurrence = " (вся повторяющаяся серия)" if event.get("recurrence") else ""
        return DeletionTarget(
            id=event_id, title=title, label=f"{when} — {title}{recurrence}"
        )

    async def delete_event(self, *, user_id: int, event_id: str) -> None:
        await self._execute(
            user_id=user_id,
            operation=lambda service: (
                service.events()
                .delete(calendarId="primary", eventId=event_id)
                .execute()
            ),
        )

    async def disconnect(self, *, user_id: int) -> None:
        credentials_data = await self._storage.load_credentials(user_id=user_id)
        if credentials_data is None:
            return
        refresh_token = credentials_data.get("refresh_token")
        if isinstance(refresh_token, str):
            try:
                await asyncio.to_thread(self._revoke, refresh_token)
            except OSError:
                logger.warning("Failed to revoke Google token for user %s", user_id)
        await self._storage.delete_connection(user_id=user_id)

    async def _execute(
        self,
        *,
        user_id: int,
        operation: Callable[[Any], Any],
    ) -> Any:
        credentials_data = await self._storage.load_credentials(user_id=user_id)
        if credentials_data is None:
            raise CalendarNotConnectedError
        try:
            credentials = Credentials.from_authorized_user_info(  # type: ignore[no-untyped-call]
                credentials_data,
                SCOPES,
            )
            if (
                not credentials.valid
                and credentials.expired
                and credentials.refresh_token
            ):
                await asyncio.to_thread(credentials.refresh, Request())
                await self._storage.save_credentials(
                    user_id=user_id,
                    credentials=json.loads(credentials.to_json()),
                )
            if not credentials.valid:
                raise CalendarAuthorizationError
            service = await asyncio.to_thread(
                build, "calendar", "v3", credentials=credentials, cache_discovery=False
            )
            return await asyncio.to_thread(operation, service)
        except HttpError as error:
            if error.resp.status in {401, 403}:
                raise CalendarAuthorizationError from error
            if error.resp.status in {404, 410}:
                raise CalendarEventNotFoundError from error
            raise CalendarError("Google Calendar request failed") from error
        except (CalendarAuthorizationError, CalendarEventNotFoundError):
            raise
        except (RefreshError, TypeError, ValueError) as error:
            raise CalendarAuthorizationError from error
        except Exception as error:
            raise CalendarError("Google Calendar request failed") from error

    @staticmethod
    def _event_body(*, title: str, starts_at: datetime) -> dict[str, object]:
        if starts_at.tzinfo is None:
            raise CalendarError("Event start time must contain a timezone")
        ends_at = starts_at + timedelta(hours=1)
        return {
            "summary": title,
            "start": {"dateTime": starts_at.isoformat()},
            "end": {"dateTime": ends_at.isoformat()},
        }

    @staticmethod
    def _to_event(event: object, *, allow_all_day: bool = False) -> CalendarEvent:
        if not isinstance(event, Mapping):
            raise CalendarError("Malformed Google Calendar event")
        start_payload = event.get("start")
        end_payload = event.get("end")
        if not isinstance(start_payload, Mapping) or not isinstance(
            end_payload, Mapping
        ):
            raise CalendarError("Malformed Google Calendar event")
        all_day = "dateTime" not in start_payload and "dateTime" not in end_payload
        if all_day and not allow_all_day:
            raise CalendarError("All-day events are not supported")
        key = "date" if all_day else "dateTime"
        start = start_payload.get(key)
        end = end_payload.get(key)
        if not isinstance(start, str) or not isinstance(end, str):
            raise CalendarError("Malformed Google Calendar event time")
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise CalendarError("Malformed Google Calendar event")
        title = event.get("summary")
        if title is None:
            title = "Без названия"
        if not isinstance(title, str):
            raise CalendarError("Malformed Google Calendar event")
        html_link = event.get("htmlLink")
        if html_link is not None and not isinstance(html_link, str):
            raise CalendarError("Malformed Google Calendar event")
        try:
            if all_day:
                # Preserve calendar dates regardless of the display timezone.
                # Google uses an exclusive end date for all-day events.
                starts_at = datetime.combine(date.fromisoformat(start), time(), UTC)
                ends_at = datetime.combine(date.fromisoformat(end), time(), UTC)
            else:
                starts_at = datetime.fromisoformat(start)
                ends_at = datetime.fromisoformat(end)
                if starts_at.tzinfo is None or ends_at.tzinfo is None:
                    raise ValueError("Missing event timezone")
            if ends_at <= starts_at:
                raise ValueError("Event end must follow start")
        except ValueError as error:
            raise CalendarError("Malformed Google Calendar event time") from error
        return CalendarEvent(
            event_id=event_id,
            title=title,
            starts_at=starts_at,
            ends_at=ends_at,
            html_link=html_link,
            all_day=all_day,
        )

    @staticmethod
    def _revoke(refresh_token: str) -> None:
        from urllib.parse import urlencode
        from urllib.request import Request as UrlRequest
        from urllib.request import urlopen

        request = UrlRequest(
            "https://oauth2.googleapis.com/revoke",
            data=urlencode({"token": refresh_token}).encode(),
            method="POST",
        )
        # The destination is a fixed HTTPS Google endpoint, never user-controlled.
        with urlopen(request, timeout=10) as response:  # nosec B310
            response.read()
