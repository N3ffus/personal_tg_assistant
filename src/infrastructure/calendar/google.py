import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
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
        result = await self._execute(
            user_id=user_id,
            operation=lambda service: (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=now.astimezone(UTC).isoformat(),
                    maxResults=10,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            ),
        )
        if not isinstance(result, Mapping):
            raise CalendarError("Malformed Google Calendar response")
        items = result.get("items", [])
        if not isinstance(items, list):
            raise CalendarError("Malformed Google Calendar response")

        events: list[CalendarEvent] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise CalendarError("Malformed Google Calendar event")
            start = item.get("start")
            end = item.get("end")
            if not isinstance(start, Mapping) or not isinstance(end, Mapping):
                raise CalendarError("Malformed Google Calendar event")
            if "dateTime" not in start or "dateTime" not in end:
                continue
            events.append(self._to_event(item))
        return events

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
            if error.resp.status == 404:
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
    def _to_event(event: object) -> CalendarEvent:
        if not isinstance(event, Mapping):
            raise CalendarError("Malformed Google Calendar event")
        start_payload = event.get("start")
        end_payload = event.get("end")
        if not isinstance(start_payload, Mapping) or not isinstance(
            end_payload, Mapping
        ):
            raise CalendarError("Malformed Google Calendar event")
        start = start_payload.get("dateTime")
        end = end_payload.get("dateTime")
        if not isinstance(start, str) or not isinstance(end, str):
            raise CalendarError("All-day events are not supported")
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
            starts_at = datetime.fromisoformat(start)
            ends_at = datetime.fromisoformat(end)
        except ValueError as error:
            raise CalendarError("Malformed Google Calendar event time") from error
        return CalendarEvent(
            event_id=event_id,
            title=title,
            starts_at=starts_at,
            ends_at=ends_at,
            html_link=html_link,
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
