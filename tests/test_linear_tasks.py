import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from src.application.ports.tasks import TaskCreationUncertainError, TaskTrackerError
from src.domain.tasks.models import CreatedTask
from src.infrastructure.tasks.linear import (
    CREATE_ISSUE_MUTATION,
    LINEAR_GRAPHQL_URL,
    LinearTaskClient,
)

ResponseHandler = Callable[[httpx.Request], httpx.Response]


def successful_payload(*, issue: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "data": {
            "issueCreate": {
                "success": True,
                "issue": issue
                or {
                    "identifier": "APP-42",
                    "title": "Returned Linear title",
                    "url": "https://linear.app/example/issue/APP-42",
                },
            }
        }
    }


def json_handler(
    payload: Any,
    *,
    status_code: int = 200,
) -> ResponseHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        if payload is None:
            return httpx.Response(
                status_code,
                content=b"null",
                headers={"Content-Type": "application/json"},
                request=request,
            )
        return httpx.Response(status_code, json=payload, request=request)

    return handler


@asynccontextmanager
async def linear_client(handler: ResponseHandler) -> AsyncIterator[LinearTaskClient]:
    client = LinearTaskClient(
        api_key="lin_api_secret",
        team_id="team-123",
        transport=httpx.MockTransport(handler),
    )
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_create_task_posts_expected_graphql_request_and_returns_task() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=successful_payload(), request=request)

    async with linear_client(handler) as client:
        result = await client.create_task(title="Requested title")

    assert result == CreatedTask(
        identifier="APP-42",
        title="Returned Linear title",
        url="https://linear.app/example/issue/APP-42",
    )
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url == httpx.URL(LINEAR_GRAPHQL_URL)
    assert request.headers["Authorization"] == "lin_api_secret"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "query": CREATE_ISSUE_MUTATION,
        "variables": {
            "input": {
                "teamId": "team-123",
                "title": "Requested title",
            }
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 429, 500])
async def test_create_task_maps_http_error_statuses_to_task_tracker_error(
    status_code: int,
) -> None:
    async with linear_client(
        json_handler({"error": "request failed"}, status_code=status_code)
    ) as client:
        with pytest.raises(TaskTrackerError, match="Linear request failed") as exc:
            await client.create_task(title="Task")

    assert isinstance(exc.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.asyncio
async def test_create_task_maps_transport_errors_to_task_tracker_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Linear is unavailable", request=request)

    async with linear_client(handler) as client:
        with pytest.raises(TaskTrackerError, match="Linear request failed") as exc:
            await client.create_task(title="Task")

    assert isinstance(exc.value.__cause__, httpx.ConnectError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
    ],
)
async def test_create_task_marks_ambiguous_transport_result_as_uncertain(
    error_type: type[httpx.TransportError],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error_type("timed out", request=request)

    async with linear_client(handler) as client:
        with pytest.raises(TaskCreationUncertainError) as exc:
            await client.create_task(title="Task")

    assert isinstance(exc.value.__cause__, error_type)


@pytest.mark.asyncio
async def test_create_task_maps_malformed_json_to_task_tracker_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"this is not JSON",
            headers={"Content-Type": "application/json"},
            request=request,
        )

    async with linear_client(handler) as client:
        with pytest.raises(TaskTrackerError, match="Linear request failed") as exc:
            await client.create_task(title="Task")

    assert isinstance(exc.value.__cause__, ValueError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"errors": [{"message": "Team not found"}]},
        {
            **successful_payload(),
            "errors": [{"message": "Creation was rejected"}],
        },
    ],
    ids=["errors-only", "errors-take-precedence-over-data"],
)
async def test_create_task_rejects_graphql_errors_even_when_data_is_present(
    payload: dict[str, Any],
) -> None:
    async with linear_client(json_handler(payload)) as client:
        with pytest.raises(
            TaskTrackerError,
            match="Linear rejected task creation",
        ):
            await client.create_task(title="Task")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not an object",
        42,
    ],
    ids=["null", "array", "string", "number"],
)
async def test_create_task_rejects_non_object_graphql_payload(payload: Any) -> None:
    async with linear_client(json_handler(payload)) as client:
        with pytest.raises(
            TaskTrackerError,
            match="Linear rejected task creation",
        ):
            await client.create_task(title="Task")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": {}},
        {"data": None},
        {"data": []},
        {"data": {"issueCreate": None}},
        {"data": {"issueCreate": []}},
        {
            "data": {
                "issueCreate": {
                    "success": False,
                    "issue": successful_payload()["data"]["issueCreate"]["issue"],
                }
            }
        },
        {"data": {"issueCreate": {"success": True}}},
        {"data": {"issueCreate": {"success": True, "issue": None}}},
        {"data": {"issueCreate": {"success": True, "issue": []}}},
    ],
    ids=[
        "missing-data",
        "missing-issue-create",
        "null-data",
        "non-object-data",
        "null-issue-create",
        "non-object-issue-create",
        "success-false",
        "missing-issue",
        "null-issue",
        "non-object-issue",
    ],
)
async def test_create_task_rejects_incomplete_or_unsuccessful_response(
    payload: dict[str, Any],
) -> None:
    async with linear_client(json_handler(payload)) as client:
        with pytest.raises(TaskTrackerError):
            await client.create_task(title="Task")


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["identifier", "title", "url"])
async def test_create_task_rejects_missing_task_fields(field: str) -> None:
    issue = {
        "identifier": "APP-42",
        "title": "Task",
        "url": "https://linear.app/example/issue/APP-42",
    }
    issue.pop(field)

    async with linear_client(json_handler(successful_payload(issue=issue))) as client:
        with pytest.raises(TaskTrackerError, match="Linear returned an invalid task"):
            await client.create_task(title="Task")


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["identifier", "title", "url"])
@pytest.mark.parametrize(
    "invalid_value",
    [None, 1, True, [], {}, "", "   "],
    ids=["null", "number", "boolean", "array", "object", "empty", "whitespace"],
)
async def test_create_task_rejects_invalid_task_field_values(
    field: str,
    invalid_value: Any,
) -> None:
    issue: dict[str, Any] = {
        "identifier": "APP-42",
        "title": "Task",
        "url": "https://linear.app/example/issue/APP-42",
    }
    issue[field] = invalid_value

    async with linear_client(json_handler(successful_payload(issue=issue))) as client:
        with pytest.raises(TaskTrackerError, match="Linear returned an invalid task"):
            await client.create_task(title="Task")


@pytest.mark.asyncio
async def test_close_closes_underlying_http_client_and_is_idempotent() -> None:
    client = LinearTaskClient(
        api_key="lin_api_secret",
        team_id="team-123",
        transport=httpx.MockTransport(json_handler(successful_payload())),
    )

    assert client._client.is_closed is False

    await client.close()
    await client.close()

    assert client._client.is_closed is True
