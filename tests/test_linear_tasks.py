import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from src.application.ports.tasks import (
    TaskCreationUncertainError,
    TaskDeletionUncertainError,
    TaskNotFoundError,
    TaskTrackerError,
)
from src.domain.tasks.models import CreatedTask, Task
from src.infrastructure.tasks.linear import (
    CREATE_ISSUE_MUTATION,
    LINEAR_GRAPHQL_URL,
    LIST_ISSUES_QUERY,
    LinearTaskClient,
)

ResponseHandler = Callable[[httpx.Request], httpx.Response]


def task_page(*, number: int = 1, next_cursor: str | None = None) -> dict[str, Any]:
    payload = issue_page(number=number, next_cursor=next_cursor)
    issue = payload["data"]["issues"]["nodes"][0]
    issue["url"] = f"https://linear.app/example/issue/APP-{number}"
    issue["state"] = {"name": "In Progress"}
    return payload


@pytest.mark.asyncio
async def test_list_tasks_reads_all_pages_scoped_to_team_and_excludes_archived() -> (
    None
):
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        page = (
            task_page(number=2)
            if payload["variables"]["after"]
            else task_page(next_cursor="next")
        )
        return httpx.Response(200, json=page)

    async with linear_client(handler) as client:
        result = await client.list_tasks()
    assert result == [
        Task(
            identifier=f"APP-{i}",
            title=f"Task {i}",
            url=f"https://linear.app/example/issue/APP-{i}",
            status="In Progress",
        )
        for i in (1, 2)
    ]
    assert [r["variables"]["after"] for r in requests] == [None, "next"]
    assert all(
        r["variables"]["filter"] == {"team": {"id": {"eq": "team-123"}}}
        for r in requests
    )
    assert all(r["query"] == LIST_ISSUES_QUERY for r in requests)
    assert "includeArchived: false" in LIST_ISSUES_QUERY
    assert "state { name }" in LIST_ISSUES_QUERY


@pytest.mark.asyncio
async def test_list_tasks_empty_and_repeated_issues() -> None:
    empty = task_page()
    empty["data"]["issues"]["nodes"] = []
    async with linear_client(json_handler(empty)) as client:
        assert await client.list_tasks() == []

    def handler(request: httpx.Request) -> httpx.Response:
        after = json.loads(request.content)["variables"]["after"]
        return httpx.Response(
            200, json=task_page(next_cursor=None if after else "next")
        )

    async with linear_client(handler) as client:
        assert len(await client.list_tasks()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "issues",
    [
        None,
        {"nodes": None},
        {"nodes": [None]},
        {"nodes": [], "pageInfo": None},
        {"nodes": [], "pageInfo": {"hasNextPage": "false"}},
        {"nodes": [], "pageInfo": {"hasNextPage": True, "endCursor": None}},
    ],
)
async def test_list_tasks_rejects_incomplete_responses(issues: object) -> None:
    async with linear_client(json_handler({"data": {"issues": issues}})) as client:
        with pytest.raises(TaskTrackerError):
            await client.list_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("identifier", ""),
        ("title", None),
        ("url", 123),
        ("state", None),
        ("state", {"name": None}),
        ("state", {"name": " "}),
    ],
)
async def test_list_tasks_rejects_invalid_task_fields(key: str, value: object) -> None:
    payload = task_page()
    payload["data"]["issues"]["nodes"][0][key] = value
    async with linear_client(json_handler(payload)) as client:
        with pytest.raises(TaskTrackerError):
            await client.list_tasks()


@pytest.mark.asyncio
async def test_list_tasks_rejects_repeated_pagination_cursor() -> None:
    async with linear_client(json_handler(task_page(next_cursor="same"))) as client:
        with pytest.raises(TaskTrackerError, match="pagination"):
            await client.list_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "graphql", "transport"])
async def test_list_tasks_does_not_return_partial_results_after_page_failure(
    failure: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["variables"]["after"] is None:
            return httpx.Response(200, json=task_page(next_cursor="next"))
        if failure == "transport":
            raise httpx.ReadTimeout("unavailable", request=request)
        return httpx.Response(
            503 if failure == "http" else 200,
            json={**task_page(number=2), "errors": [{"message": "unavailable"}]},
        )

    async with linear_client(handler) as client:
        with pytest.raises(TaskTrackerError):
            await client.list_tasks()


def issue_page(*, number: int = 1, next_cursor: str | None = None) -> dict[str, Any]:
    return {
        "data": {
            "issues": {
                "nodes": [
                    {
                        "id": f"uuid-{number}",
                        "identifier": f"APP-{number}",
                        "title": f"Task {number}",
                    }
                ],
                "pageInfo": {
                    "hasNextPage": next_cursor is not None,
                    "endCursor": next_cursor,
                },
            }
        }
    }


@pytest.mark.asyncio
async def test_find_all_tasks_paginates_and_restricts_team_including_archived() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        page = (
            issue_page(number=2)
            if payload["variables"]["after"]
            else issue_page(next_cursor="cursor-1")
        )
        return httpx.Response(200, json=page)

    async with linear_client(handler) as client:
        targets = await client.find_tasks(title=None)
    assert [t.id for t in targets] == ["uuid-1", "uuid-2"]
    assert [r["variables"]["after"] for r in requests] == [None, "cursor-1"]
    assert all(
        r["variables"]["filter"] == {"team": {"id": {"eq": "team-123"}}}
        for r in requests
    )
    assert all("includeArchived: true" in r["query"] for r in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("title", "expected_filter", "count"),
    [
        ("Task", {"title": {"containsIgnoreCase": "Task"}}, 1),
        ("app-1", {"number": {"eq": 1}}, 1),
        ("OTHER-1", {"number": {"eq": 1}}, 0),
    ],
)
async def test_find_task_by_title_or_identifier(
    title: str, expected_filter: dict[str, object], count: int
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=issue_page())

    async with linear_client(handler) as client:
        targets = await client.find_tasks(title=title)
    assert len(targets) == count
    assert requests[0]["variables"]["filter"] == {
        "team": {"id": {"eq": "team-123"}},
        **expected_filter,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"errors": [{"message": "denied"}]},
        {"data": None},
        {"data": {"issues": None}},
        {"data": {"issues": {"nodes": [None]}}},
        {"data": {"issues": {"nodes": [], "pageInfo": None}}},
        {
            "data": {
                "issues": {
                    "nodes": [],
                    "pageInfo": {"hasNextPage": True, "endCursor": None},
                }
            }
        },
    ],
)
async def test_find_tasks_rejects_malformed_or_partial_responses(payload: Any) -> None:
    async with linear_client(json_handler(payload)) as client:
        with pytest.raises(TaskTrackerError):
            await client.find_tasks(title=None)


@pytest.mark.asyncio
async def test_find_tasks_rejects_blank_search_and_repeated_cursor() -> None:
    async with linear_client(json_handler(issue_page(next_cursor="same"))) as client:
        with pytest.raises(TaskTrackerError):
            await client.find_tasks(title=" ")
        with pytest.raises(TaskTrackerError, match="pagination"):
            await client.find_tasks(title=None)


@pytest.mark.asyncio
async def test_delete_task_uses_issue_delete_and_checks_success() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"issueDelete": {"success": True}}})

    async with linear_client(handler) as client:
        await client.delete_task(task_id="uuid-1")
    assert requests[0]["variables"] == {"id": "uuid-1"}
    assert "issueDelete(id: $id)" in requests[0]["query"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"issueDelete": {"success": False}}},
        {"data": {}},
        {"errors": ["denied"]},
    ],
)
async def test_delete_task_rejects_unsuccessful_response(
    payload: dict[str, Any],
) -> None:
    async with linear_client(json_handler(payload)) as client:
        with pytest.raises(TaskTrackerError):
            await client.delete_task(task_id="uuid-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
async def test_delete_task_transport_failure_is_not_retried(
    error: type[httpx.TransportError],
) -> None:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        raise error("failure", request=request)

    async with linear_client(handler) as client:
        with pytest.raises(
            TaskDeletionUncertainError
            if error is httpx.ReadTimeout
            else TaskTrackerError
        ):
            await client.delete_task(task_id="uuid-1")
    assert count == 1


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


@pytest.mark.asyncio
async def test_list_tasks_skips_trashed_issues() -> None:
    payload = {
        "data": {
            "issues": {
                "nodes": [
                    {
                        "identifier": "APP-1",
                        "title": "Trashed Task",
                        "url": "https://linear.app/example/issue/APP-1",
                        "trashed": True,
                        "state": {"name": "Backlog"},
                    },
                    {
                        "identifier": "APP-2",
                        "title": "Active Task",
                        "url": "https://linear.app/example/issue/APP-2",
                        "trashed": False,
                        "state": {"name": "In Progress"},
                    },
                ],
                "pageInfo": {
                    "hasNextPage": False,
                    "endCursor": None,
                },
            }
        }
    }
    async with linear_client(json_handler(payload)) as client:
        result = await client.list_tasks()
    assert result == [
        Task(
            identifier="APP-2",
            title="Active Task",
            url="https://linear.app/example/issue/APP-2",
            status="In Progress",
        )
    ]


@pytest.mark.asyncio
async def test_find_tasks_skips_trashed_issues() -> None:
    payload = {
        "data": {
            "issues": {
                "nodes": [
                    {
                        "id": "uuid-1",
                        "identifier": "APP-1",
                        "title": "Trashed Task",
                        "trashed": True,
                    },
                    {
                        "id": "uuid-2",
                        "identifier": "APP-2",
                        "title": "Active Task",
                        "trashed": False,
                    },
                ],
                "pageInfo": {
                    "hasNextPage": False,
                    "endCursor": None,
                },
            }
        }
    }
    async with linear_client(json_handler(payload)) as client:
        targets = await client.find_tasks(title=None)
    assert [t.id for t in targets] == ["uuid-2"]


@pytest.mark.asyncio
async def test_delete_task_maps_entity_not_found_to_task_not_found_error() -> None:
    error_payload = {
        "errors": [
            {
                "message": "Entity not found: Issue",
                "extensions": {
                    "type": "invalid input",
                    "code": "INPUT_ERROR",
                },
            }
        ],
        "data": None,
    }
    async with linear_client(json_handler(error_payload)) as client:
        with pytest.raises(TaskNotFoundError, match="not found"):
            await client.delete_task(task_id="uuid-already-deleted")
