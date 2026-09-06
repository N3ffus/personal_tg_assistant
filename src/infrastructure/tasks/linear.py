from datetime import date, datetime
from typing import Any

import httpx

from src.application.ports.tasks import (
    TaskCreationUncertainError,
    TaskDeletionUncertainError,
    TaskNotFoundError,
    TaskTrackerError,
)
from src.domain.assistant.deletions import DeletionTarget
from src.domain.assistant.retrieval import (
    MAX_RETRIEVAL_ITEMS,
    MAX_RETRIEVAL_PAGES,
    RetrievalLimitError,
    TaskQuery,
)
from src.domain.tasks.models import CreatedTask, Task

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

CREATE_ISSUE_MUTATION = """
mutation CreateIssue($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      identifier
      title
      url
    }
  }
}
"""

FIND_ISSUES_QUERY = """
query FindIssues($filter: IssueFilter!, $after: String) {
  issues(filter: $filter, first: 100, after: $after, includeArchived: true) {
    nodes { id identifier title trashed }
    pageInfo { hasNextPage endCursor }
  }
}
"""

LIST_ISSUES_QUERY = """
query ListIssues($filter: IssueFilter!, $after: String) {
  issues(filter: $filter, first: 100, after: $after, includeArchived: false) {
    nodes { identifier title url trashed state { name } }
    pageInfo { hasNextPage endCursor }
  }
}
"""

DELETE_ISSUE_MUTATION = """
mutation DeleteIssue($id: String!) {
  issueDelete(id: $id) { success }
}
"""

SEARCH_ISSUES_QUERY = """
query SearchIssues($filter: IssueFilter!, $after: String, $includeArchived: Boolean!) {
  issues(filter: $filter, first: 100, after: $after, includeArchived: $includeArchived) {
    nodes {
      identifier title url trashed description priority createdAt updatedAt completedAt dueDate
      state { name type }
      project { name }
      assignee { name }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""


def task_filter(team_id: str, query: TaskQuery) -> dict[str, object]:
    filters: list[dict[str, object]] = [{"team": {"id": {"eq": team_id}}}]
    for terms in ([query.query] if query.query else [], query.search_terms):
        if terms:
            filters.append(
                {
                    "or": [
                        {field: {"containsIgnoreCase": term}}
                        for term in terms
                        for field in ("title", "description")
                    ]
                }
            )
    if query.status:
        filters.append({"state": {"name": {"eqIgnoreCase": query.status}}})
    if query.status_group != "all":
        types = {
            "open": ["triage", "backlog", "unstarted", "started"],
            "completed": ["completed"],
            "canceled": ["canceled"],
        }[query.status_group]
        filters.append({"state": {"type": {"in": types}}})
    for field in ("project", "assignee"):
        value = getattr(query, field)
        if value:
            filters.append({field: {"name": {"containsIgnoreCase": value}}})
    if query.label:
        filters.append({"labels": {"name": {"eqIgnoreCase": query.label}}})
    if query.priority is not None:
        filters.append({"priority": {"eq": query.priority}})
    dates = {}
    for key, value in (("gte", query.date_from), ("lt", query.date_to)):
        if value:
            dates[key] = (
                value.date().isoformat()
                if query.date_field == "due"
                else value.isoformat()
            )
    if dates:
        field = {
            "created": "createdAt",
            "updated": "updatedAt",
            "completed": "completedAt",
            "due": "dueDate",
        }[query.date_field]
        filters.append({field: dates})
    return filters[0] if len(filters) == 1 else {"and": filters}


def _task_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TaskTrackerError("Linear returned an invalid date")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("Missing timezone")
        return parsed
    except ValueError as error:
        raise TaskTrackerError("Linear returned an invalid date") from error


def _task_details(issue: dict[str, Any]) -> dict[str, Any]:
    try:
        due = date.fromisoformat(issue["dueDate"]) if issue.get("dueDate") else None
        priority = issue.get("priority", 0)
        if type(priority) is not int or not 0 <= priority <= 4:
            raise ValueError("Invalid priority")
        return {
            "description": str(issue.get("description") or ""),
            "priority": priority,
            "status_type": str(issue["state"].get("type") or ""),
            "created_at": _task_datetime(issue.get("createdAt")),
            "updated_at": _task_datetime(issue.get("updatedAt")),
            "completed_at": _task_datetime(issue.get("completedAt")),
            "due_date": due,
            "project": str((issue.get("project") or {}).get("name") or ""),
            "assignee": str((issue.get("assignee") or {}).get("name") or ""),
        }
    except (ValueError, TypeError, AttributeError) as error:
        raise TaskTrackerError("Linear returned invalid issue details") from error


class LinearTaskClient:
    def __init__(
        self,
        *,
        api_key: str,
        team_id: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._team_id = team_id
        self._client = httpx.AsyncClient(
            headers={"Authorization": api_key},
            timeout=10,
            transport=transport,
        )

    async def create_task(self, *, title: str) -> CreatedTask:
        try:
            response = await self._client.post(
                LINEAR_GRAPHQL_URL,
                json={
                    "query": CREATE_ISSUE_MUTATION,
                    "variables": {
                        "input": {
                            "teamId": self._team_id,
                            "title": title,
                        }
                    },
                },
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ) as error:
            raise TaskCreationUncertainError(
                "Linear task creation result is uncertain"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise TaskTrackerError("Linear request failed") from error

        if not isinstance(payload, dict) or payload.get("errors"):
            raise TaskTrackerError("Linear rejected task creation")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise TaskTrackerError("Linear returned an invalid response")

        issue_create = data.get("issueCreate")
        if (
            not isinstance(issue_create, dict)
            or issue_create.get("success") is not True
        ):
            raise TaskTrackerError("Linear did not create the task")

        issue = issue_create.get("issue")
        if not isinstance(issue, dict):
            raise TaskTrackerError("Linear returned an invalid task")

        identifier = issue.get("identifier")
        returned_title = issue.get("title")
        url = issue.get("url")
        if (
            not isinstance(identifier, str)
            or not isinstance(returned_title, str)
            or not isinstance(url, str)
            or not identifier.strip()
            or not returned_title.strip()
            or not url.strip()
        ):
            raise TaskTrackerError("Linear returned an invalid task")

        return CreatedTask(
            identifier=identifier,
            title=returned_title,
            url=url,
        )

    async def _query(
        self, *, query: str, variables: dict[str, object]
    ) -> dict[str, Any]:
        try:
            response = await self._client.post(
                LINEAR_GRAPHQL_URL, json={"query": query, "variables": variables}
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise TaskTrackerError("Linear request failed") from error
        if not isinstance(payload, dict) or payload.get("errors"):
            errors = payload.get("errors") if isinstance(payload, dict) else None
            detail = ""
            if isinstance(errors, list) and errors:
                first = errors[0]
                if isinstance(first, dict) and first.get("message"):
                    detail = f": {first['message']}"
            raise TaskTrackerError(f"Linear rejected the request{detail}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise TaskTrackerError("Linear returned an invalid response")
        return data

    async def list_tasks(self, *, query: TaskQuery | None = None) -> list[Task]:
        tasks: dict[str, Task] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(MAX_RETRIEVAL_PAGES):
            data = await self._query(
                query=SEARCH_ISSUES_QUERY if query is not None else LIST_ISSUES_QUERY,
                variables={
                    "filter": task_filter(self._team_id, query or TaskQuery()),
                    "after": cursor,
                    **(
                        {"includeArchived": query.include_archived}
                        if query is not None
                        else {}
                    ),
                },
            )
            issues = data.get("issues")
            if not isinstance(issues, dict) or not isinstance(
                issues.get("nodes"), list
            ):
                raise TaskTrackerError("Linear returned invalid issues")
            for issue in issues["nodes"]:
                if not isinstance(issue, dict) or any(
                    not isinstance(issue.get(key), str) or not issue[key].strip()
                    for key in ("identifier", "title", "url")
                ):
                    raise TaskTrackerError("Linear returned an invalid issue")
                if issue.get("trashed") is True:
                    continue
                state = issue.get("state")
                if (
                    not isinstance(state, dict)
                    or not isinstance(state.get("name"), str)
                    or not state["name"].strip()
                ):
                    raise TaskTrackerError("Linear returned an invalid issue status")
                tasks[issue["identifier"]] = Task(
                    identifier=issue["identifier"],
                    title=issue["title"],
                    url=issue["url"],
                    status=state["name"],
                    **(_task_details(issue) if query is not None else {}),
                )
                if len(tasks) > MAX_RETRIEVAL_ITEMS:
                    raise RetrievalLimitError
            page = issues.get("pageInfo")
            if not isinstance(page, dict) or not isinstance(
                page.get("hasNextPage"), bool
            ):
                raise TaskTrackerError("Linear returned invalid pagination")
            if not page["hasNextPage"]:
                return list(tasks.values())
            cursor = page.get("endCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                raise TaskTrackerError("Linear returned invalid pagination")
            seen_cursors.add(cursor)

        raise RetrievalLimitError

    async def find_tasks(self, *, title: str | None) -> list[DeletionTarget]:
        import re

        if title is not None and not title.strip():
            raise TaskTrackerError("Task search must not be blank")
        issue_filter: dict[str, object] = {"team": {"id": {"eq": self._team_id}}}
        identifier = (
            re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)-(\d+)", title.strip())
            if title
            else None
        )
        if identifier:
            issue_filter["number"] = {"eq": int(identifier[2])}
            # The team ID always remains in the filter; a foreign identifier
            # must never resolve to the same issue number in this team.
        elif title is not None:
            issue_filter["title"] = {"containsIgnoreCase": title.strip()}
        targets: dict[str, DeletionTarget] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            data = await self._query(
                query=FIND_ISSUES_QUERY,
                variables={"filter": issue_filter, "after": cursor},
            )
            issues = data.get("issues")
            if not isinstance(issues, dict) or not isinstance(
                issues.get("nodes"), list
            ):
                raise TaskTrackerError("Linear returned invalid issues")
            for issue in issues["nodes"]:
                if not isinstance(issue, dict) or any(
                    not isinstance(issue.get(key), str) or not issue[key].strip()
                    for key in ("id", "identifier", "title")
                ):
                    raise TaskTrackerError("Linear returned an invalid issue")
                if issue.get("trashed") is True:
                    continue
                if (
                    identifier
                    and title is not None
                    and issue["identifier"].casefold() != title.strip().casefold()
                ):
                    continue
                targets[issue["id"]] = DeletionTarget(
                    id=issue["id"],
                    title=issue["title"],
                    label=f"{issue['identifier']}: {issue['title']}",
                )
            page = issues.get("pageInfo")
            if not isinstance(page, dict) or not isinstance(
                page.get("hasNextPage"), bool
            ):
                raise TaskTrackerError("Linear returned invalid pagination")
            if not page["hasNextPage"]:
                return list(targets.values())
            cursor = page.get("endCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                raise TaskTrackerError("Linear returned invalid pagination")
            seen_cursors.add(cursor)

    async def delete_task(self, *, task_id: str) -> None:
        try:
            data = await self._query(
                query=DELETE_ISSUE_MUTATION, variables={"id": task_id}
            )
        except TaskTrackerError as error:
            if isinstance(
                error.__cause__,
                (
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                    httpx.ReadError,
                    httpx.WriteError,
                    httpx.RemoteProtocolError,
                ),
            ):
                raise TaskDeletionUncertainError(
                    "Linear deletion result is uncertain"
                ) from error
            if "not found" in str(error).lower():
                raise TaskNotFoundError("Linear task not found") from error
            raise
        result = data.get("issueDelete")
        if not isinstance(result, dict) or result.get("success") is not True:
            raise TaskTrackerError("Linear did not delete the task")

    async def close(self) -> None:
        await self._client.aclose()
