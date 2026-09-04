from typing import Any

import httpx

from src.application.ports.tasks import TaskCreationUncertainError, TaskTrackerError
from src.domain.tasks.models import CreatedTask

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

    async def close(self) -> None:
        await self._client.aclose()
