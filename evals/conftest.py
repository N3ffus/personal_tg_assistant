from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from pydantic import ValidationError

from evals.settings import load_eval_settings

if TYPE_CHECKING:
    from evals.judge import DeepInfraJudge


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-llm-evals",
        action="store_true",
        default=False,
        help="Run paid prompt evaluations against the application LLM and DeepInfra judge",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-llm-evals"):
        try:
            load_eval_settings()
        except ValidationError as exc:
            fields = sorted({str(error["loc"][0]).upper() for error in exc.errors()})
            raise pytest.UsageError(
                "Missing or invalid eval settings: "
                + ", ".join(fields)
                + ". Configure .env.eval or environment variables; see evals/README.md."
            ) from None
        return
    skip = pytest.mark.skip(reason="Live LLM evaluation requires --run-llm-evals")
    for item in items:
        if "llm_eval" in item.keywords:
            item.add_marker(skip)


@pytest_asyncio.fixture
async def judge() -> AsyncIterator["DeepInfraJudge"]:
    from evals.judge import DeepInfraJudge

    instance = DeepInfraJudge(load_eval_settings())
    try:
        yield instance
    finally:
        await instance.close()
