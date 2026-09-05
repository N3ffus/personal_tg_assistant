import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import httpx2 as httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI, AuthenticationError, OpenAI
from pydantic import BaseModel, SecretStr, ValidationError

pytest.importorskip(
    "deepeval", reason="Install the eval dependency group to test the judge"
)

from deepeval.metrics import GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

from evals.judge import DeepInfraJudge
from evals.settings import DEEPINFRA_BASE_URL, JUDGE_MODEL, EvalSettings


class Verdict(BaseModel):
    score: int
    reason: str


@dataclass
class JudgeTransport:
    requests: list[httpx.Request] = field(default_factory=list)
    content: str | None = '{"score": 9, "reason": "Matches the requirements"}'
    finish_reason: str = "stop"
    empty_choices: bool = False
    status: int = 200

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self.status,
            json={
                "id": "eval-completion",
                "object": "chat.completion",
                "created": 0,
                "model": JUDGE_MODEL,
                "choices": []
                if self.empty_choices
                else [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.content},
                        "finish_reason": self.finish_reason,
                    }
                ],
            },
        )


@pytest_asyncio.fixture
async def mock_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[DeepInfraJudge, JudgeTransport]]:
    transport = JudgeTransport()
    monkeypatch.setattr(
        "evals.judge.OpenAI",
        partial(
            OpenAI,
            http_client=httpx.Client(transport=httpx.MockTransport(transport.handle)),
        ),
    )
    monkeypatch.setattr(
        "evals.judge.AsyncOpenAI",
        partial(
            AsyncOpenAI,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(transport.handle)
            ),
        ),
    )
    settings = EvalSettings(  # type: ignore[call-arg]
        _env_file=None,
        deepinfra_api_key=SecretStr("judge-test-key"),
        llm_api_key=SecretStr("application-test-key"),
    )
    judge = DeepInfraJudge(settings)
    try:
        yield judge, transport
    finally:
        await judge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_judge_uses_deepinfra_and_validates_schema(
    mock_judge: tuple[DeepInfraJudge, JudgeTransport], asynchronous: bool
) -> None:
    judge, transport = mock_judge
    result = (
        await judge.a_generate("Evaluate this output", schema=Verdict)
        if asynchronous
        else judge.generate("Evaluate this output", schema=Verdict)
    )
    assert isinstance(result, Verdict)
    assert result.score == 9
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert str(request.url) == DEEPINFRA_BASE_URL + "/chat/completions"
    assert request.headers["authorization"] == "Bearer judge-test-key"
    body = json.loads(request.content)
    assert body["model"] == JUDGE_MODEL
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert '"score"' in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "Evaluate this output"
    assert judge.get_model_name() == JUDGE_MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json", '{"score":"invalid"}', "null"])
async def test_judge_rejects_invalid_verdict(
    mock_judge: tuple[DeepInfraJudge, JudgeTransport], content: str
) -> None:
    judge, transport = mock_judge
    transport.content = content
    with pytest.raises(ValidationError):
        await judge.a_generate("Evaluate", schema=Verdict)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty_choices", "empty_content", "truncated"])
async def test_judge_rejects_missing_or_truncated_output(
    mock_judge: tuple[DeepInfraJudge, JudgeTransport], failure: str
) -> None:
    judge, transport = mock_judge
    if failure == "empty_choices":
        transport.empty_choices = True
    elif failure == "empty_content":
        transport.content = None
    else:
        transport.finish_reason = "length"
    with pytest.raises(ValueError, match="DeepInfra judge returned"):
        await judge.a_generate("Evaluate", schema=Verdict)


@pytest.mark.asyncio
async def test_judge_does_not_fallback_on_authentication_error(
    mock_judge: tuple[DeepInfraJudge, JudgeTransport],
) -> None:
    judge, transport = mock_judge
    transport.status = 401
    with pytest.raises(AuthenticationError):
        await judge.a_generate("Evaluate", schema=Verdict)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_geval_accepts_custom_judge_schema(
    mock_judge: tuple[DeepInfraJudge, JudgeTransport],
) -> None:
    judge, transport = mock_judge
    metric = GEval(
        name="Test correctness",
        model=judge,
        threshold=0.8,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=["Check arithmetic correctness."],
    )
    await metric.a_measure(LLMTestCase(input="2+2?", actual_output="4"))
    assert metric.score == 0.9
    assert metric.is_successful()
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actual", "expected", "passes"),
    [
        ([], [], True),
        (["linear.create_task"], [], False),
        ([], ["linear.create_task"], False),
        (["linear.create_task"] * 2, ["linear.create_task"] * 2, True),
        (["linear.create_task"], ["linear.create_task"] * 2, False),
        (["linear.create_task"] * 2, ["linear.create_task"], False),
        (
            ["calendar.create_event", "linear.create_task"],
            ["linear.create_task", "calendar.create_event"],
            False,
        ),
    ],
)
async def test_tool_metric_rejects_extra_missing_reordered_and_duplicate_calls(
    mock_judge: tuple[DeepInfraJudge, JudgeTransport],
    actual: list[str],
    expected: list[str],
    passes: bool,
) -> None:
    judge, transport = mock_judge
    metric = ToolCorrectnessMetric(
        model=judge, threshold=1, should_exact_match=True, should_consider_ordering=True
    )
    case = LLMTestCase(
        input="Synthetic integration trace",
        actual_output="Synthetic response",
        tools_called=[ToolCall(name=name, input_parameters=None) for name in actual],
        expected_tools=[
            ToolCall(name=name, input_parameters=None) for name in expected
        ],
    )
    await metric.a_measure(case)
    assert metric.is_successful() is passes
    assert transport.requests == []


@pytest.mark.parametrize("field_name", ["deepinfra_api_key", "llm_api_key"])
def test_eval_settings_reject_blank_keys(field_name: str) -> None:
    values: dict[str, Any] = {
        "_env_file": None,
        "deepinfra_api_key": "test",
        "llm_api_key": "test",
        field_name: "   ",
    }
    with pytest.raises(ValidationError):
        EvalSettings(**values)
