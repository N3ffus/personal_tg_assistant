from dataclasses import dataclass
from datetime import datetime

import pytest

from evals.checks import assert_contract
from evals.harness import run_scenario
from evals.scenarios import SCENARIOS, Scenario
from src.domain.assistant.models import AssistantDecision


@dataclass
class GoldenLLM:
    scenario: Scenario

    async def parse_message(
        self, *, text: str, now: datetime, timezone: str, context: str = ""
    ) -> AssistantDecision:
        assert text == self.scenario.prompt
        assert now == self.scenario.now
        assert timezone == self.scenario.timezone
        return self.scenario.golden_decision()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.id)
async def test_integration_contract(scenario: Scenario) -> None:
    run = await run_scenario(scenario, GoldenLLM(scenario))
    assert_contract(scenario, run, exact_text=True)
    expected_errors = {
        "linear_error": "TaskTrackerError",
        "linear_uncertain": "TaskCreationUncertainError",
        "calendar_error": "CalendarError",
        "calendar_disconnected": "CalendarNotConnectedError",
    }
    assert [call.error for call in run.calls if call.error] == (
        [expected_errors[scenario.failure]] if scenario.failure else []
    )
    for call in run.calls:
        if call.error:
            assert call.output is None
        else:
            assert call.output is not None
