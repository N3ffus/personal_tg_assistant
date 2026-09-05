from evals.harness import EvaluationRun
from evals.scenarios import Scenario


def assert_contract(
    scenario: Scenario, run: EvaluationRun, *, exact_text: bool = False
) -> None:
    """Hard invariants cannot be waived by a high semantic judge score."""
    expected = scenario.golden_decision()
    assert [a.type for a in run.decision.actions] == [a.type for a in expected.actions]
    assert tuple(call.name for call in run.calls) == scenario.tools
    for actual, golden in zip(run.decision.actions, expected.actions, strict=True):
        golden_data = golden.model_dump(mode="json")
        actual_data = actual.model_dump(mode="json")
        if exact_text:
            assert actual_data == golden_data
        if "starts_at" in golden_data:
            assert actual_data["starts_at"] == golden_data["starts_at"]
        for key in ("title", "text"):
            if key in actual_data:
                assert actual_data[key].strip()

    action_index = 0
    for call in run.calls:
        if call.name.startswith("calendar."):
            assert call.arguments["user_id"] == scenario.user_id
        if call.name == "calendar.update_event":
            assert call.arguments["event_id"] == scenario.selected_event_id
        if call.name == "calendar.list_events":
            assert call.arguments["now"] == scenario.now.isoformat()
        # Compare observed arguments to the LLM decision, not reconstructed tool calls.
        while action_index < len(run.decision.actions):
            action = run.decision.actions[action_index]
            action_index += 1
            if call.name.rsplit(".", 1)[1] == action.type.value:
                data = action.model_dump(mode="json")
                for key in ("title", "starts_at"):
                    if key in data:
                        assert call.arguments[key] == data[key]
                break
        else:
            raise AssertionError(f"Unexpected integration call: {call.name}")
    for fragment in scenario.reply_contains:
        assert fragment in run.reply
