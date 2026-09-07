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
    found_event_ids: list[str] = []
    for call in run.calls:
        if call.name.startswith("calendar."):
            assert call.arguments["user_id"] == scenario.user_id
        if call.name == "calendar.find_events":
            targets = call.output
            assert isinstance(targets, list)
            found_event_ids = [target["id"] for target in targets]
        if call.name == "calendar.update_event":
            # Editing may only touch the single event the lookup resolved.
            assert found_event_ids == [call.arguments["event_id"]]
        if call.name == "calendar.list_events":
            assert call.arguments["now"] == scenario.now.isoformat()
        # Compare observed arguments to the LLM decision, not reconstructed tool calls.
        while action_index < len(run.decision.actions):
            action = run.decision.actions[action_index]
            data = action.model_dump(mode="json")
            # An update is a lookup call followed by the update of the same action.
            if (
                action.type.value == "update_event"
                and call.name == "calendar.find_events"
            ):
                assert call.arguments["title"] == data["event_title"]
                break
            action_index += 1
            lookup_calls = {
                "delete_task": "linear.find_tasks",
                "delete_all_tasks": "linear.find_tasks",
                "delete_event": "calendar.find_events",
                "delete_all_events": "calendar.find_events",
            }
            if (
                call.name.rsplit(".", 1)[1] == action.type.value
                or lookup_calls.get(action.type.value) == call.name
            ):
                for key in ("title", "starts_at"):
                    if key in data:
                        assert call.arguments[key] == data[key]
                break
        else:
            raise AssertionError(f"Unexpected integration call: {call.name}")
    for fragment in scenario.reply_contains:
        assert fragment in run.reply
