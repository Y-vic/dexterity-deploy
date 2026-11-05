import json

from ur_common.execution_log import EVENT_SCHEMA, execution_event_json


def test_execution_event_has_correlation_clocks_and_fields():
    payload = json.loads(
        execution_event_json(
            "ur_node", "hardware_command_received", action_id="a", step_index=3
        )
    )

    assert payload["schema"] == EVENT_SCHEMA
    assert payload["node"] == "ur_node"
    assert payload["event"] == "hardware_command_received"
    assert payload["action_id"] == "a"
    assert payload["step_index"] == 3
    assert payload["timestamp_ns"] > 0
    assert payload["monotonic_ns"] > 0
