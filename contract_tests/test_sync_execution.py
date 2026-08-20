import pytest

from sharpa_interface.server.action import parse_policy_action
from sharpa_interface.server.execution import (
    ExecutionValidationError,
    SyncExecutionGate,
    build_execution_done,
)

from test_server_action import valid_action


def execution_done(**updates) -> dict:
    payload = {
        "schema": "sharpa_execution_done.v1",
        "request_id": 0,
        "action_id": "episode-trex:chunk:11",
        "revision": 4,
        "execute_start": 4,
        "execute_length": 4,
        "executed_steps": 8,
        "success": True,
        "done": True,
        "error": None,
        "executor_extra": None,
    }
    payload.update(updates)
    return payload


def test_sync_gate_waits_for_matching_action_execute_done() -> None:
    gate = SyncExecutionGate()
    request_id, feedback = gate.begin_inference()
    response = valid_action()
    response["request_id"] = request_id
    action = parse_policy_action(
        response,
        expected_session_id="episode-trex",
        expected_request_id=request_id,
    )
    executable = gate.accept_action(action)

    assert feedback == {"last_action_id": None, "executed_steps": 0, "success": True}
    assert executable["schema"] == "sharpa_executable_action.v1"
    assert executable["action_id"] == action.action_id
    assert executable["action"]["left_wrist"]["eef"].shape[0] == 4
    done = build_execution_done(executable, executed_in_slice=4, success=True)
    done["executor_extra"] = None
    next_feedback = gate.complete(done)
    assert next_feedback == {
        "last_action_id": "episode-trex:chunk:11",
        "executed_steps": 8,
        "success": True,
    }
    assert gate.begin_inference()[0] == 1


def test_sync_gate_rejects_stale_done() -> None:
    gate = SyncExecutionGate()
    request_id, _ = gate.begin_inference()
    response = valid_action()
    response["request_id"] = request_id
    gate.accept_action(
        parse_policy_action(
            response,
            expected_session_id="episode-trex",
            expected_request_id=request_id,
        )
    )

    with pytest.raises(ExecutionValidationError, match="does not match"):
        gate.complete(execution_done(action_id="stale"))


def test_async_is_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="not supported"):
        SyncExecutionGate("async")
