import pytest

from sharpa_interface.server.action import parse_policy_action
from sharpa_interface.server.execution import (
    ExecutionValidationError,
    SyncExecutionGate,
    SyncPhase,
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
        "executed_steps": 4,
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


def test_failed_execution_requires_explicit_resume_and_preserves_feedback() -> None:
    gate = SyncExecutionGate()
    request_id, _ = gate.begin_inference()
    response = valid_action()
    response["request_id"] = request_id
    action = parse_policy_action(
        response,
        expected_session_id="episode-trex",
        expected_request_id=request_id,
    )
    gate.accept_action(action)

    feedback = gate.complete(
        execution_done(
            executed_steps=5,
            success=False,
            error="machine execution gate closed",
        )
    )

    assert gate.phase is SyncPhase.FAILED
    assert feedback["success"] is False
    with pytest.raises(RuntimeError, match="phase=failed"):
        gate.begin_inference()

    gate.resume_after_failure()

    next_request_id, next_feedback = gate.begin_inference()
    assert next_request_id == 1
    assert next_feedback == feedback


def test_async_is_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="not supported"):
        SyncExecutionGate("async")


def test_rtc_prefetch_buffers_next_action_until_current_done() -> None:
    gate = SyncExecutionGate()
    first_request, _ = gate.begin_inference()
    first_payload = valid_action()
    first_payload["request_id"] = first_request
    first_payload["execution"]["rtc"] = 2
    first = parse_policy_action(
        first_payload,
        expected_session_id="episode-trex",
        expected_request_id=first_request,
    )
    gate.accept_action(first)

    next_request, feedback = gate.begin_prefetch(
        request_id=first.request_id,
        action_id=first.action_id,
        revision=first.revision,
        execute_start=first.execution.execute_start,
        execute_length=first.execution.execute_length,
        executed_steps=6,
    )
    assert feedback == {
        "last_action_id": first.action_id,
        "executed_steps": 2,
        "success": True,
    }

    next_payload = valid_action()
    next_payload.update(
        request_id=next_request,
        action_id="episode-trex:chunk:12",
        revision=0,
    )
    next_action = parse_policy_action(
        next_payload,
        expected_session_id="episode-trex",
        expected_request_id=next_request,
    )
    gate.accept_action(next_action)

    assert gate.active_action is first
    assert gate.queued_action is next_action
    assert gate.phase is SyncPhase.BUFFERED

    gate.complete(execution_done())

    assert gate.active_action is next_action
    assert gate.queued_action is None
    assert gate.phase is SyncPhase.EXECUTING


def test_rtc_current_may_finish_before_prefetched_response_arrives() -> None:
    gate = SyncExecutionGate()
    first_request, _ = gate.begin_inference()
    first_payload = valid_action()
    first_payload["request_id"] = first_request
    first_payload["execution"]["rtc"] = 2
    first = parse_policy_action(
        first_payload,
        expected_session_id="episode-trex",
        expected_request_id=first_request,
    )
    gate.accept_action(first)
    next_request, _ = gate.begin_prefetch(
        request_id=first.request_id,
        action_id=first.action_id,
        revision=first.revision,
        execute_start=first.execution.execute_start,
        execute_length=first.execution.execute_length,
        executed_steps=6,
    )

    gate.complete(execution_done())
    assert gate.active_action is None
    assert gate.active_request_id == next_request
    assert gate.phase is SyncPhase.PREFETCHING

    next_payload = valid_action()
    next_payload.update(
        request_id=next_request,
        action_id="episode-trex:chunk:12",
        revision=0,
    )
    next_action = parse_policy_action(
        next_payload,
        expected_session_id="episode-trex",
        expected_request_id=next_request,
    )
    gate.accept_action(next_action)

    assert gate.active_action is next_action
    assert gate.phase is SyncPhase.EXECUTING


def test_rtc_prefetch_rejects_the_wrong_remaining_step_count() -> None:
    gate = SyncExecutionGate()
    request_id, _ = gate.begin_inference()
    payload = valid_action()
    payload["request_id"] = request_id
    payload["execution"]["rtc"] = 2
    action = parse_policy_action(
        payload,
        expected_session_id="episode-trex",
        expected_request_id=request_id,
    )
    gate.accept_action(action)

    with pytest.raises(ExecutionValidationError, match="exactly"):
        gate.begin_prefetch(
            request_id=action.request_id,
            action_id=action.action_id,
            revision=action.revision,
            execute_start=action.execution.execute_start,
            execute_length=action.execution.execute_length,
            executed_steps=5,
        )
