import numpy as np
import pytest
from types import SimpleNamespace

from policy_client_node.policy_client_node import (
    PolicyClientCore,
    PolicyRequestMetrics,
    activate_reset_metadata,
    completion_belongs_to_session,
    deployment_ready,
    metadata_identity_error,
    summarize_metadata,
    validate_server_eef_contract,
)


def _metadata_format(format_id):
    return {
        "format_id": format_id,
        "image": {},
        "state": {"history_len": 0, "current": True},
        "sensor": {},
    }


def test_reset_response_reactivates_trex_slow_format_for_request_zero():
    metadata = {"metadata_format": _metadata_format("trex_fast_v1")}

    activated = activate_reset_metadata(
        metadata,
        {"metadata_format": _metadata_format("trex_slow_v1")},
    )

    assert activated["metadata_format"]["format_id"] == "trex_slow_v1"
    assert metadata["metadata_format"]["format_id"] == "trex_fast_v1"


def test_reset_response_requires_an_explicit_metadata_format():
    with pytest.raises(ValueError, match="missing metadata_format"):
        activate_reset_metadata({}, {"reset": True})


def action(session, request):
    eef = np.zeros((2, 9), dtype=np.float32); eef[:, 3] = 1; eef[:, 7] = 1
    return {
        "schema": "sharpa_policy_action.v5", "session_id": session,
        "request_id": request, "action_id": "a1", "revision": 0,
        "timestamp_ns": 1,
        "execution": {"frequency_hz": 30.0, "action_length": 2, "execute_start": 0, "execute_length": 2, "rtc": 0},
        "action": {
            "left_wrist": {"joint": None, "eef": eef, "eef_def": "absolute"},
            "right_wrist": {"joint": None, "eef": eef, "eef_def": "absolute"},
            "hand_joint": {"left": np.zeros((2, 22), np.float32), "right": np.zeros((2, 22), np.float32)},
        },
        "auxiliary": {"video": {"ego": None, "left_wrist": None, "right_wrist": None}, "tactile": {"deformation": None, "wrench": None, "hand_tau": None}},
        "diagnostics": None, "next_metadata_format": None,
    }


def test_core_preserves_gate_and_cumulative_done():
    core = PolicyClientCore("s")
    assert core.ready
    request = core.begin()
    assert not core.ready
    command = core.accept_action(action("s", request.request_id))
    assert not core.ready
    assert command["execution"]["execute_start"] == 0
    feedback = core.complete({"schema": "sharpa_execution_done.v1", "request_id": 0, "action_id": "a1", "revision": 0, "execute_start": 0, "execute_length": 2, "executed_steps": 2, "success": True, "done": True, "error": None})
    assert feedback["executed_steps"] == 2
    assert core.ready
    assert core.begin().request_id == 1


def test_core_prefetch_keeps_current_action_until_terminal_done():
    core = PolicyClientCore("s")
    request = core.begin()
    first = action("s", request.request_id)
    first["execution"]["rtc"] = 1
    core.accept_action(first)

    pending = core.prefetch(
        SimpleNamespace(
            request_id=0,
            action_id="a1",
            revision=0,
            execute_start=0,
            execute_length=2,
            executed_steps=1,
        )
    )
    assert pending.request_id == 1
    assert core.executing
    assert core.inference_pending

    second = action("s", pending.request_id)
    second["action_id"] = "a2"
    core.accept_action(second)
    assert core.phase == "buffered"
    assert core.executing

    core.complete(
        {
            "schema": "sharpa_execution_done.v1",
            "request_id": 0,
            "action_id": "a1",
            "revision": 0,
            "execute_start": 0,
            "execute_length": 2,
            "executed_steps": 2,
            "success": True,
            "done": True,
            "error": None,
        }
    )
    assert core.phase == "executing"
    assert core.gate.active_action.action_id == "a2"


def test_model_lifecycle_reset_clears_request_metrics_and_request_sequence():
    metrics = PolicyRequestMetrics()
    metrics.record(7, 0.125)
    assert metrics.inference_count == 1
    assert metrics.last_request_id == 7

    metrics.reset()

    assert metrics.inference_count == 0
    assert metrics.last_request_id == 0
    assert metrics.latency_s is None

    core = PolicyClientCore("s")
    first = core.begin()
    core.cancel_request()
    assert first.request_id == 0
    core.reset()
    assert core.begin().request_id == 0


def test_core_accepts_the_action_already_parsed_by_transport():
    from sharpa_interface.server.action import parse_policy_action

    core = PolicyClientCore("s")
    request = core.begin()
    parsed = parse_policy_action(
        action("s", request.request_id),
        expected_session_id="s",
        expected_request_id=request.request_id,
    )
    command = core.accept_parsed_action(parsed)
    assert command["action_id"] == "a1"
    assert not core.ready


def test_core_rejects_done_identity_mismatch():
    core = PolicyClientCore("s"); request = core.begin(); core.accept_action(action("s", request.request_id))
    with pytest.raises(Exception):
        core.complete({"schema": "sharpa_execution_done.v1", "request_id": 9, "action_id": "a1", "revision": 0, "execute_start": 0, "execute_length": 2, "executed_steps": 2, "success": True, "done": True, "error": None})


def test_stop_then_start_recovers_failed_chunk_and_keeps_failure_feedback():
    core = PolicyClientCore("s")
    core.update_actor_gate(True)
    request = core.begin()
    core.accept_action(action("s", request.request_id))

    core.update_actor_gate(False)
    feedback = core.complete(
        {
            "schema": "sharpa_execution_done.v1",
            "request_id": 0,
            "action_id": "a1",
            "revision": 0,
            "execute_start": 0,
            "execute_length": 2,
            "executed_steps": 1,
            "success": False,
            "done": True,
            "error": "machine execution gate closed",
        }
    )

    assert core.failed
    assert core.waiting_for_restart
    assert core.update_actor_gate(True)
    assert core.ready
    next_request = core.begin()
    assert next_request.request_id == 1
    assert next_request.feedback == feedback


def test_fast_stop_start_recovers_when_failed_done_arrives_late():
    core = PolicyClientCore("s")
    core.update_actor_gate(True)
    request = core.begin()
    core.accept_action(action("s", request.request_id))

    core.update_actor_gate(False)
    core.update_actor_gate(True)
    core.complete(
        {
            "schema": "sharpa_execution_done.v1",
            "request_id": 0,
            "action_id": "a1",
            "revision": 0,
            "execute_start": 0,
            "execute_length": 2,
            "executed_steps": 0,
            "success": False,
            "done": True,
            "error": "machine execution gate is not running",
        }
    )

    assert core.ready
    assert not core.waiting_for_restart


def test_policy_ignores_completion_from_replay_session():
    message = type("Done", (), {"session_id": "replay_sample_run"})()
    assert not completion_belongs_to_session(message, "ur-sharpa")
    message.session_id = "ur-sharpa"
    assert completion_belongs_to_session(message, "ur-sharpa")


def test_deployment_requires_both_devices_and_actor_source():
    assert deployment_ready(2, 1, 3)
    assert not deployment_ready(2, 1, 3, execution_paused=True)
    assert not deployment_ready(1, 1, 3)
    assert not deployment_ready(2, 0, 3)
    assert not deployment_ready(2, 1, 2)
    assert deployment_ready(0, 0, 3, execution_dry_run=True)
    assert not deployment_ready(
        0, 0, 3, execution_paused=True, execution_dry_run=True
    )


def test_metadata_summary_keeps_only_operator_fields():
    metadata = {
        "policy_family": "groot_n17",
        "task_id": "pick_place_push",
        "run_id": "run-1",
        "checkpoint_id": "checkpoint-5000",
        "metadata_format": {
            "format_id": "vision-state",
            "image": {
                "ego_cam": {"history_len": 2, "current": True},
                "left_wrist_cam": {"history_len": 0, "current": False},
                "right_wrist_cam": {"history_len": 0, "current": False},
            },
            "state": {"history_len": 18, "current": True},
            "sensor": {
                "tau": {"history_len": 0, "current": False},
                "wrench": {"history_len": 0, "current": False},
                "deformation": {"history_len": 0, "current": False},
            },
        },
        "checkpoint_path": "/very/long/path/not/for/dashboard",
    }

    result = summarize_metadata(metadata)

    assert result["format_id"] == "vision-state"
    assert result["requirements"] == [
        {"name": "ego_cam", "history": 2, "current": True},
        {"name": "state", "history": 18, "current": True},
    ]
    assert result["prompt"] == ""
    assert "checkpoint_path" not in result


def test_metadata_identity_must_match_selected_task_and_run():
    metadata = {
        "policy_family": "groot",
        "model_id": "groot_n17",
        "model_name": "groot_n17",
        "task_id": "ur_twist_tube",
        "run_id": "twist-run",
        "prompt": "twist the tube",
    }

    assert not metadata_identity_error(
        metadata,
        expected_model="groot_n17",
        expected_task_id="ur_twist_tube",
        expected_run_id="twist-run",
        expected_prompt="twist the tube",
    )
    error = metadata_identity_error(
        metadata,
        expected_model="pace",
        expected_task_id="ur_pick_place_push",
        expected_run_id="push-run",
    )

    assert "模型身份不匹配" in error
    assert "model groot_n17 != pace" in error
    assert "ur_twist_tube" in error
    assert "twist-run" in error


@pytest.mark.parametrize(
    "policy_family", ["groot", "groot_rtc", "pi05", "gcc", "pace", ""]
)
def test_every_model_uses_the_same_canonical_absolute_eef_contract(policy_family):
    validate_server_eef_contract(
        {
            "policy_family": policy_family,
            "interface": {
                "eef_contract": {
                    "schema": "sharpa62_direct_absolute_eef.v1",
                    "observation_eef_def": "absolute",
                    "action_eef_def": "absolute",
                    "eef_pose_layout": "xyz,r00,r10,r20,r01,r11,r21",
                    "server_xyz_direction_transform": "none",
                }
            },
        }
    )


def test_metadata_without_optional_eef_description_still_uses_public_contract():
    validate_server_eef_contract({"policy_family": "older_server"})


def test_inconsistent_direct_eef_contract_is_rejected():
    with pytest.raises(
        ValueError, match="server EEF contract is not the canonical absolute ABI"
    ):
        validate_server_eef_contract(
            {
                "policy_family": "groot_n17",
                "interface": {
                    "eef_contract": {
                        "schema": "sharpa62_direct_absolute_eef.v1",
                        "observation_eef_def": "absolute",
                        "action_eef_def": "absolute",
                        "eef_pose_layout": "xyz,r00,r01,r10,r11,r20,r21",
                        "server_xyz_direction_transform": "none",
                    }
                },
            }
        )
