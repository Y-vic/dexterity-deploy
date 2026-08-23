import numpy as np
import pytest

from policy_client_node.policy_client_node import (
    LegacyServerPoseAdapter,
    PolicyClientCore,
    deployment_ready,
    metadata_identity_error,
    summarize_metadata,
)
from ur_common import geometry as g
from ur_common.geometry import UrSharpAWireGeometry


def action(session, request):
    eef = np.zeros((2, 9), dtype=np.float32); eef[:, 3] = 1; eef[:, 7] = 1
    return {
        "schema": "sharpa_policy_action.v4", "session_id": session,
        "request_id": request, "action_id": "a1", "revision": 0,
        "timestamp_ns": 1,
        "execution": {"frequency_hz": 30.0, "action_length": 2, "execute_start": 0, "execute_length": 2},
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


def test_deployment_requires_both_devices_and_actor_source():
    assert deployment_ready(2, 1, 3)
    assert not deployment_ready(1, 1, 3)
    assert not deployment_ready(2, 0, 3)
    assert not deployment_ready(2, 1, 2)


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
        "task_id": "ur_twist_tube",
        "run_id": "twist-run",
        "prompt": "twist the tube",
    }

    assert not metadata_identity_error(
        metadata,
        expected_task_id="ur_twist_tube",
        expected_run_id="twist-run",
        expected_prompt="twist the tube",
    )
    error = metadata_identity_error(
        metadata,
        expected_task_id="ur_pick_place_push",
        expected_run_id="push-run",
    )

    assert "模型身份不匹配" in error
    assert "ur_twist_tube" in error
    assert "twist-run" in error


def canonical_pose() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = g.rotvec_to_matrix((0.3, -0.4, 0.2))
    transform[:3, 3] = (0.25, -0.15, 0.6)
    return g.transform_to_column_pose9(transform)


def test_legacy_server_adapter_keeps_ros_observation_canonical():
    geometry = UrSharpAWireGeometry()
    adapter = LegacyServerPoseAdapter(geometry)
    pose = canonical_pose()
    payload = {
        "state": {
            "history": {
                "left_wrist": {"joint": None, "eef": pose[None], "eef_def": "absolute"},
                "right_wrist": {"joint": None, "eef": pose[None], "eef_def": "absolute"},
            },
            "current": {
                "left_wrist": {"joint": None, "eef": pose, "eef_def": "absolute"},
                "right_wrist": {"joint": None, "eef": pose, "eef_def": "absolute"},
            },
        }
    }

    server_payload = adapter.observation_to_server(payload)

    assert np.array_equal(payload["state"]["current"]["left_wrist"]["eef"], pose)
    for side in ("left", "right"):
        server_pose = server_payload["state"]["current"][f"{side}_wrist"]["eef"]
        assert not np.allclose(server_pose, pose)
        assert np.allclose(
            geometry.legacy_server_pose_to_canonical_pose(server_pose, side),
            pose,
            atol=1.0e-5,
        )


def test_legacy_server_action_is_restored_to_canonical():
    geometry = UrSharpAWireGeometry()
    adapter = LegacyServerPoseAdapter(geometry)
    pose = canonical_pose()
    command = {
        "action": {
            f"{side}_wrist": {
                "joint": None,
                "eef": geometry.canonical_pose_to_legacy_server_pose(pose, side)[None],
                "eef_def": "absolute",
            }
            for side in ("left", "right")
        }
    }
    command["action"]["hand_joint"] = {"left": None, "right": None}

    canonical = adapter.action_to_canonical(command)

    for side in ("left", "right"):
        restored = canonical["action"][f"{side}_wrist"]["eef"]
        assert restored.shape == (1, 9)
        assert np.allclose(restored[0], pose, atol=1.0e-5)
