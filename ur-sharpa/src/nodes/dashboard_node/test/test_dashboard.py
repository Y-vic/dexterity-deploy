from pathlib import Path
from types import SimpleNamespace

import pytest

from dashboard_node.dashboard_node import (
    DASHBOARD_ACTIONS,
    action_visualization,
    forward_action,
    hand_motion_visualization,
)


def test_dashboard_exposes_machine_and_model_controls():
    html = Path(__file__).parents[1] / "dashboard_node" / "dashboard.html"
    page = html.read_text(encoding="utf-8")

    for control_id in (
        "machineReset",
        "machineStart",
        "machineStop",
        "modelSelect",
        "modelStart",
        "modelStop",
    ):
        assert f'id="{control_id}"' in page
    assert DASHBOARD_ACTIONS == {"reset_both", "start_deploy", "stop_deploy"}
    assert "/api/model/start" in page
    assert "/api/model/stop" in page
    assert 'onclick="startModel()"' in page
    assert "if(selectedTask&&selectedRun)await startSelectedModel()" in page
    assert "模型启动中" in page
    assert "模型加载中" in page
    assert "模型就绪" in page
    assert "model_identity_ok" in page


def test_dashboard_rejects_any_other_machine_action_before_forwarding():
    with pytest.raises(ValueError, match="unsupported dashboard action"):
        forward_action("http://127.0.0.1:1/api/action", "start_record")


def test_action_visualization_preserves_every_executable_step():
    message = SimpleNamespace(
        session_id="s",
        request_id=8,
        action_id="a",
        revision=2,
        timestamp_ns=10,
        frequency_hz=10.0,
        action_length=6,
        execute_start=4,
        execute_length=2,
        left_eef=list(range(18)),
        right_eef=list(range(100, 118)),
        hand_joint=list(range(88)),
        eef_frame="canonical",
    )

    result = action_visualization(message)

    assert result["revision"] == 2
    assert result["frequency_hz"] == 10.0
    assert [frame["step"] for frame in result["frames"]] == [4, 5]
    assert result["frames"][0]["left_eef"] == [float(value) for value in range(9)]
    assert result["frames"][1]["right_eef"] == [float(value) for value in range(109, 118)]
    assert result["frames"][1]["left_hand"] == [float(value) for value in range(44, 66)]
    assert result["frames"][1]["right_hand"] == [float(value) for value in range(66, 88)]


def test_action_visualization_rejects_partial_channels_for_display():
    message = SimpleNamespace(
        session_id="s",
        request_id=0,
        action_id="a",
        revision=0,
        timestamp_ns=1,
        frequency_hz=10.0,
        action_length=2,
        execute_start=0,
        execute_length=2,
        left_eef=[0.0] * 9,
        right_eef=[],
        hand_joint=[0.0] * 44,
        eef_frame="canonical",
    )

    result = action_visualization(message)

    assert len(result["frames"]) == 2
    assert result["frames"][0]["left_eef"] == []
    assert result["frames"][0]["left_hand"] == []
    assert result["frames"][0]["right_hand"] == []


def test_hand_motion_visualization_uses_canonical_wrist_pose():
    local = {
        "left": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 19,
        "right": [[0.0, 0.0, 0.0]] * 21,
    }
    pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    result = hand_motion_visualization(
        pose,
        pose,
        [0.0] * 22,
        [0.0] * 22,
        keypoint_builder=lambda *_: local,
    )

    assert result["frame"] == "canonical"
    assert result["kinematics"] == "sharpa_urdf"
    assert result["left_keypoints"][0] == [1.0, 2.0, 3.0]
    assert result["left_keypoints"][1] == [1.1, 2.0, 3.0]


def test_dashboard_uses_urdf_keypoints_instead_of_approximate_hand_geometry():
    html = Path(__file__).parents[1] / "dashboard_node" / "dashboard.html"
    page = html.read_text(encoding="utf-8")

    assert "left_keypoints" in page
    assert "right_keypoints" in page
    assert "function handPoints" not in page
    assert "sceneSpan=.9" in page
    assert "mins=[0,1,2]" not in page


def test_model_picker_only_shows_launchable_runs():
    html = Path(__file__).parents[1] / "dashboard_node" / "dashboard.html"
    page = html.read_text(encoding="utf-8")

    assert "只显示配置完整并且可以直接部署" in page
    assert ".filter(run=>run.launchable)" in page
    assert "'READY':'INVALID'" not in page
    assert "runStatusLabels" not in page
    assert "defaultBadge" not in page


def test_dashboard_plays_prediction_chunk_without_measured_history():
    html = Path(__file__).parents[1] / "dashboard_node" / "dashboard.html"
    page = html.read_text(encoding="utf-8")

    assert "requestAnimationFrame(animatePrediction)" in page
    assert "prediction.frames" in page
    assert "measured.history" not in page
