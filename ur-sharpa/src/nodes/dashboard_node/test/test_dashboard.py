from pathlib import Path
from types import SimpleNamespace

import pytest

from dashboard_node.dashboard_node import (
    DASHBOARD_ACTIONS,
    DashboardState,
    actor_execution_active,
    action_visualization,
    forward_action,
    hand_motion_visualization,
    launch_live_replay,
    launch_replay_preview,
    replay_config_matches_status,
    stop_policy_deployment,
    validate_live_replay_command,
    validate_replay_command,
    wait_for_dashboard_state,
)


def test_dashboard_exposes_machine_and_model_controls():
    html = Path(__file__).parents[1] / "dashboard_node" / "dashboard.html"
    page = html.read_text(encoding="utf-8")

    for control_id in (
        "machineReset",
        "machineStart",
        "machinePause",
        "machineStop",
        "machineContinue",
        "machinePreview",
        "modelSelect",
        "modelServer",
        "modelStart",
        "modelStop",
        "sourcePolicy",
        "sourceReplay",
        "replayLoad",
        "replayClear",
        "replaySteps",
        "replayFrequency",
        "cameraEgoState",
        "cameraLeftState",
        "cameraRightState",
        "modelLogPanel",
        "modelLogOutput",
        "chunkLatencyCurrent",
        "chunkLatencyAverage",
    ):
        assert f'id="{control_id}"' in page
    assert DASHBOARD_ACTIONS == {
        "reset_both",
        "start_deploy",
        "continue_deploy",
        "preview_deploy",
        "pause_deploy",
        "stop_deploy",
        "select_policy",
        "select_replay",
    }
    assert "/api/model/start" in page
    assert "/api/model/stop" in page
    assert "/api/servers" in page
    assert "/api/server/select" in page
    assert 'onclick="startModel()"' in page
    assert "if(selectedTask&&selectedRun)await startSelectedModel()" in page
    assert "模型启动中" in page
    assert "模型加载中" in page
    assert "模型就绪" in page
    assert "model_identity_ok" in page
    assert "model.log_tail" in page
    assert "paintModelLog()" in page
    assert "latest.actor_latency" in page
    assert "current_latency_ms" in page
    assert "average_latency_ms" in page
    assert "#inferenceValue, #inferenceValue.ok, #inferenceValue.bad { color: var(--green); }" in page
    assert "action:'pause_deploy'" in page
    assert "continuing?'continue_deploy':'start_deploy'" in page
    assert "control.execution_paused" in page
    assert "control.execution_dry_run" in page
    assert "syncPredictionPlayback(actorRunning,paused)" in page
    assert 'aria-label="启动和停止"' in page
    assert 'aria-label="暂停和继续"' in page
    assert "continueSelectedExecution()" in page
    assert 'onclick="previewSelectedExecution()">预览</button>' in page
    assert ">预览轨迹</button>" not in page
    assert 'onclick="clearReplay()">卸载样本</button>' in page
    assert "async function stopReplayExecution()" in page
    assert "JSON.stringify({action:'stop'})" in page
    assert "JSON.stringify({action:'unload'})" in page
    assert "/api/replay" in page
    assert "/api/replay/live/start" in page
    assert "/api/replay/preview/start" in page
    assert "execution_steps:Number.parseInt(replaySteps.value,10)" in page
    assert "frequency_hz:Number(replayFrequency.value)" in page
    assert "history ${history} · current ${current}" in page
    assert "/media/yns/T9/0824_yns_unscrew_cap/success/2026-08-24_16-01-44.683076" in page

    dashboard_source = (
        Path(__file__).parents[1] / "dashboard_node" / "dashboard_node.py"
    ).read_text(encoding="utf-8")
    assert '"/actor_ik/latency_status"' in dashboard_source
    assert '"/actor_ik/trajectory"' in dashboard_source
    assert '"/policy/executable_action"' not in dashboard_source
    assert '"/replay/preview_action"' not in dashboard_source


def test_dashboard_rejects_any_other_machine_action_before_forwarding():
    with pytest.raises(ValueError, match="unsupported dashboard action"):
        forward_action("http://127.0.0.1:1/api/action", "start_record")


def test_replay_stop_and_unload_are_distinct_commands():
    assert validate_replay_command({"action": "stop"}) == {"action": "stop"}
    assert validate_replay_command({"action": "unload"}) == {"action": "unload"}


def test_policy_stop_closes_actor_gate_without_unloading_model():
    events = []

    class ModelManager:
        def stop(self):
            raise AssertionError("execution stop must keep the model loaded")

        def snapshot(self):
            return {"phase": "running"}

    result = stop_policy_deployment(
        "http://status/api/action",
        ModelManager(),
        forward=lambda _url, action: (
            events.append(action) or {"accepted": True, "ur_state": 0}
        ),
    )

    assert events == ["stop_deploy"]
    assert result["accepted"] is True
    assert result["model"]["phase"] == "running"
    assert result["model_unload_requested"] is False


def test_policy_stop_does_not_touch_model_if_actor_gate_rejects_stop():
    class ModelManager:
        def stop(self):
            raise AssertionError("model must not unload before Actor is stopped")

        def snapshot(self):
            raise AssertionError("rejected stop must not touch model state")

    with pytest.raises(Exception, match="not running"):
        stop_policy_deployment(
            "http://status/api/action",
            ModelManager(),
            forward=lambda _url, _action: {
                "accepted": False,
                "gate_reason": "actor is not running",
            },
        )


def test_actor_execution_active_covers_live_preview_and_paused_modes():
    assert actor_execution_active(
        {"control": {"ur_state": 2, "sharpa_state": 1}}
    )
    assert actor_execution_active(
        {"control": {"execution_dry_run": True}}
    )
    assert actor_execution_active(
        {"control": {"execution_paused": True}}
    )
    assert not actor_execution_active(
        {"control": {"ur_state": 0, "sharpa_state": 0}}
    )


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


def test_action_visualization_supports_replay_joint_trajectory():
    message = SimpleNamespace(
        session_id="replay",
        request_id=0,
        action_id="chunk",
        revision=0,
        timestamp_ns=1,
        frequency_hz=30.0,
        action_length=2,
        execute_start=0,
        execute_length=2,
        left_eef=[],
        right_eef=[],
        left_arm_joint=list(range(12)),
        right_arm_joint=list(range(20, 32)),
        hand_joint=list(range(88)),
        eef_frame="canonical",
    )

    result = action_visualization(message)

    assert result["frames"][0]["left_eef"] == []
    assert result["frames"][1]["left_arm_joint"] == [float(v) for v in range(6, 12)]
    assert result["frames"][1]["right_arm_joint"] == [float(v) for v in range(26, 32)]
    assert len(result["frames"][0]["left_hand"]) == 22


def test_replay_command_validation_keeps_execution_parameters():
    command = validate_replay_command(
        {
            "action": "load",
            "sample_dir": "/data/episode",
            "execution_steps": 40,
            "frequency_hz": 30,
            "joint_source": "measured",
            "loop": False,
            "dry_run": True,
        }
    )
    assert command["execution_steps"] == 40
    assert command["frequency_hz"] == 30.0
    with pytest.raises(ValueError, match="frequency_hz"):
        validate_replay_command({**command, "frequency_hz": 0})


def test_live_replay_validation_always_disables_dry_run():
    command = validate_live_replay_command(
        {
            "sample_dir": "/data/episode",
            "execution_steps": 40,
            "frequency_hz": 30,
            "joint_source": "measured",
            "loop": False,
            "dry_run": True,
        }
    )

    assert command["action"] == "load"
    assert command["dry_run"] is False


def test_live_replay_loaded_config_matches_without_resetting_cursor():
    command = {
        "sample_dir": "/data/episode",
        "execution_steps": 40,
        "frequency_hz": 30.0,
        "joint_source": "measured",
        "loop": False,
    }
    replay = {
        **command,
        "state": "paused",
        "dry_run": False,
        "next_row": 480,
    }

    assert replay_config_matches_status(replay, command)
    assert not replay_config_matches_status({**replay, "dry_run": True}, command)
    assert not replay_config_matches_status({**replay, "frequency_hz": 20}, command)


def test_wait_for_dashboard_state_observes_async_transition():
    state = DashboardState()
    state.update("replay", {"state": "ready"})

    result = wait_for_dashboard_state(
        state,
        lambda snapshot: (snapshot.get("replay") or {}).get("state") == "ready",
        timeout_s=0.1,
    )

    assert result["replay"]["state"] == "ready"


def test_live_replay_launch_waits_for_each_gate_before_starting():
    state = DashboardState()
    # A reset is optional: live Replay may open the gate from STOPPED and its
    # own alignment phase handles the move to the sample's first frame.
    state.update("control", {"ur_state": 0, "sharpa_state": 0})
    state.update(
        "replay",
        {
            "state": "ready",
            "execution_id": "old",
            "provider": "policy",
            "machine_running": False,
        },
    )
    events = []

    def publish(command):
        events.append(("replay", command["action"]))
        replay = dict(state.snapshot()["replay"])
        if command["action"] == "load":
            replay.update(
                state="ready",
                execution_id="new",
                dry_run=False,
            )
        elif command["action"] == "start":
            replay["state"] = "aligning"
        state.update("replay", replay)

    def forward(_url, action):
        events.append(("gate", action))
        replay = dict(state.snapshot()["replay"])
        if action == "select_replay":
            replay["provider"] = "replay"
        elif action == "start_deploy":
            replay["machine_running"] = True
        state.update("replay", replay)
        return {"accepted": True}

    result = launch_live_replay(
        state,
        "http://status/api/action",
        publish,
        {
            "action": "load",
            "sample_dir": "/data/episode",
            "execution_steps": 40,
            "frequency_hz": 30.0,
            "joint_source": "measured",
            "loop": False,
            "dry_run": False,
        },
        forward=forward,
    )

    assert result["execution_id"] == "new"
    assert events == [
        ("replay", "load"),
        ("gate", "select_replay"),
        ("gate", "start_deploy"),
        ("replay", "start"),
    ]


def test_dashboard_start_button_does_not_require_reset_state():
    html = Path(__file__).parents[1] / "dashboard_node" / "dashboard.html"
    page = html.read_text(encoding="utf-8")

    assert "机器复位可选，也可直接从当前位置启动" in page
    assert "control.ur_state!==1||!hardwareReady||!sourceReady" not in page


def test_live_replay_continue_reuses_loaded_execution_and_cursor():
    state = DashboardState()
    state.update("control", {"ur_state": 1, "sharpa_state": 0})
    state.update(
        "replay",
        {
            "state": "paused",
            "execution_id": "same-execution",
            "sample_dir": "/data/episode",
            "execution_steps": 40,
            "frequency_hz": 30.0,
            "joint_source": "measured",
            "loop": False,
            "dry_run": False,
            "next_row": 480,
            "provider": "policy",
            "machine_running": False,
        },
    )
    events = []

    def publish(command):
        events.append(("replay", command["action"]))
        replay = dict(state.snapshot()["replay"])
        assert replay["next_row"] == 480
        replay["state"] = "playing"
        state.update("replay", replay)

    def forward(_url, action):
        events.append(("gate", action))
        replay = dict(state.snapshot()["replay"])
        if action == "select_replay":
            replay["provider"] = "replay"
        elif action == "start_deploy":
            replay["machine_running"] = True
        state.update("replay", replay)
        return {"accepted": True}

    result = launch_live_replay(
        state,
        "http://status/api/action",
        publish,
        {
            "action": "load",
            "sample_dir": "/data/episode",
            "execution_steps": 40,
            "frequency_hz": 30.0,
            "joint_source": "measured",
            "loop": False,
            "dry_run": False,
        },
        forward=forward,
    )

    assert result["execution_id"] == "same-execution"
    assert events == [
        ("gate", "select_replay"),
        ("gate", "start_deploy"),
        ("replay", "start"),
    ]


def test_replay_preview_loads_dry_run_and_waits_for_playback():
    state = DashboardState()
    state.update("control", {"ur_state": 0, "sharpa_state": 0})
    state.update("replay", {"state": "idle", "execution_id": "old"})
    events = []

    def publish(command):
        events.append(("replay", command["action"]))
        replay = dict(state.snapshot()["replay"])
        if command["action"] == "load":
            replay.update(
                state="ready",
                execution_id="preview-execution",
                dry_run=True,
            )
        else:
            replay.update(state="playing", preview_active=True)
        state.update("replay", replay)

    def forward(_url, action):
        events.append(("gate", action))
        replay = dict(state.snapshot()["replay"])
        if action == "select_replay":
            replay["provider"] = "replay"
        elif action == "preview_deploy":
            replay["machine_running"] = True
        state.update("replay", replay)
        return {"accepted": True}

    result = launch_replay_preview(
        state,
        "http://status/api/action",
        publish,
        {
            "action": "load",
            "sample_dir": "/data/episode",
            "execution_steps": 40,
            "frequency_hz": 30.0,
            "joint_source": "measured",
            "loop": False,
            "dry_run": True,
        },
        forward=forward,
    )

    assert result["execution_id"] == "preview-execution"
    assert events == [
        ("replay", "load"),
        ("gate", "select_replay"),
        ("gate", "preview_deploy"),
        ("replay", "start"),
    ]


def test_completed_replay_requires_stop_to_reset_progress_before_preview():
    state = DashboardState()
    state.update("control", {"ur_state": 0, "sharpa_state": 0})
    state.update(
        "replay",
        {
            "state": "complete",
            "execution_id": "complete-execution",
            "sample_dir": "/data/episode",
        },
    )

    with pytest.raises(Exception, match="点击停止重置执行进度"):
        launch_replay_preview(
            state,
            "http://status/api/action",
            lambda _command: None,
            {
                "action": "load",
                "sample_dir": "/data/episode",
                "execution_steps": 40,
                "frequency_hz": 30.0,
                "joint_source": "measured",
                "loop": False,
                "dry_run": True,
            },
        )


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
    assert "plotPoint=p=>[p[0],p[2],p[1]]" in page
    assert "sceneCenter=[0,.28,.35],sceneSpan=.9,visualZoom=1.14" in page
    assert "gridStep=sceneSpan/8,spaceHeight=.64" in page
    assert "ctx.lineWidth=1.55" in page
    assert "'#35d4c3'" in page
    assert "'#ff9a6c'" in page
    assert "ORIGINAL VIEW" in page
    assert "handVisualScale" not in page
    assert "canonicalScreenX" not in page
    assert "drawWristAxes" not in page
    assert "mins=[0,1,2]" not in page


def test_camera_subscribers_match_best_effort_publisher_qos():
    nodes = Path(__file__).parents[2]
    dashboard = (nodes / "dashboard_node" / "dashboard_node" / "dashboard_node.py").read_text(encoding="utf-8")
    policy_client = (nodes / "policy_client_node" / "policy_client_node" / "policy_client_node.py").read_text(encoding="utf-8")

    for source in (dashboard, policy_client):
        assert "reliability=ReliabilityPolicy.BEST_EFFORT" in source
        assert "CameraSet," in source
        assert '"/camera/set",' in source
        assert "self._camera," in source
        assert "camera_qos" in source

    assert '"/robot_states/status"' not in dashboard
    assert 'payload.get("buffers")' in dashboard


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
