from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from dashboard_node.dashboard_node import (
    DASHBOARD_ACTIONS,
    OPERATOR_ACTIONS,
    DashboardState,
    actor_execution_active,
    action_visualization,
    forward_action,
    hand_motion_visualization,
    launch_live_replay,
    load_replay_and_wait,
    launch_replay_preview,
    replay_config_matches_status,
    stop_policy_deployment,
    prepare_workbench_shutdown,
    validate_live_replay_command,
    validate_replay_command,
    wait_for_dashboard_state,
)


def test_dashboard_forwards_selected_reset_target(monkeypatch):
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return b'{"accepted":true}'

    def fake_open(request, timeout):
        sent.append((json.loads(request.data), timeout))
        return Response()

    monkeypatch.setattr("dashboard_node.dashboard_node.urlopen", fake_open)
    target = {"mode": "episode", "root": "/media/yns/My Passport/data", "episode": "success/one"}
    assert forward_action("http://127.0.0.1:8080/api/action", "reset_both",
                          timeout_s=15, reset_target=target)["accepted"]
    assert sent == [({"action": "reset_both", "reset_target": target}, 15)]
    assert forward_action("http://127.0.0.1:8080/api/action", "start_teleop_record",
                          allowed_actions=OPERATOR_ACTIONS)["accepted"]
    assert sent[-1][0] == {"action": "start_teleop_record"}


def test_dashboard_exposes_workbench_machine_and_remote_model_controls():
    from html.parser import HTMLParser
    from ur_common.workbench import ASSETS

    class Controls(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()
            self.modes = []
            self.scripts = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs:
                self.ids.add(attrs['id'])
            if 'data-mode' in attrs:
                self.modes.append(attrs['data-mode'])
            if tag == 'script' and 'src' in attrs:
                self.scripts.append(attrs['src'])

    page = Controls()
    page.feed((ASSETS / 'index.html').read_text(encoding='utf-8'))
    assert page.modes == ['teleop', 'policy', 'replay']
    assert {'reset', 'resetMode', 'start', 'pause', 'resume', 'stop', 'killed',
            'startRecord', 'record', 'finish', 'serverSelect', 'chooseModel',
            'unloadModel', 'modelDialog', 'modelLog', 'policyRecord',
            'loadReplay', 'unloadReplay', 'replaySteps', 'replayHz',
            'replayScope', 'referenceRoot', 'referenceEpisode'} <= page.ids
    assert not {'rtcEnabled', 'executionHorizon', 'syncSkipSteps', 'rltControls'} & page.ids
    for script in page.scripts:
        assert (ASSETS / script.removeprefix('/')).is_file()
    assert DASHBOARD_ACTIONS == {
        'reset_both', 'start_deploy', 'start_hands_deploy', 'continue_deploy',
        'preview_deploy', 'preview_hands_deploy', 'pause_deploy', 'stop_deploy',
        'select_policy', 'select_replay', 'select_hil',
        'enable_policy_recording', 'disable_policy_recording',
    }


def test_dashboard_rejects_any_other_machine_action_before_forwarding():
    with pytest.raises(ValueError, match="unsupported dashboard action"):
        forward_action("http://127.0.0.1:1/api/action", "start_record")


def test_shutdown_waits_for_recording_before_unloading_model(monkeypatch):
    import time
    state = DashboardState()
    state.update('recording', {'recording': True})
    events = []
    model = {'phase': 'running'}
    manager = SimpleNamespace(stop=lambda: (events.append('unload'), model.update(phase='stopped')),
                              snapshot=lambda: model)
    def forward(url, action, **kwargs):
        assert action == 'stop' and kwargs['allowed_actions'] == OPERATOR_ACTIONS
        events.append('stop')
        state.update('recording', {'saving': True, 'received_at': time.time() + 1})
        return {'accepted': True}
    def finish(_):
        events.append('saved')
        state.update('recording', {'recording': False, 'saving': False,
                                   'received_at': time.time() + 1})
    monkeypatch.setattr('dashboard_node.dashboard_node.time.sleep', finish)
    prepare_workbench_shutdown(state, 'status', manager,
                               lambda command: events.append(command['action'] + '_replay'),
                               forward=forward)
    assert events == ['stop', 'stop_replay', 'saved', 'unload']


def test_shutdown_preserves_ros_when_recording_has_not_finished():
    state = DashboardState()
    state.update('recording', {'saving': True})
    manager = SimpleNamespace(stop=lambda: pytest.fail('must finish saving first'))
    with pytest.raises(RuntimeError, match='录制尚未保存完成'):
        prepare_workbench_shutdown(state, 'status', manager, lambda _: None,
                                   forward=lambda *args, **kwargs: {'accepted': True},
                                   timeout_s=0)


def test_shutdown_can_close_disconnected_idle_controller_but_reports_model_failure():
    state = DashboardState()
    def disconnected(*args, **kwargs):
        raise RuntimeError('status offline')
    manager = SimpleNamespace(stop=lambda: None, snapshot=lambda: {'phase': 'stopped'})
    prepare_workbench_shutdown(state, 'status', manager, lambda _: None, forward=disconnected)
    manager.snapshot = lambda: {'phase': 'fault', 'detail': '模型清理失败'}
    with pytest.raises(RuntimeError, match='模型清理失败'):
        prepare_workbench_shutdown(state, 'status', manager, lambda _: None, forward=disconnected)
    state.update('recording', {'recording': True})
    with pytest.raises(RuntimeError, match='status offline'):
        prepare_workbench_shutdown(state, 'status', manager, lambda _: None, forward=disconnected)


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
            "execution_scope": "hands_only",
            "loop": False,
            "dry_run": True,
        }
    )
    assert command["execution_steps"] == 40
    assert command["frequency_hz"] == 30.0
    assert command["execution_scope"] == "hands_only"
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


def test_replay_validation_rejects_unknown_execution_scope():
    with pytest.raises(ValueError, match="execution_scope"):
        validate_replay_command(
            {
                "action": "load",
                "sample_dir": "/data/episode",
                "execution_steps": 40,
                "frequency_hz": 30,
                "joint_source": "measured",
                "execution_scope": "arms_only",
                "loop": False,
                "dry_run": False,
            }
        )


def test_actor_execution_active_accepts_hands_only_gate_without_ur():
    assert actor_execution_active(
        {
            "control": {
                "ur_state": 0,
                "sharpa_state": 1,
                "execution_scope": "hands_only",
            }
        }
    )


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
    # Replay must delegate hardware lifecycle validation to the shared machine
    # gate. A completed reset leaves UR/SharpA in HOMED, which is a valid and
    # direct start state.
    state.update("control", {"ur_state": 1, "sharpa_state": 3})
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
            replay.update(command)
            replay["command_result"] = {"id": command["id"], "accepted": True}
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


def test_hands_only_replay_opens_only_the_hands_gate():
    state = DashboardState()
    state.update("control", {"ur_state": 0, "sharpa_state": 0})
    state.update(
        "replay",
        {
            "state": "ready",
            "execution_id": "hands-execution",
            "sample_dir": "/data/episode",
            "execution_steps": 40,
            "frequency_hz": 30.0,
            "joint_source": "measured",
            "execution_scope": "hands_only",
            "loop": False,
            "dry_run": False,
            "provider": "policy",
            "machine_running": False,
        },
    )
    events = []

    def publish(command):
        events.append(("replay", command["action"]))
        replay = dict(state.snapshot()["replay"])
        replay["state"] = "aligning"
        state.update("replay", replay)

    def forward(_url, action):
        events.append(("gate", action))
        replay = dict(state.snapshot()["replay"])
        if action == "select_replay":
            replay["provider"] = "replay"
        elif action == "start_hands_deploy":
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
            "execution_scope": "hands_only",
            "loop": False,
            "dry_run": False,
        },
        forward=forward,
    )

    assert result["execution_id"] == "hands-execution"
    assert events == [
        ("gate", "select_replay"),
        ("gate", "start_hands_deploy"),
        ("replay", "start"),
    ]


def test_live_replay_continue_reuses_loaded_execution_and_cursor():
    state = DashboardState()
    state.update("control", {"ur_state": 1, "sharpa_state": 3})
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
            replay.update(command)
            replay["command_result"] = {"id": command["id"], "accepted": True}
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


def test_dashboard_camera_subscription_matches_qos_and_policy_uses_raw_frames():
    import ast
    nodes = Path(__file__).parents[2]
    dashboard = (nodes / 'dashboard_node/dashboard_node/dashboard_node.py').read_text(encoding='utf-8')
    policy = (nodes / 'policy_client_node/policy_client_node/policy_client_node.py').read_text(encoding='utf-8')

    def camera_subscriptions(source):
        return [node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'create_subscription' and len(node.args) >= 4
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == '/camera/set']

    subscriptions = camera_subscriptions(dashboard)
    assert len(subscriptions) == 1
    assert isinstance(subscriptions[0].args[3], ast.Name)
    assert subscriptions[0].args[3].id == 'camera_qos'
    qos = next(node.value for node in ast.walk(ast.parse(dashboard))
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'camera_qos' for target in node.targets))
    reliability = next(item.value for item in qos.keywords if item.arg == 'reliability')
    assert isinstance(reliability, ast.Attribute) and reliability.attr == 'BEST_EFFORT'
    # The model must not silently fall back to the slower UI JPEG stream.
    assert camera_subscriptions(policy) == []
    assert 'SharedCameraReader' in policy and 'self._raw_cameras' in policy
    # Native-image encoding, freshness and callback behavior have behavioral
    # coverage in test_remote_observation.py and test_remote_callbacks.py.


def test_remote_model_picker_explains_unavailable_entries():
    from ur_common.workbench import ASSETS
    script = (ASSETS / 'workbench.js').read_text(encoding='utf-8')
    assert 'run.disabled_reason' in script
    assert "e.dataset.launchable !== 'true'" in script
    assert '/api/servers' in script and '/api/server/select' in script
    assert '尚未配置模型目录' in script
    # Actual selection, rejection, empty catalogs and load requests execute
    # against an offline DOM/network fixture in workbench_remote_ui_test.cjs.


@pytest.mark.parametrize("launch", [launch_live_replay, launch_replay_preview])
def test_replay_load_failure_reports_error_without_new_execution_id(launch):
    state = DashboardState()
    state.update("replay", {"state": "ready", "execution_id": "existing"})
    def publish(command):
        state.update("replay", {
            "state": "fault", "execution_id": "existing",
            "command_result": {"id": command["id"], "accepted": False,
                            "reason": "sample checksum mismatch"},
        })
    def forward(*args):
        pytest.fail("load failure must not open a machine gate")
    with pytest.raises(RuntimeError, match="sample checksum mismatch"):
        launch(state, "unused", publish, {"action": "load"}, forward=forward)


def test_replay_load_ignores_stale_fault_until_matching_reply():
    command = dict(action="load", sample_dir="/sample", execution_steps=40,
                   frequency_hz=30., joint_source="measured", execution_scope="full",
                   dry_run=True, loop=False)
    class Replies:
        calls = 0
        request_id = None
        def snapshot(self):
            self.calls += 1
            if self.calls < 3:
                return {"replay": {"state": "fault", "error": "previous failure",
                    "command_result": {"id": "previous", "accepted": False}}}
            return {"replay": {**command, "state": "ready", "execution_id": "new",
                "command_result": {"id": self.request_id, "accepted": True}}}
    state = Replies()
    def publish(payload):
        state.request_id = payload["id"]
    assert load_replay_and_wait(state, publish, command)["replay"]["execution_id"] == "new"
