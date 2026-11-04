import time
from status_node.gates import (
    ExecutionScope,
    GateMachine,
    RecordState,
    SharpaState,
    UrState,
)
from status_node.status_node import resolve_reset_target

def fresh(m, now):
    m.update_cameras({"ego":int(now*1e9),"left_wrist":int(now*1e9),"right_wrist":int(now*1e9)}, {"ego":True,"left_wrist":True,"right_wrist":True}, now)

def hardware_ready(m):
    m.update_hardware_readiness(ur_ready=True, sharpa_ready=True)


def test_reset_target_selection_preserves_bias_and_resolves_episode(monkeypatch):
    assert resolve_reset_target({"action": "reset_both"}) is None
    assert resolve_reset_target({"reset_target": {"mode": "bias"}}) is None
    calls = []
    monkeypatch.setattr(
        "status_node.status_node.episode_initial_target",
        lambda root, episode: calls.append((root, episode)) or {"source": "episode"},
    )
    target = resolve_reset_target({"reset_target": {
        "mode": "episode", "root": "/media/yns/My Passport/data", "episode": "success/one",
    }})
    assert target == {"source": "episode"}
    assert calls == [("/media/yns/My Passport/data", "success/one")]

def bias_status(m, phase, **extra):
    payload = {"phase": phase, "action_id": m.bias_action_id, **extra}
    return m.on_bias_status(payload)

def finish_start(m):
    outcome = m.on_execution_start_readiness(position_ready=True)
    assert outcome and outcome.hardware_stop is False

def test_all_legal_transitions_and_side_effects():
    m=GateMachine()
    hardware_ready(m)
    assert not m.attempt_run_ur(); assert "reset both" in m.gate_reason
    reset = m.attempt_reset_both()
    assert reset.bias_request == "home_both" and reset.hardware_stop is False
    assert m.ur_state == UrState.HOMING and m.sharpa_state == SharpaState.HOMING
    bias_status(m, "done")
    assert m.ur_state == UrState.HOMED and m.sharpa_state == SharpaState.HOMED
    started = m.attempt_run_ur()
    assert started.hardware_stop is False
    assert m.ur_state == UrState.RUNNING and m.sharpa_state == SharpaState.RUNNING
    now=time.time(); fresh(m,now)
    assert m.attempt_start_recording().record_command == "start"
    assert m.record_state == RecordState.RECORDING
    assert m.attempt_stop_recording().record_command == "stop"

def test_recording_does_not_require_running_machines_or_vive():
    m=GateMachine()
    now=time.time(); fresh(m,now)
    outcome=m.attempt_start_recording(now=now)
    assert outcome.record_command == "start"
    assert m.ur_state == UrState.STOPPED
    assert m.sharpa_state == SharpaState.STOPPED


def test_recording_still_requires_progressing_cameras():
    m=GateMachine()
    assert not m.attempt_start_recording()
    assert "camera" in m.gate_reason

def test_deploy_reset_start_stop_cycle():
    m=GateMachine(mode="deploy")
    hardware_ready(m)
    m.ur_state=UrState.RUNNING; m.sharpa_state=SharpaState.RUNNING
    m.machine_ready=True
    m.record_state=RecordState.RECORDING; m.session_id="episode_test"

    assert not m.attempt_reset_both()
    paused = m.attempt_pause()
    assert paused.hardware_stop is True and paused.record_command == "stop"
    reset=m.attempt_reset_both()
    assert reset.bias_request == "home_both"
    assert reset.record_command == ""
    assert reset.hardware_stop is False
    assert m.ur_state == UrState.HOMING
    assert m.sharpa_state == SharpaState.HOMING

    bias_status(m, "done")
    assert m.ur_state == UrState.HOMED
    assert m.sharpa_state == SharpaState.HOMED
    assert m.attempt_start_deploy()
    assert m.ur_state == UrState.RUNNING and m.sharpa_state == SharpaState.RUNNING

    stopped=m.attempt_stop_deploy()
    assert stopped.hardware_stop is True
    assert m.ur_state == UrState.STOPPED
    assert m.sharpa_state == SharpaState.STOPPED


def test_ur_home_actions_release_hardware_stop_latch():
    machine = GateMachine()
    hardware_ready(machine)
    assert machine.attempt_start_ur().hardware_stop is False
    assert not machine.attempt_reset_ur()
    bias_status(machine, "done")
    assert machine.attempt_reset_ur().hardware_stop is False

def test_start_deploy_requires_deploy_mode_and_hardware_but_not_reset():
    teleop=GateMachine(mode="teleop")
    teleop.ur_state=UrState.HOMED; teleop.sharpa_state=SharpaState.RUNNING
    assert not teleop.attempt_start_deploy()
    assert "expected deploy" in teleop.gate_reason

    deploy=GateMachine(mode="deploy")
    assert not deploy.attempt_start_deploy()
    assert "UR is not in Normal mode" in deploy.gate_reason
    hardware_ready(deploy)
    assert deploy.attempt_start_deploy().hardware_stop is False
    assert deploy.ur_state == UrState.STARTING
    assert deploy.sharpa_state == SharpaState.STARTING
    finish_start(deploy)
    assert deploy.ur_state == UrState.RUNNING
    assert deploy.sharpa_state == SharpaState.RUNNING
    assert deploy.machine_ready is False


def test_hands_only_deploy_does_not_require_or_open_ur_gate():
    machine = GateMachine(mode="deploy")
    machine.update_hardware_readiness(ur_ready=False, sharpa_ready=True)

    started = machine.attempt_start_hands_deploy()

    assert started.hardware_stop is False
    assert machine.execution_scope == ExecutionScope.HANDS_ONLY
    assert machine.ur_state == UrState.STOPPED
    assert machine.sharpa_state == SharpaState.STARTING
    finish_start(machine)
    assert machine.ur_state == UrState.STOPPED
    assert machine.sharpa_state == SharpaState.RUNNING
    assert machine.status_payload()["machine_running"] is True
    assert machine.status_payload()["execution_scope"] == "hands_only"


def test_hands_only_preview_requires_sharpa_but_not_ur():
    machine = GateMachine(mode="deploy")
    machine.update_hardware_readiness(ur_ready=False, sharpa_ready=True)

    assert machine.attempt_preview_hands_deploy()
    assert machine.execution_dry_run is True
    assert machine.execution_scope == ExecutionScope.HANDS_ONLY
    assert machine.ur_state == UrState.STOPPED
    assert machine.sharpa_state == SharpaState.STOPPED


def test_deploy_started_at_current_pose_can_pause_and_continue_without_reset():
    machine = GateMachine(mode="deploy")
    hardware_ready(machine)

    assert machine.attempt_start_deploy()
    finish_start(machine)
    paused = machine.attempt_pause_deploy()
    assert paused.hardware_stop is None
    assert machine.execution_paused is True
    assert machine.ur_state == UrState.RUNNING
    assert machine.sharpa_state == SharpaState.RUNNING
    assert not machine.attempt_start_deploy()
    assert "use continue" in machine.gate_reason
    continued = machine.attempt_continue_deploy()
    assert continued.hardware_stop is None
    assert machine.execution_paused is False
    assert machine.ur_state == UrState.RUNNING
    assert machine.sharpa_state == SharpaState.RUNNING


def test_stop_deploy_aborts_a_soft_paused_policy():
    machine = GateMachine(mode="deploy")
    hardware_ready(machine)
    assert machine.attempt_start_deploy()
    finish_start(machine)
    assert machine.attempt_pause_deploy()

    stopped = machine.attempt_stop_deploy()

    assert stopped.hardware_stop is True
    assert machine.execution_paused is False
    assert machine.ur_state == UrState.STOPPED
    assert machine.sharpa_state == SharpaState.STOPPED


def test_preview_deploy_keeps_hardware_stopped_and_resumes_same_mode():
    machine = GateMachine(mode="deploy")
    hardware_ready(machine)

    started = machine.attempt_preview_deploy()

    assert started.hardware_stop is None
    assert machine.execution_dry_run is True
    assert machine.ur_state == UrState.STOPPED
    assert machine.sharpa_state == SharpaState.STOPPED
    assert not machine.attempt_start_deploy()
    assert "stop trajectory preview" in machine.gate_reason

    assert machine.attempt_pause_deploy()
    assert machine.execution_paused is True
    assert not machine.attempt_start_deploy()
    assert machine.attempt_continue_deploy()
    assert machine.execution_paused is False
    assert machine.execution_dry_run is True

    stopped = machine.attempt_stop_deploy()
    assert stopped.hardware_stop is None
    assert machine.execution_dry_run is False
    assert machine.execution_paused is False


def test_preview_blocks_provider_switch_until_stopped():
    machine = GateMachine(mode="deploy")
    hardware_ready(machine)
    assert machine.attempt_preview_deploy()
    assert not machine.attempt_set_execution_provider("replay")
    assert machine.execution_provider == "policy"


def test_deploy_provider_switch_is_explicit_and_stopped_only():
    machine = GateMachine(mode="deploy")
    assert machine.execution_provider == "policy"
    assert machine.attempt_set_execution_provider("replay")
    assert machine.status_payload()["execution_provider"] == "replay"

    machine.ur_state = UrState.RUNNING
    machine.sharpa_state = SharpaState.RUNNING
    assert not machine.attempt_set_execution_provider("policy")
    assert "stop the machine" in machine.gate_reason
    assert machine.execution_provider == "replay"


def test_provider_switch_is_not_available_in_teleop():
    machine = GateMachine(mode="teleop")
    assert not machine.attempt_set_execution_provider("replay")
    assert machine.execution_provider == "policy"

def test_machine_start_pause_is_a_single_toggle():
    m=GateMachine()
    hardware_ready(m)
    assert m.attempt_reset_both()
    bias_status(m, "done")
    assert m.status_payload()["machine_ready"] is True
    assert m.attempt_run_ur().hardware_stop is False
    assert not m.attempt_reset_both()
    paused=m.attempt_pause()
    assert paused.hardware_stop is True
    assert m.ur_state == UrState.STOPPED and m.sharpa_state == SharpaState.STOPPED
    assert not m.attempt_run_ur()

def test_machine_cannot_reset_or_start_when_hardware_is_not_ready():
    m=GateMachine()
    assert not m.attempt_reset_both()
    assert "UR is not in Normal mode" in m.gate_reason
    m.update_hardware_readiness(ur_ready=True, sharpa_ready=True)
    assert m.attempt_reset_both()
    bias_status(m, "done")
    m.update_hardware_readiness(ur_ready=False)
    assert not m.attempt_run_ur()
    assert "UR is not in Normal mode" in m.gate_reason

def test_stale_bias_result_cannot_complete_or_fail_current_reset():
    m = GateMachine()
    hardware_ready(m)
    assert m.attempt_reset_both()
    current = m.bias_action_id
    m.on_bias_status({"phase": "failed", "action_id": "older-reset", "detail": "stale"})
    assert m.bias_action_id == current
    assert m.ur_state == UrState.HOMING
    assert m.sharpa_state == SharpaState.HOMING
    outcome = bias_status(m, "failed", detail="SharpA NACK")
    assert outcome.hardware_stop is True
    assert m.ur_state == UrState.STOPPED
    assert m.sharpa_state == SharpaState.STOPPED
    assert m.machine_ready is False

def test_failed_reset_does_not_poison_later_deploy_start():
    m = GateMachine(mode="deploy")
    hardware_ready(m)
    assert m.attempt_reset_both()
    bias_status(m, "failed", detail="device NACK")
    assert m.bias_phase == "failed"
    assert not m.bias_active
    assert m.attempt_start_deploy()
    finish_start(m)
    assert m.ur_state == UrState.RUNNING
    assert m.sharpa_state == SharpaState.RUNNING

def test_start_from_stopped_waits_for_position_and_times_out_closed():
    m = GateMachine(mode="deploy", execution_start_timeout_s=1.0)
    hardware_ready(m)
    assert m.attempt_start_deploy()
    assert m.execution_start_pending
    assert m.ur_state == UrState.STARTING
    assert m.sharpa_state == SharpaState.STARTING
    assert m.on_execution_start_readiness(position_ready=False) is None
    outcome = m.poll_async_transitions(m.execution_start_started_at + 1.1)
    assert outcome and outcome.hardware_stop is True
    assert m.ur_state == UrState.STOPPED
    assert m.sharpa_state == SharpaState.STOPPED

def test_start_position_failure_closes_both_gates():
    m = GateMachine(mode="deploy")
    hardware_ready(m)
    assert m.attempt_start_deploy()
    outcome = m.on_execution_start_readiness(
        position_ready=False, control_error="right hand mode trigger failed"
    )
    assert outcome and outcome.hardware_stop is True
    assert m.ur_state == UrState.STOPPED
    assert m.sharpa_state == SharpaState.STOPPED

def test_discard_recording_publishes_distinct_command():
    m=GateMachine(); m.record_state=RecordState.RECORDING; m.session_id="episode_test"
    outcome=m.attempt_discard_recording()
    assert outcome.record_command == "discard"
    assert m.record_state == RecordState.IDLE and not m.session_id

def test_camera_allows_fixed_source_delay_but_rejects_frozen_timestamp():
    m=GateMachine(camera_max_age_s=.1); m.ur_state=UrState.RUNNING; m.sharpa_state=SharpaState.RUNNING
    arrival=time.time(); old=arrival-2
    m.update_cameras({k:int(old*1e9) for k in ("ego","left_wrist","right_wrist")},{k:True for k in ("ego","left_wrist","right_wrist")},arrival)
    assert m.attempt_start_recording()
    assert m.cameras.stale_views(arrival + .05, .1) == []
    m.update_cameras({k:int(old*1e9) for k in ("ego","left_wrist","right_wrist")},{k:True for k in ("ego","left_wrist","right_wrist")},arrival + .05)
    stale = m.cameras.stale_views(arrival + .11, .1)
    assert {view for view, _age, _why in stale} == {"ego", "left_wrist", "right_wrist"}
    assert all("not advancing" in why for _view, _age, why in stale)

def test_jog_requires_dry_run_then_matching_confirmation():
    m=GateMachine(); assert not m.confirm_jog("left:x:+1mm"); assert "dry run" in m.gate_reason
    assert m.dry_run_jog("left","x",1,.01,.05,[0,0,0,0,0,0]); token=m.jog_preview.request.confirmation_token
    assert not m.confirm_jog("wrong"); assert m.jog_preview is not None
    assert m.confirm_jog(token); assert m.jog_preview is None
