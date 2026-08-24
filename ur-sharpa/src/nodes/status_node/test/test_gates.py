import time
from status_node.gates import GateMachine, UrState, SharpaState, RecordState, Source

def fresh(m, now):
    m.update_cameras({"ego":int(now*1e9),"left_wrist":int(now*1e9),"right_wrist":int(now*1e9)}, {"ego":True,"left_wrist":True,"right_wrist":True}, now)

def hardware_ready(m):
    m.update_hardware_readiness(ur_ready=True, sharpa_ready=True)

def test_all_legal_transitions_and_side_effects():
    m=GateMachine()
    hardware_ready(m)
    assert not m.attempt_run_ur(); assert "reset both" in m.gate_reason
    reset = m.attempt_reset_both()
    assert reset.bias_request == "home_both" and reset.hardware_stop is False
    m.on_bias_status({"phase":"done"})
    assert m.ur_state == UrState.HOMED and m.sharpa_state == SharpaState.STOPPED
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
    assert m.ur_state == UrState.STOPPED
    assert m.sharpa_state == SharpaState.STOPPED

    m.on_bias_status({"phase":"done"})
    assert m.ur_state == UrState.HOMED
    assert m.sharpa_state == SharpaState.STOPPED
    assert m.attempt_start_deploy()
    assert m.ur_state == UrState.RUNNING and m.sharpa_state == SharpaState.RUNNING

    stopped=m.attempt_stop_deploy()
    assert stopped.hardware_stop is True
    assert m.ur_state == UrState.HOMED
    assert m.sharpa_state == SharpaState.STOPPED


def test_ur_home_actions_release_hardware_stop_latch():
    machine = GateMachine()
    assert machine.attempt_start_ur().hardware_stop is False
    assert machine.attempt_reset_ur().hardware_stop is False

def test_start_deploy_requires_deploy_mode_and_complete_reset():
    teleop=GateMachine(mode="teleop")
    teleop.ur_state=UrState.HOMED; teleop.sharpa_state=SharpaState.RUNNING
    assert not teleop.attempt_start_deploy()
    assert "expected deploy" in teleop.gate_reason

    deploy=GateMachine(mode="deploy")
    deploy.ur_state=UrState.HOMED
    assert not deploy.attempt_start_deploy()
    assert "reset both" in deploy.gate_reason


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
    m.on_bias_status({"phase":"done"})
    assert m.status_payload()["machine_ready"] is True
    assert m.attempt_run_ur().hardware_stop is False
    assert not m.attempt_reset_both()
    paused=m.attempt_pause()
    assert paused.hardware_stop is True
    assert m.ur_state == UrState.HOMED and m.sharpa_state == SharpaState.STOPPED
    assert m.attempt_run_ur().hardware_stop is False

def test_machine_cannot_reset_or_start_when_hardware_is_not_ready():
    m=GateMachine()
    assert not m.attempt_reset_both()
    assert "UR is not in Normal mode" in m.gate_reason
    m.update_hardware_readiness(ur_ready=True, sharpa_ready=True)
    assert m.attempt_reset_both()
    m.on_bias_status({"phase":"done"})
    m.update_hardware_readiness(ur_ready=False)
    assert not m.attempt_run_ur()
    assert "UR is not in Normal mode" in m.gate_reason

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
