"""Offline ownership transitions; no ROS graph or hardware is started."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from status_node.gates import GateMachine, RecordState, SharpaState, Source, UrState
from status_node.status_node import StatusNode


def test_workbench_switch_changes_source_only_with_closed_gates():
    machine = GateMachine(allow_mode_switch=True)
    assert machine.active_source == Source.VIVE
    for mode in ('policy', 'hil', 'replay'):
        assert machine.attempt_set_workbench_mode(mode)
        assert machine.active_source == Source.ACTOR
        assert machine.execution_provider == mode
        assert machine.ur_state == UrState.STOPPED
    assert machine.attempt_set_workbench_mode('teleop')
    assert machine.active_source == Source.VIVE
    assert not GateMachine().attempt_set_workbench_mode('hil')


@pytest.mark.parametrize('fields', [
    {'ur_state': UrState.RUNNING}, {'ur_state': UrState.HOMING},
    {'sharpa_state': SharpaState.STARTING}, {'execution_dry_run': True},
    {'execution_paused': True}, {'execution_start_pending': True},
    {'record_state': RecordState.RECORDING},
])
def test_busy_mode_switch_keeps_previous_owner(fields):
    machine = GateMachine(allow_mode_switch=True, **fields)
    assert not machine.attempt_set_workbench_mode('hil')
    assert machine.active_source == Source.VIVE


def fake_status():
    # Bind the real HTTP/ROS command adapter to a pure offline fixture.
    node = SimpleNamespace(machine=GateMachine(allow_mode_switch=True), _hil_enabled=True,
                           upstream={'record_status': {'capture_mode': 'teleop'}},
                           _record_seen=time.monotonic(), _publish=lambda: None)
    node._snapshot=lambda: {'status': node.machine.status_payload()}
    return node


@pytest.mark.parametrize('record,age', [({'saving': True}, 0), ({'recording': True}, 0),
                                       ({'rollout_requested': True}, 0), ({}, 3)])
def test_status_rejects_switch_until_recorder_is_fresh_and_finished(record, age):
    node = fake_status()
    node.upstream['record_status'] = record
    node._record_seen -= age
    result = StatusNode._apply_action_locked(node, {'action': 'select_hil'})
    assert not result['accepted']
    assert node.machine.mode == 'teleop'


def test_status_switch_waits_for_recorder_mode_ack_before_start():
    node = fake_status()
    assert StatusNode._apply_action_locked(node, {'action': 'select_hil'})['accepted']
    result = StatusNode._apply_action_locked(node, {'action': 'start_deploy'})
    assert not result['accepted']
    assert '数采节点确认' in result['gate_reason']
    assert not StatusNode._apply_action_locked(node, {'action': 'run_ur'})['accepted']


def test_operator_stop_cancels_reset_and_ignores_late_bias_completion():
    machine = GateMachine(allow_mode_switch=True, ur_hardware_ready=True, sharpa_hardware_ready=True)
    assert machine.attempt_reset_both()
    action_id = machine.bias_action_id
    assert machine.attempt_stop().hardware_stop is True
    machine.on_bias_status({'action_id': action_id, 'phase': 'done'})
    assert machine.ur_state == UrState.STOPPED
    assert not machine.machine_ready
    assert machine.attempt_set_workbench_mode('hil')


@pytest.mark.parametrize("dry_run", [False, True])
def test_paused_execution_can_reset_and_cannot_resume_old_chunk(dry_run):
    machine = GateMachine(mode="deploy", ur_hardware_ready=True,
                          sharpa_hardware_ready=True, execution_paused=True,
                          execution_dry_run=dry_run, ur_state=UrState.RUNNING,
                          sharpa_state=SharpaState.RUNNING)
    outcome = machine.attempt_reset_both()
    assert outcome.accepted and outcome.bias_request == "home_both"
    assert machine.ur_state == UrState.HOMING
    assert machine.sharpa_state == SharpaState.HOMING
    assert not machine.execution_paused and not machine.execution_dry_run
    assert not machine.attempt_continue_deploy()


@pytest.mark.parametrize("fields", [
    {"ur_state": UrState.RUNNING}, {"sharpa_state": SharpaState.RUNNING},
    {"execution_dry_run": True},
])
def test_reset_still_rejects_unpaused_execution(fields):
    machine = GateMachine(mode="deploy", ur_hardware_ready=True, sharpa_hardware_ready=True, **fields)
    assert not machine.attempt_reset_both()


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("record,age", [
    ({"saving": True, "recording": True}, 0),
    ({"saving": True, "recording": False, "rollout_requested": True}, 0),
    ({"saving": True, "recording": False}, 3),
    ({"saving": True}, 0),
])
def test_reset_waits_for_fresh_closed_sampling_not_just_saving_flag(record, age, paused):
    node = fake_status()
    node.machine.execution_paused = paused
    node.upstream["record_status"] = record
    node._record_seen -= age
    result = StatusNode._apply_action_locked(node, {"action": "reset_both"})
    assert not result["accepted"]
    assert "采样结束" in result["gate_reason"]
    assert node.machine.execution_paused == paused


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("saving", [False, True])
def test_status_accepts_reset_during_background_save_after_sampling_ends(paused, saving):
    node = fake_status()
    node.machine.execution_paused = paused
    node.upstream["record_status"] = {"recording": False, "saving": saving}
    node.machine.update_hardware_readiness(ur_ready=True, sharpa_ready=True)
    published = []
    node.stop_pub = SimpleNamespace(publish=lambda message: published.append(message.data))
    node.bias_pub = object()
    node._publish_string = lambda publisher, body: published.append(json.loads(body))
    result = StatusNode._apply_action_locked(node, {"action": "reset_both", "reset_target": {"mode": "bias"}})
    assert result["accepted"]
    assert node.machine.ur_state == UrState.HOMING
    assert published[1]["request"] == "home_both"


def ready_capture_node():
    node = fake_status()
    machine = node.machine
    machine.ur_state = UrState.HOMED
    machine.sharpa_state = SharpaState.HOMED
    machine.machine_ready = True
    machine.update_hardware_readiness(ur_ready=True, sharpa_ready=True)
    now = time.time()
    views = ('ego', 'left_wrist', 'right_wrist')
    machine.update_cameras(dict.fromkeys(views, int(now * 1e9)), dict.fromkeys(views, True), now)
    node.published = []
    node.stop_pub = SimpleNamespace(publish=lambda msg: node.published.append(('stop', msg.data)))
    node.record_pub = object()
    node._publish_string = lambda publisher, body: node.published.append(('record', body))
    return node


def test_combined_capture_starts_both_lifecycles_and_stop_saves():
    node = ready_capture_node()
    result = StatusNode._apply_action_locked(node, {'action': 'start_teleop_record'})
    assert result['accepted']
    assert node.machine.ur_state == UrState.RUNNING
    assert node.machine.sharpa_state == SharpaState.RUNNING
    assert node.machine.record_state == RecordState.RECORDING
    assert node.machine.session_id
    assert node.published == [('stop', False), ('record', 'start')]
    node.published.clear()
    assert not StatusNode._apply_action_locked(node, {'action': 'start_teleop_record'})['accepted']
    assert node.published == []
    assert StatusNode._apply_action_locked(node, {'action': 'stop'})['accepted']
    assert node.published == [('stop', True), ('record', 'stop')]


@pytest.mark.parametrize('record,age', [
    ({'capture_mode': 'rollout'}, 0), ({'capture_mode': 'teleop'}, 3),
    ({'capture_mode': 'teleop', 'saving': True}, 0),
    ({'capture_mode': 'teleop', 'recording': True}, 0),
    ({'capture_mode': 'teleop', 'rollout_requested': True}, 0),
])
@pytest.mark.parametrize('switchable', [False, True])
def test_combined_capture_rejects_unavailable_recorder_without_motion(record, age, switchable):
    node = ready_capture_node()
    node.machine.allow_mode_switch = switchable
    node.upstream['record_status'] = record
    node._record_seen -= age
    assert not StatusNode._apply_action_locked(node, {'action': 'start_teleop_record'})['accepted']
    assert node.machine.ur_state == UrState.HOMED
    assert node.machine.sharpa_state == SharpaState.HOMED
    assert node.machine.machine_ready
    assert node.machine.record_state == RecordState.IDLE
    assert node.published == []


@pytest.mark.parametrize('failure', ['camera', 'reset', 'ur', 'hand', 'mode'])
def test_combined_capture_checks_all_gates_before_starting_either_lifecycle(failure):
    node = ready_capture_node()
    machine = node.machine
    if failure == 'camera':
        machine.update_cameras({'ego': 1}, {'ego': False})
    elif failure == 'reset':
        machine.machine_ready = False
    elif failure == 'ur':
        machine.ur_hardware_ready = False
    elif failure == 'hand':
        machine.sharpa_hardware_ready = False
    else:
        machine.mode = 'deploy'
    assert not StatusNode._apply_action_locked(node, {'action': 'start_teleop_record'})['accepted']
    assert machine.ur_state == UrState.HOMED
    assert machine.sharpa_state == SharpaState.HOMED
    assert machine.record_state == RecordState.IDLE
    assert machine.session_id == ''
    assert node.published == []


@pytest.mark.parametrize('switchable', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_authoritative_status_gate_enforces_hil_launch_capability(switchable, enabled):
    node = fake_status()
    node.machine.allow_mode_switch = switchable
    node.machine.mode = 'teleop' if switchable else 'deploy'
    node._hil_enabled = enabled
    result = StatusNode._apply_action_locked(node, {'action': 'select_hil'})
    assert result['accepted'] is enabled
    assert node.machine.execution_provider == ('hil' if enabled else 'policy')
    if not enabled:
        assert 'enable_hil:=true' in result['gate_reason']
        assert node.machine.mode == ('teleop' if switchable else 'deploy')


@pytest.mark.parametrize('payload', ['select_hil', '{"action":"select_hil"}'])
def test_ros_command_cannot_bypass_disabled_hil(payload):
    node = fake_status()
    node._hil_enabled = False
    node._apply_action = lambda body: StatusNode._apply_action_locked(node, body)
    StatusNode._command(node, SimpleNamespace(data=payload))
    assert node.machine.mode == 'teleop'
    assert node.machine.execution_provider == 'policy'
    assert 'HIL 未启用' in node.machine.gate_reason
