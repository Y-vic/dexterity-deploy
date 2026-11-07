"""Exercise Actor's actual ROS wrapper with real messages and fake transports."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from std_msgs.msg import String
from ur_msgs.msg import ExecutableAction

from actor_ik_node import actor_ik_node as actor
from ur_common.command_guards import execution_profile, hand_bounds
from ur_common.geometry import CANONICAL_EEF_FRAME


class FakeNode:
    def __init__(self, name):
        self.parameters, self.messages = {}, {}

    def declare_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def create_publisher(self, cls, topic, depth):
        messages = self.messages.setdefault(topic, [])
        return SimpleNamespace(publish=messages.append)

    def create_subscription(self, *args, **kwargs):
        return None

    def create_timer(self, *args, **kwargs):
        return None

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=time.time_ns(), to_msg=Time))

    def get_logger(self):
        return SimpleNamespace(warning=lambda message: None, info=lambda message: None, error=lambda message: None)

    def destroy_node(self):
        pass


class FakeTrace:
    def __init__(self, root, **kwargs):
        self.root, self.path = Path('/tmp'), ''
        self.events = []
        self.retention_days = self.max_runs = self.max_total_bytes = 1

    def event(self, kind, **kwargs):
        self.events.append((kind, kwargs))

    def start(self, **kwargs):
        return ''

    def stop(self, *args, **kwargs):
        return None

    def trajectory(self, payload):
        pass


@pytest.fixture
def run_actor(monkeypatch):
    import rclpy
    import rclpy.node
    import ur_common.kinematics

    monkeypatch.setattr(rclpy, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rclpy, 'ok', lambda: False)
    monkeypatch.setattr(rclpy.node, 'Node', FakeNode)
    monkeypatch.setattr(ur_common.kinematics, 'UrCommonKinematics', lambda **kwargs: None)
    monkeypatch.setattr(actor, 'ExecutionTraceWriter', FakeTrace)

    def run(check):
        monkeypatch.setattr(rclpy, 'spin', check)
        actor.main()
    return run


def live(node):
    node._machine_running = True
    node._ur_valid = node._hand_valid = True
    node._ur_stamp = node._hand_stamp = time.time_ns()
    node._control_seen = time.monotonic()
    lo, hi = hand_bounds()
    node._hand = np.clip(np.zeros(44, dtype=np.float32), lo, hi)


def chunk(node, *, session='session', action_id='chunk', length=2, rtc=0):
    return ExecutableAction(
        session_id=session, action_id=action_id, timestamp_ns=time.time_ns(),
        frequency_hz=30., action_length=length, execute_length=length, rtc=rtc,
        eef_frame=CANONICAL_EEF_FRAME,
        left_arm_joint=np.tile(node._left, length).tolist(),
        right_arm_joint=np.tile(node._right, length).tolist(),
        hand_joint=np.tile(node._hand, length).tolist())


def step(node):
    node._ur_stamp = node._hand_stamp = time.time_ns()
    node._control_seen = time.monotonic()
    if node._scheduled:
        node._scheduled['due'] = time.monotonic() - .001
    node._process()


def acknowledge_recording(node):
    node._remote._record_status(String(data=json.dumps({
        'rollout_id': node._remote.capture.rollout_id, 'recording': True, 'saving': False})))


def test_actor_reports_real_slice_dispatch_and_terminal_progress(run_actor):
    def check(node):
        live(node)
        target = chunk(node, length=3, rtc=1)
        target.action_length = 30
        target.execute_start = 8
        node._action(target)
        assert node._execution_progress.snapshot()['dispatched_steps'] == 1
        node._last_progress_published_at -= .11
        step(node)
        progress = json.loads(node.messages['/actor_ik/latency_status'][-1].data)['execution_progress']
        assert progress['active']
        assert progress['action_length'] == 30
        assert progress['execute_start'] == 8
        assert progress['execute_length'] == 3
        assert progress['dispatched_steps'] == 2
        assert progress['executed_steps'] == 10
        assert progress['acknowledged_steps'] == 0
        step(node)
        progress = json.loads(node.messages['/actor_ik/latency_status'][-1].data)['execution_progress']
        assert not progress['active'] and progress['success']
        assert progress['dispatched_steps'] == 3
        done = node.messages['/policy/execution_done'][-1]
        assert done.executed_steps == 11
        assert done.execute_start == 8 and done.execute_length == 3
    run_actor(check)


def test_actor_pause_and_stop_publish_frozen_partial_progress(run_actor):
    from ur_msgs.msg import ControlStatus

    def check(node):
        live(node)
        node._action(chunk(node, length=3))
        status = ControlStatus(active_source=3, ur_state=2, sharpa_state=1,
                               execution_paused=True, timestamp_ns=time.time_ns())
        node._control_status(status)
        step(node)
        progress = json.loads(node.messages['/actor_ik/latency_status'][-1].data)['execution_progress']
        assert progress['active'] and progress['paused']
        assert progress['dispatched_steps'] == 1
        status.active_source = 0
        status.timestamp_ns = time.time_ns()
        node._control_status(status)
        progress = json.loads(node.messages['/actor_ik/latency_status'][-1].data)['execution_progress']
        assert not progress['active'] and not progress['success']
        assert progress['dispatched_steps'] == 1
        assert progress['error'] == 'deployment stopped'
        assert len(node.messages['/ur/command']) == 1
    run_actor(check)


def test_actor_ack_diagnostics_follow_current_scope_and_ignore_previous_session(run_actor):
    from ur_msgs.msg import HardwareCommandResult

    def check(node):
        live(node)
        node._execution_scope = 'hands_only'
        node._action(chunk(node, length=1))
        assert not node.messages['/ur/command']
        result = HardwareCommandResult(source=3, action_id='chunk', revision=0,
                                       step_index=0, success=True, timestamp_ns=time.time_ns())
        result.device = 'ur'
        node._ack(result, 'ur')
        assert node._execution_progress.snapshot()['acknowledged_steps'] == 0
        result.device = 'sharpa'
        node._ack(result, 'sharpa')
        assert node._execution_progress.snapshot()['acknowledged_steps'] == 1
        node._action(chunk(node, session='next-session', action_id='next-chunk', length=2))
        node._ack(result, 'sharpa')
        progress = node._execution_progress.snapshot()
        assert progress['session_id'] == 'next-session'
        assert progress['acknowledged_steps'] == 0
        assert progress['dispatched_steps'] == 1
    run_actor(check)


def test_actor_publishes_prefetch_then_terminal_and_accepts_new_session(run_actor):
    def check(node):
        live(node)
        node._action(chunk(node, length=2, rtc=1))
        assert len(node.messages['/ur/command']) == 1
        fetch = node.messages['/policy/execution_done'][-1]
        assert fetch.execute_fetch and not fetch.execute_done
        step(node)
        done = node.messages['/policy/execution_done'][-1]
        assert done.execute_done and done.success and done.executed_steps == 2
        assert node._plans.current is None
        node._action(chunk(node, session='session-after-reset', action_id='new-chunk', length=1))
        step(node)
        assert node.messages['/policy/execution_done'][-1].session_id == 'session-after-reset'
        assert node.messages['/policy/execution_done'][-1].success
    run_actor(check)


def test_actor_defers_first_command_until_own_recording_ack(run_actor):
    def check(node):
        live(node)
        node._remote.record = True
        node._action(chunk(node, length=1))
        assert node._scheduled is not None
        assert not node.messages['/ur/command']
        node._remote._record_status(String(data=json.dumps({'rollout_id': 'old', 'recording': True})))
        step(node)
        assert not node.messages['/ur/command']
        acknowledge_recording(node)
        step(node)
        assert len(node.messages['/ur/command']) == 1
        assert node.messages['/policy/execution_done'][-1].success
    run_actor(check)


def test_actor_pause_preserves_chunk_and_resume_completes_it(run_actor):
    from ur_msgs.msg import ControlStatus

    def check(node):
        live(node)
        node._action(chunk(node, length=2))
        paused = ControlStatus(active_source=3, ur_state=2, sharpa_state=1, execution_paused=True,
                               timestamp_ns=time.time_ns())
        node._control_status(paused)
        step(node)
        assert len(node.messages['/ur/command']) == 1
        assert not node.messages['/policy/execution_done']
        paused.execution_paused = False
        paused.timestamp_ns = time.time_ns()
        node._control_status(paused)
        step(node)
        assert len(node.messages['/ur/command']) == 2
        assert node.messages['/policy/execution_done'][-1].success
    run_actor(check)


def test_human_commands_use_actor_outlet_and_do_not_complete_policy_chunk(run_actor):
    def check(node):
        live(node)
        node._execution_provider = 'hil'
        node._remote.tick()
        acknowledge_recording(node)
        node._remote._learning(String(data=json.dumps({'id': 'takeover-1', 'action': 'takeover', 'issued_at': time.time()})))
        step(node)
        assert node._remote.human.active
        assert len(node.messages['/ur/command']) == 1
        assert node.messages['/sharpa/command'][-1].is_hil
        assert not node.messages['/policy/execution_done']
        node._action(chunk(node, length=1))
        assert node._plans.current is None
        assert len(node.messages['/ur/command']) == 1
    run_actor(check)


def test_missing_ack_observations_expire_during_human_control(run_actor):
    def check(node):
        live(node)
        node._execution_provider = 'hil'
        node._remote.tick()
        acknowledge_recording(node)
        node._remote._learning(String(data=json.dumps({'id': 'takeover-1', 'action': 'takeover', 'issued_at': time.time()})))
        step(node)
        assert node._ack_observations
        for observation in node._ack_observations.values():
            observation['warning_deadline'] = 0.
            observation['expires_at'] = 0.
        node._remote.next_due = time.monotonic() + 1.
        step(node)
        assert not node._ack_observations
        assert node._remote.human.active
        assert len(node.messages['/ur/command']) == 1
    run_actor(check)


def test_human_release_blends_new_session_without_stale_policy_completion(run_actor):
    def check(node):
        live(node)
        node._execution_provider = 'hil'
        node._remote.tick()
        acknowledge_recording(node)
        node._remote._learning(String(data=json.dumps({'id': 'takeover-1', 'action': 'takeover', 'issued_at': time.time()})))
        step(node)
        node._remote._learning(String(data=json.dumps({'id': 'release-1', 'action': 'release', 'issued_at': time.time()})))
        assert not node._remote.human.active
        assert node._remote.handback is not None
        node._action(chunk(node, session='policy-after-human', length=1))
        assert len(node.messages['/ur/command']) == 2
        assert not node.messages['/sharpa/command'][-1].is_hil
        assert len(node.messages['/policy/execution_done']) == 1
        assert node.messages['/policy/execution_done'][0].session_id == 'policy-after-human'
    run_actor(check)


def test_policy_projects_complete_active_slice_and_dispatches_strict_identity(run_actor):
    def check(node):
        live(node)
        target = chunk(node, session='actual-rollout', length=2)
        target.action_length, target.execute_start = 30, 5
        lower, _ = hand_bounds()
        raw = np.asarray(target.hand_joint, dtype=np.float32).reshape(2, 44)
        raw[:, 0] = lower[0] - .031
        target.hand_joint = raw.reshape(-1).tolist()
        original = list(target.hand_joint)
        node._action(target)
        hand = node.messages['/sharpa/command'][-1]
        assert hand.strict_targets and hand.policy_session_id == 'actual-rollout'
        expected = node._hand.copy()
        expected[0] = lower[0]
        np.testing.assert_array_equal(hand.q_cmd, expected)
        trajectory = node.messages['/actor_ik/trajectory'][-1]
        assert trajectory.execute_start == 5 and trajectory.execute_length == 2
        assert np.asarray(trajectory.hand_joint).reshape(2, 44)[1, 0] == lower[0]
        assert list(target.hand_joint) == original
        projection = next(values for kind, values in node._trace.events if kind == 'actor_hand_plan_projected')
        assert projection['corrected_values'] == 2
        assert projection['max_projection_rad'] == pytest.approx(.031)
        assert projection['raw_hand_joint'] == original
        step(node)
        assert len(node.messages['/sharpa/command']) == 2
        assert all(command.strict_targets and command.policy_session_id == 'actual-rollout'
                   for command in node.messages['/sharpa/command'])
    run_actor(check)


def test_policy_prefetch_preflights_from_current_plan_tail(run_actor):
    def check(node):
        live(node)
        first = chunk(node, session='prefetch-rollout', action_id='first', length=2)
        raw = np.asarray(first.hand_joint, dtype=np.float32).reshape(2, 44)
        raw[:, 0] = [.2, .35]
        first.hand_joint = raw.reshape(-1).tolist()
        node._action(first)
        second = chunk(node, session='prefetch-rollout', action_id='second', length=1)
        raw = np.asarray(second.hand_joint, dtype=np.float32)
        raw[0] = .5  # Above .4 from current measurement, close to the queued tail.
        second.hand_joint = raw.tolist()
        node._action(second)
        assert node._plans.next.action.action_id == 'second'
        node._hand[0] = .2
        step(node)
        node._hand[0] = .35
        step(node)
        hand = node.messages['/sharpa/command'][-1]
        assert hand.action_id == 'second'
        assert hand.q_cmd[0] == .5
        assert hand.strict_targets and hand.policy_session_id == 'prefetch-rollout'
    run_actor(check)


def test_policy_rejects_empty_session_before_queue_or_outlet(run_actor):
    def check(node):
        live(node)
        node._action(chunk(node, session='', length=1))
        assert node._plans.current is None
        assert not node.messages['/ur/command'] and not node.messages['/sharpa/command']
        assert 'session ID' in node.messages['/policy/execution_done'][-1].error
    run_actor(check)


@pytest.mark.parametrize('invalid', ['excessive', np.nan, np.inf])
def test_policy_rejects_invalid_later_target_before_queue_or_outlet(run_actor, invalid):
    def check(node):
        live(node)
        target = chunk(node, length=3)
        raw = np.asarray(target.hand_joint, dtype=np.float32).reshape(3, 44)
        lower, _ = hand_bounds()
        raw[2, 0] = (lower[0] - execution_profile()['max_projection_rad'][0] - .01
                     if invalid == 'excessive' else invalid)
        target.hand_joint = raw.reshape(-1).tolist()
        node._action(target)
        assert not node.messages['/ur/command']
        assert not node.messages['/sharpa/command']
        assert not node.messages['/actor_ik/trajectory']
        assert node._plans.current is None and node._scheduled is None
        assert not node.messages['/policy/execution_done'][-1].success
        assert not any(kind == 'actor_plan_ready' for kind, _ in node._trace.events)
    run_actor(check)


def test_policy_rejects_startup_pose_before_either_device_command(run_actor):
    def check(node):
        live(node)
        target = chunk(node, length=1)
        raw = np.asarray(target.hand_joint, dtype=np.float32)
        raw[0] = .8
        target.hand_joint = raw.tolist()
        node._action(target)
        assert not node.messages['/ur/command'] and not node.messages['/sharpa/command']
        assert node._plans.current is None
        done = node.messages['/policy/execution_done'][-1]
        assert not done.success and 'startup_pose_incompatible' in done.error
    run_actor(check)


def test_policy_rechecks_final_limited_hand_before_publishing_any_command(run_actor):
    def check(node):
        live(node)
        node._remote.record = True
        node._action(chunk(node, length=1))
        assert node._scheduled is not None
        assert not node.messages['/ur/command'] and not node.messages['/sharpa/command']
        node._hand[0] = 3.
        acknowledge_recording(node)
        step(node)
        assert not node.messages['/ur/command'] and not node.messages['/sharpa/command']
        done = node.messages['/policy/execution_done'][-1]
        assert not done.success and 'hand_range_rejected' in done.error
    run_actor(check)


def test_policy_slews_desired_target_within_declared_startup_cap(run_actor):
    def check(node):
        live(node)
        node._hand[0] = .712250
        target = chunk(node, session='measured-slew', length=1)
        desired = np.asarray(target.hand_joint, dtype=np.float32)
        desired[0] = 1.121647
        target.hand_joint = desired.tolist()
        delta = float(desired[0] - node._hand[0])
        assert .4 < delta < execution_profile()['startup_step_rad'][0]
        node._action(target)
        command = node.messages['/sharpa/command'][-1]
        expected = np.float32(node._hand[0] + np.float32(.4))
        assert command.q_cmd[0] == expected
        assert command.strict_targets and command.policy_session_id == 'measured-slew'
        trajectory = node.messages['/actor_ik/trajectory'][-1]
        assert trajectory.hand_joint[0] == desired[0]
        assert list(target.hand_joint) == desired.tolist()
        event = next(values for kind, values in node._trace.events if kind == 'command_published')
        assert event['hand_joint'][0] == expected
        assert node._policy_hand_guard.last_target[0] == expected
        assert node.messages['/policy/execution_done'][-1].success
    run_actor(check)


def test_policy_final_limited_command_jump_refuses_both_next_outlets(run_actor):
    def check(node):
        live(node)
        node._action(chunk(node, length=2))
        assert len(node.messages['/ur/command']) == len(node.messages['/sharpa/command']) == 1
        node._hand[0] = 1.1
        step(node)
        assert len(node.messages['/ur/command']) == len(node.messages['/sharpa/command']) == 1
        done = node.messages['/policy/execution_done'][-1]
        assert not done.success and 'hand_command_jump' in done.error
    run_actor(check)


@pytest.mark.parametrize('provider', ['replay', 'hil'])
def test_other_actor_providers_keep_nonstrict_hand_slew(run_actor, provider):
    def check(node):
        live(node)
        node._execution_provider = provider
        target = chunk(node, length=1)
        raw = np.asarray(target.hand_joint, dtype=np.float32)
        raw[0] = .8
        target.hand_joint = raw.tolist()
        node._action(target, provider=provider)
        if provider == 'hil':
            acknowledge_recording(node)
            step(node)
        hand = node.messages['/sharpa/command'][-1]
        assert not hand.strict_targets and hand.policy_session_id == ''
        assert hand.q_cmd[0] == pytest.approx(.4)
    run_actor(check)
