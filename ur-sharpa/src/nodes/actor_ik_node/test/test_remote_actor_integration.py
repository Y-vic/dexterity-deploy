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
from ur_common.command_guards import hand_bounds
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
