"""Recording/HIL integration at the Actor boundary; no hardware or ROS graph."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from actor_ik_node.remote_bridge import RemoteExecutionBridge
from ur_common.command_guards import hand_bounds
from ur_common.hil import PolicyHandback


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


@pytest.fixture
def bridge():
    publishers, subscriptions, aborts = {}, {}, []
    lo, hi = hand_bounds()
    node = SimpleNamespace(
        _hil_enabled=True, _trace=SimpleNamespace(path=''),
        _machine_running=True, _execution_paused=False,
        _execution_provider='policy', _execution_scope='full', _execution_dry_run=False,
        _control_seen=time.monotonic(), _hand_stamp=time.time_ns(), _ur_stamp=time.time_ns(),
        _left=np.zeros(6), _right=np.zeros(6), _hand=np.clip(np.zeros(44), lo, hi),
        _hil_return_duration_s=1.,
        _abort_active=lambda reason, **kw: aborts.append(reason),
        get_logger=lambda: SimpleNamespace(warning=lambda message: None),
    )
    def publisher(kind, topic, qos):
        publishers[topic] = Publisher()
        return publishers[topic]
    node.create_publisher = publisher
    node.create_subscription = lambda kind, topic, cb, qos: subscriptions.update({topic: cb})
    result = RemoteExecutionBridge(node)
    result.test_publishers, result.test_aborts = publishers, aborts
    return result


def message(value):
    return SimpleNamespace(data=json.dumps(value))


def acknowledge(bridge):
    bridge._record_status(message(dict(rollout_id=bridge.capture.rollout_id, recording=True, saving=False)))


def test_recording_waits_for_own_rollout_ack(bridge):
    bridge.record = True
    assert not bridge.tick()
    rollout = bridge.capture.rollout_id
    assert rollout
    bridge._record_status(message(dict(rollout_id='previous', recording=True)))
    assert not bridge.tick()
    acknowledge(bridge)
    assert bridge.tick()
    bridge.publish_status()
    state = json.loads(bridge.test_publishers['/hil/state'].messages[-1])
    assert state['rollout_recording'] == dict(rollout_id=rollout, required=True, requested=True, recording=True)


def test_pause_finishes_recording_and_resume_requires_new_ack(bridge):
    bridge.record = True
    bridge.tick()
    acknowledge(bridge)
    assert bridge.tick()
    rollout = bridge.capture.rollout_id
    bridge.node._execution_paused = True
    bridge.tick()
    assert not bridge.capture.rollout_id
    event = json.loads(bridge.test_publishers['/hil/rollout'].messages[-1])
    assert event['finished_rollout_id'] == rollout
    assert event['termination']['outcome'] == 'truncated'
    bridge.node._execution_paused = False
    assert not bridge.tick()
    assert bridge.capture.rollout_id != rollout
    acknowledge(bridge)
    assert bridge.tick()
    bridge.node._machine_running = False
    bridge.tick()
    assert not bridge.capture.rollout_id


def test_recorder_failure_stops_once_and_requires_restart(bridge):
    bridge.record = True
    bridge.tick()
    acknowledge(bridge)
    assert bridge.tick()
    bridge._record_status(message(dict(rollout_id=bridge.capture.rollout_id, recording=False)))
    assert not bridge.tick()
    assert bridge.error
    assert not bridge.tick()
    assert bridge.test_publishers['/control/ui_command'].messages == ['stop_deploy']
    assert len(bridge.test_aborts) == 1
    bridge.node._machine_running = False
    bridge.tick()
    bridge.node._machine_running = True
    assert not bridge.tick()  # A new recorder acknowledgement is required.
    assert not bridge.error


def test_preview_does_not_start_recording(bridge):
    bridge.record = True
    bridge.node._execution_dry_run = True
    assert bridge.tick()
    assert not bridge.capture.rollout_id


@pytest.mark.parametrize('stale', ['robot', 'control', 'policy', 'future'])
def test_stale_feedback_prevents_execution(bridge, stale):
    assert bridge.tick()
    if stale == 'robot':
        bridge.node._hand_stamp -= 2_000_000_000
    elif stale == 'future':
        bridge.node._ur_stamp += 2_000_000_000
    elif stale == 'control':
        bridge.node._control_seen -= 2.
    else:
        bridge.policy_seen -= 4.
    assert not bridge.tick()
    assert bridge.error
    assert bridge.test_publishers['/control/ui_command'].messages == ['stop_deploy']


def test_remote_fault_is_not_ignored(bridge):
    assert bridge.tick()
    bridge._policy_status(message(dict(phase='fault', error='remote disconnected')))
    assert not bridge.tick()
    assert bridge.error == 'remote disconnected'


def test_stop_ack_waits_for_state_but_blocks_further_commands(bridge):
    assert bridge.tick()
    bridge.rejected('arbitration_rejected', 'stopping')
    assert not bridge.tick()
    assert not bridge.error
    bridge.node._machine_running = False
    bridge.tick()
    assert not bridge.error
    assert bridge.pending_rejection is None


def test_rejection_without_stop_state_eventually_faults(bridge):
    assert bridge.tick()
    bridge.rejected('hardware_stopped', 'hardware stopped')
    bridge.pending_rejection = (time.monotonic()-1, 'hardware stopped')
    assert not bridge.tick()
    assert bridge.error == 'hardware stopped'


@pytest.mark.parametrize('frequency', [15., 30., 50., 60.])
def test_hil_handback_uses_remote_command_frequency(bridge, frequency):
    anchor = np.r_[bridge.node._left, bridge.node._right, bridge.node._hand]
    bridge.timeline_steps = 2.4  # Start between 30 Hz reference ticks.
    bridge.handback = PolicyHandback(anchor, bridge.timeline_steps, 1.)
    target = anchor.copy()
    target[:12] += .1
    for _ in range(int(frequency)):
        effective = bridge.effective(target, frequency)
        bridge.committed(effective, frequency)
    np.testing.assert_allclose(effective, target, atol=1e-7)
    assert bridge.handback.progress(bridge.timeline_steps) == pytest.approx(1.)


def test_hil_command_reports_matching_success_and_rejection(bridge):
    bridge.node._execution_provider = 'hil'
    bridge._learning(message(dict(action='takeover', id='take', issued_at=time.time())))
    assert bridge.human.active
    assert bridge.learning_command == dict(id='take', accepted=True, reason='')
    bridge._learning(message(dict(action='release', id='expired', issued_at=0)))
    assert bridge.human.active
    assert bridge.learning_command['id'] == 'expired'
    assert not bridge.learning_command['accepted']
    assert 'expired' in bridge.learning_command['reason']
    bridge._learning(message(dict(action='release', id='release', issued_at=time.time())))
    assert not bridge.human.active
    assert bridge.learning_command['accepted']
    bridge._learning(message(dict(action='release', id='release', issued_at=time.time())))
    assert bridge.learning_command['accepted']  # Idempotent duplicate.


def test_new_run_does_not_reuse_old_takeover_anchor(bridge):
    assert bridge.tick()
    bridge.last_target = np.ones(56)
    bridge.node._machine_running = False
    bridge.tick()
    bridge.node._machine_running = True
    bridge.node._execution_provider = 'hil'
    bridge.tick()
    assert bridge.last_target is None
    bridge._learning(message(dict(action='takeover', id='new-run', issued_at=time.time())))
    np.testing.assert_allclose(bridge.human.anchor, np.r_[bridge.node._left, bridge.node._right, bridge.node._hand])
