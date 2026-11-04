"""Force return contracts and failure handling with an SDK double; no hardware."""
import io
import json
import os
import threading
from types import SimpleNamespace as NS

import pytest

from ae01_node.feedback import FeedbackInput, FeedbackWriter, ForceRelay, fingertip_forces


@pytest.fixture
def clock():
    return [10.]


def frame(stamp=10_000_000_000):
    return NS(timestamp_ns=stamp, left=list(range(30)), right=list(range(100, 130)),
              left_valid=[True]*5, right_valid=[True]*5)


def control(**kwargs):
    return NS(**dict(dict(sharpa_state=1, active_source=2, execution_paused=False,
                         execution_dry_run=False), **kwargs))


def source(clock):
    return FeedbackInput(.25, .05, clock=lambda: clock[0], wall_ns=lambda: int(clock[0]*1e9))


def test_force_axes_finger_order_and_per_finger_invalidity():
    message = frame()
    assert fingertip_forces(message.left, message.left_valid) == [
        [24., 25., 26.], [18., 19., 20.], [12., 13., 14.], [6., 7., 8.], [0., 1., 2.]]
    message.left_valid[4] = False  # thumb in the public tactile order
    message.left[6] = float('nan')  # ring
    message.left[15] = float('nan')  # middle torque is deliberately ignored
    forces = fingertip_forces(message.left, message.left_valid)
    assert forces[0] == forces[3] == [0., 0., 0.]
    assert forces[2] == [12., 13., 14.]
    with pytest.raises(ValueError):
        fingertip_forces(message.left[:-1], message.left_valid)


def test_duplicate_timestamp_cannot_extend_source_or_receive_deadline(clock):
    inputs = source(clock)
    message = frame()
    inputs.observe(message)
    assert inputs.packet((10., control()), {'left': True})['sides']['left']['enabled']
    clock[0] = 10.2
    message.left_valid[4] = False
    inputs.observe(message)
    packet = inputs.packet((10.2, control()), {'left': True})
    assert packet['expires_s'] == 10.25
    assert packet['sides']['left']['forces'][0] == [0.]*3
    assert not packet['sides']['right']['enabled']
    clock[0] = 10.251
    assert not inputs.packet((10.251, control()), {'left': True})['sides']['left']['enabled']


@pytest.mark.parametrize('stamp', [0, 9_000_000_000, 10_100_000_000])
def test_bad_source_time_revokes_previous_force(clock, stamp):
    inputs = source(clock)
    inputs.observe(frame())
    inputs.observe(frame(stamp))
    assert inputs.packet((10., control()), {'left': True})['sides']['left']['forces'] == [[0.]*3]*5


def test_old_source_and_control_lifetime_are_preserved(clock):
    inputs = source(clock)
    inputs.observe(frame(9_900_000_000))
    assert inputs.packet((9.6, control()), {'left': True})['expires_s'] == pytest.approx(10.1)
    clock[0] = 10.11
    assert not inputs.packet((9.6, control()), {'left': True})['sides']['left']['enabled']
    assert inputs.packet((10.11, control()), {'left': True})['sides']['left']['enabled']
    clock[0] = 10.151
    assert not inputs.packet((10.11, control()), {'left': True})['sides']['left']['enabled']


@pytest.mark.parametrize('status', [None, (9., control()), (10., control(sharpa_state=0)),
    (10., control(sharpa_state=2)), (10., control(active_source=1)),
    (10., control(active_source=0)), (10., control(execution_paused=True)),
    (10., control(execution_dry_run=True))])
def test_feedback_requires_fresh_physical_execution(clock, status):
    inputs = source(clock)
    inputs.observe(frame())
    assert not inputs.packet(status, {'left': True, 'right': True})['sides']['left']['enabled']


class Device:
    def __init__(self):
        self.tasks = []
        self.fail = set()

    def set_task(self, payload):
        task = json.loads(payload)
        self.tasks.append(task)
        return (1 if task['key'] in self.fail else 0), '{}'


@pytest.fixture
def relay(clock):
    result = ForceRelay(NS(ErrorCode=NS(SUCCESS=0)), .25, .05,
                        clock=lambda: clock[0], wall_ns=lambda: int(clock[0]*1e9))
    states = {side: dict(online=True, ready=True, error='', raw_seen_s=10., robot_seen_s=10.)
              for side in ('left', 'right')}
    result.update(states, {side: Device() for side in states})
    inputs = source(clock)
    inputs.observe(frame())
    result.receive(inputs.packet((10., control(active_source=3)), {'left': True, 'right': True}))
    return result


def test_official_sdk_contract_startup_clear_and_source_deadline(relay, clock):
    relay.tick()
    tasks = relay.devices['left'].tasks
    assert tasks[0] == dict(key='clear_3d_force', device_side='LEFT')
    assert tasks[1] == dict(key='set_3d_force', device_side='LEFT', source='wave',
                          force_x=[24., 18., 12., 6., 0.], force_y=[25., 19., 13., 7., 1.],
                          force_z=[26., 20., 14., 8., 2.])
    assert relay.status()['left_active'] and relay.status()['right_active']
    clock[0] = 10.251  # No further acquisition calls are needed to stop feedback.
    relay.tick()
    assert tasks[-1]['key'] == 'clear_3d_force'
    assert not relay.status()['left_active'] and not relay.status()['input_fresh']


@pytest.mark.parametrize('fault', ['raw_seen_s', 'robot_seen_s', 'error', 'online', 'ready'])
def test_glove_fault_clears_only_affected_side(relay, fault):
    relay.tick()
    relay.states['left'][fault] = 'SDK error' if fault == 'error' else 0
    relay.tick()
    assert not relay.status()['left_active'] and relay.status()['right_active']
    assert relay.devices['left'].tasks[-1]['key'] == 'clear_3d_force'


@pytest.mark.parametrize('bad', [None, {}, [], {'sides': None},
    {'monotonic_s': float('nan'), 'expires_s': 10.25, 'timestamp_ns': 10_000_000_000}])
def test_malformed_ipc_clears_latched_force(relay, bad):
    relay.tick()
    relay.receive(bad)
    relay.tick()
    assert not relay.status()['left_active']
    assert relay.devices['left'].tasks[-1]['key'] == 'clear_3d_force'


@pytest.mark.parametrize('forces', [None, [[float('inf')]*3]*5, [[1.]*2]*5, [[True]*3]*5])
def test_malformed_force_does_not_latch_or_kill_watchdog(relay, forces):
    relay.tick()
    relay.latest['sides']['left']['forces'] = forces
    relay.tick()
    assert not relay.status()['left_active'] and relay.status()['right_active']


def test_clear_failure_uses_vendor_zero_fallback_and_blocks_force_until_ack(relay):
    device = relay.devices['left']
    device.fail.add('clear_3d_force')
    relay.tick()
    assert device.tasks[-1] == dict(key='set_3d_force', device_side='LEFT',
                                   force_x=[0.]*5, force_y=[0.]*5, force_z=[0.]*5)
    assert not relay.status()['left_active']
    assert 'clear_3d_force' in relay.status()['error']
    device.fail.clear()
    relay.tick()
    assert relay.status()['left_active'] and not relay.status()['error']


def test_force_send_error_clears_and_is_reported(relay):
    relay.devices['left'].fail.add('set_3d_force')
    relay.tick()
    assert not relay.status()['left_active']
    assert relay.devices['left'].tasks[-1]['key'] == 'clear_3d_force'
    assert 'set_3d_force' in relay.status()['error']


def test_eof_detach_and_close_clear_before_device_release(relay):
    relay.tick()
    relay.read_stream(io.StringIO('{bad json}\n'))
    relay.tick()
    assert not relay.status()['left_active']
    relay.devices['left'].tasks.clear()
    device = relay.devices['left']
    relay.detach('left')
    assert device.tasks[-1]['key'] == 'clear_3d_force'
    assert 'left' not in relay.devices
    right = relay.devices['right']
    relay.close()
    assert right.tasks[-1]['key'] == 'clear_3d_force'


def test_watchdog_clears_on_parent_eof_without_another_acquisition_poll(relay):
    force_sent, force_cleared, parent_closed = (threading.Event() for _ in range(3))
    device = relay.devices['left']
    original = device.set_task
    def task(payload):
        result = original(payload)
        if json.loads(payload)['key'] == 'set_3d_force':
            force_sent.set()
        elif force_sent.is_set():
            force_cleared.set()
        return result
    device.set_task = task
    class Stream:
        def readline(self, *_):
            parent_closed.wait(2)
            return ''
    relay.start(Stream())
    try:
        assert force_sent.wait(2)
        parent_closed.set()
        assert force_cleared.wait(2)
    finally:
        parent_closed.set()
        relay.close()


def test_nonblocking_atomic_pipe_drops_when_full_and_returns_false_on_closed_reader():
    reader, writer_fd = os.pipe()
    with os.fdopen(writer_fd, 'w') as stream:
        writer = FeedbackWriter(stream)
        assert writer.send({'sample': 1})
        assert json.loads(os.read(reader, 4096)) == {'sample': 1}
        with pytest.raises(ValueError):
            writer.send({'sample': 'x'*4096})
        while writer.send({'sample': 'x'*1000}):
            pass
        assert not writer.send({'sample': 'x'*1000})
        os.close(reader)
        assert not writer.send({'sample': 1})
