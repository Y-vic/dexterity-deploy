import json

import numpy as np
import pytest

from recording_node.recording_node import RecorderCore
from test_recording_core import msgset


@pytest.fixture
def core_and_messages(tmp_path, monkeypatch):
    import recording_node.recording_node as module
    monkeypatch.setattr(module, '_fk', lambda q: ([0.]*15, [0.]*15))
    monkeypatch.setattr(module.time, 'time', lambda: 1_000_000_000)
    core = RecorderCore(tmp_path, record_tactile=False, rollout_recording=True)
    messages = msgset()
    messages['hand'].q_cmd_source = 3
    messages['hand'].is_hil = False
    for name, message in messages.items():
        core.update(name, message, received=1_000_000_000)
    assert core.start()
    yield core, messages
    core.writer.close()


def invalidate(hand, reason):
    hand.q_cmd_valid = False
    hand.q_cmd = [float('nan')]*44
    hand.q_cmd_source = 0
    hand.q_cmd_timestamp_ns = 0
    hand.q_cmd_invalid_reason = reason


def valid(hand):
    hand.q_cmd_valid = True
    hand.q_cmd = [.25]*44
    hand.q_cmd_source = 3
    hand.q_cmd_timestamp_ns = 999_000_000
    hand.q_cmd_invalid_reason = ''


def test_start_stop_boundaries_keep_invalid_rows_without_false_errors(core_and_messages):
    core, messages = core_and_messages
    hand = messages['hand']
    reasons = ['not_commanded', 'source_changed', 'stopped']
    for i, reason in enumerate(reasons):
        invalidate(hand, reason)
        assert not core.sample(now=1_000_000_000+i/30)
        assert not core.last_error and not core.quality_errors
        assert '等待首条有效' in core.status()['capture_notice']
    valid(hand)
    assert core.sample(now=1_000_000_000+3/30)
    assert not core.status()['capture_notice']
    invalidate(hand, 'stopped')
    assert not core.sample(now=1_000_000_000+4/30)
    assert '正在收尾' in core.status()['capture_notice']
    saved = core.stop()
    assert (saved/'COMPLETE').is_file()
    assert not core.last_error and not core.quality_errors
    assert core.frames == 1 and core.dropped == 4
    assert '启停边界 4 帧' in core.status()['capture_notice']
    with np.load(saved/'sharpa/joints_44.npz', allow_pickle=False) as data:
        assert data['q_cmd_valid'].tolist() == [False]*3+[True, False]
        assert data['q_cmd_invalid_reason'].tolist() == reasons+['', 'stopped']
        assert np.isnan(data['q_cmd'][[0, 1, 2, 4]]).all()
    quality = json.loads((saved/'manifest.json').read_text())['quality']
    assert not quality['warnings']
    assert quality['boundary_frames'] == {'not_commanded': 1, 'source_changed': 1, 'stopped': 2}
    assert len(json.loads((saved/'data.json').read_text())['data']) == 1


@pytest.mark.parametrize('reason', ['command_failed', 'partial_command', 'invalid_command', '', 'unknown'])
@pytest.mark.parametrize('after_valid', [False, True])
def test_real_or_unknown_invalid_command_remains_an_error(core_and_messages, reason, after_valid):
    core, messages = core_and_messages
    hand = messages['hand']
    if after_valid:
        assert core.sample(now=1_000_000_000)
    invalidate(hand, reason)
    assert not core.sample(now=1_000_000_000+.03)
    assert 'no complete successful SDK command' in core.last_error
    assert core.quality_errors and not core.boundary_frames
    error = core.last_error
    invalidate(hand, 'stopped')
    assert not core.sample(now=1_000_000_000+.06)
    assert core.last_error == error  # Stop must not erase the genuine failure.


@pytest.mark.parametrize('reason', ['not_commanded', 'source_changed'])
def test_startup_reason_is_not_suppressed_in_middle_of_episode(core_and_messages, reason):
    core, messages = core_and_messages
    assert core.sample(now=1_000_000_000)
    invalidate(messages['hand'], reason)
    assert not core.sample(now=1_000_000_000+.03)
    assert core.last_error and not core.boundary_frames


def test_boundary_does_not_mask_other_invalid_sensor_feedback(core_and_messages):
    core, messages = core_and_messages
    invalidate(messages['hand'], 'not_commanded')
    messages['tau'].left_valid = False
    assert not core.sample(now=1_000_000_000)
    assert core.last_error == 'SharpA tau invalid'
    assert not core.boundary_frames


def test_teleop_boundaries_and_next_episode_reset(core_and_messages):
    core, messages = core_and_messages
    core.stop(save=False)
    core.rollout_recording = False
    core.hand_source = 2
    assert core.start()
    hand = messages['hand']
    invalidate(hand, 'not_commanded')
    assert not core.sample(now=1_000_000_000)
    valid(hand)
    hand.q_cmd_source = 2
    assert core.sample(now=1_000_000_000+.03)
    invalidate(hand, 'stopped')
    assert not core.sample(now=1_000_000_000+.06)
    assert not core.last_error and sum(core.boundary_frames.values()) == 2
    core.stop(save=False)
    assert core.start()
    assert not core.boundary_frames and not core.status()['capture_notice']
    invalidate(hand, 'source_changed')
    assert not core.sample(now=1_000_000_000+.09)
    assert not core.last_error and core.boundary_frames == {'source_changed': 1}
