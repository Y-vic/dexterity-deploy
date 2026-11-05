import json

import numpy as np
import pytest

from recording_node.recording_node import RecorderCore
from recording_node.rollout import RolloutRecorder
from ur_common.rollout_capture import RolloutCapture
from test_recording_core import msgset


def test_rollout_keeps_demo_schema_and_labels_successful_commands(tmp_path, monkeypatch):
    import recording_node.recording_node as module
    monkeypatch.setattr(module, '_fk', lambda q: ([0.0] * 15, [0.0] * 15))
    monkeypatch.setattr(module.time, 'time', lambda: 1_000_000_000)
    teleop = RecorderCore(tmp_path / 'teleop', record_tactile=False)
    rollout = RecorderCore(tmp_path / 'rollout', record_tactile=False, rollout_recording=True)
    messages = msgset()
    try:
        for key, message in messages.items():
            teleop.update(key, message, received=1_000_000_000)
            rollout.update(key, message, received=1_000_000_000)
        reference = teleop._build_item(1_000_000_000)
        messages['hand'].q_cmd_source = 3
        messages['hand'].is_hil = False
        actual = rollout._build_item(1_000_000_000)
        assert actual['metadata'].pop('is_hil') is False
        assert actual['metadata']['hand_action_source'] == 3
        actual['metadata']['hand_action_source'] = 2
        assert actual == reference
        assert rollout.start()
        for index, is_hil in enumerate((False, True, False)):
            messages['hand'].is_hil = is_hil
            assert rollout.sample(now=1_000_000_000 + index / 30), rollout.last_error
        saved = rollout.stop()
        assert saved is not None
        data = json.loads((saved / 'data.json').read_text())
        assert set(data) == {'info', 'data'}
        assert [row['metadata']['is_hil'] for row in data['data']] == [False, True, False]
        assert all(row['actions'] == reference['actions'] for row in data['data'])
        assert all(row['states'] == reference['states'] for row in data['data'])
        with np.load(saved / 'timeline/clock.npz') as clock:
            assert set(clock.files) == {'time_unix_ns', 'sampled_unix_ns', 'elapsed_ns', 'sample_rate_hz', 'is_hil'}
            assert clock['is_hil'].dtype == np.bool_
            assert clock['is_hil'].tolist() == [False, True, False]
        with np.load(saved / 'sharpa/joints_44.npz') as joints:
            assert joints['q_cmd_valid'].tolist() == [True] * 3
            assert joints['q_cmd_source'].tolist() == [3] * 3
        validation = json.loads((saved / 'validation.json').read_text())
        action = next(item for item in validation['items'] if item['key'] == 'hand_action')
        assert action['frames'] == 3
        # Original paths, media files, validation and checksums are still written.
        for name in ('data.json', 'schema.json', 'manifest.json', 'checksums.sha256',
                     'ur/physical_12.npz', 'timeline/source_quality.npz',
                     'cameras/ego/rgb.mp4', 'cameras/left_wrist/rgb.mp4',
                     'cameras/right_wrist/rgb.mp4'):
            assert (saved / name).is_file(), name
        follower = RolloutRecorder(rollout, lambda: None, lambda: False)
        follower.rollout_id = follower.finished_id = 'one'
        follower.request(dict(active=False, finished_rollout_id='one', category='fail'), now=10)
        follower.tick(now=10)
        failed = saved.parent / 'fail' / saved.name
        assert failed.is_dir() and not saved.exists()
        assert json.loads((failed / 'manifest.json').read_text())['category'] == 'fail'
        assert json.loads((failed / 'data.json').read_text()) == data
        assert follower.status()['category'] == 'fail'
    finally:
        teleop.writer.close()
        rollout.writer.close()


def test_rollout_invalid_or_wrong_source_command_is_not_hil(tmp_path):
    core = RecorderCore(tmp_path, record_tactile=False, rollout_recording=True)
    try:
        hand = msgset()['hand']
        hand.is_hil = True
        core.update('hand', hand, received=1)
        assert core._build_raw_row(1)['is_hil'] is False  # Manus candidate, not Actor.
        hand.q_cmd_source = 3
        assert core._build_raw_row(1)['is_hil'] is True
        hand.q_cmd_valid = False
        assert core._build_raw_row(1)['is_hil'] is False
        hand.q_cmd_valid = True
        assert core._build_raw_row(2)['is_hil'] is False  # Stale feedback.
    finally:
        core.writer.close()


class Core:
    recording = False
    starts = 0
    saves = 0
    def __init__(self):
        self.errors = []
    def start(self):
        self.starts += 1
        self.recording = True
        return True
    def save(self):
        self.saves += 1
        self.recording = False
    def _note_quality_error(self, error):
        self.errors.append(error)


def test_recording_handshake_and_pause_resume_save_separate_episodes():
    core = Core()
    follower = RolloutRecorder(core, core.save, lambda: False)
    clock = [10.0]
    gate = RolloutCapture(lambda request: follower.request(request, now=clock[0]))
    assert gate.tick(False, now=10)  # Preview/idle never starts recording.
    follower.tick(now=10)
    assert core.starts == 0
    assert not gate.tick(True, now=10.1)
    first = gate.rollout_id
    follower.tick(now=10.1)
    gate.observe(dict(follower.status(), recording=True, saving=False), now=10.2)
    assert gate.tick(True, now=10.2)
    # Policy / HIL switches leave wanted=True and keep a single directory.
    assert gate.tick(True, now=10.3)
    assert core.starts == 1
    clock[0] = 10.4
    assert gate.tick(False, now=10.4)
    follower.tick(now=10.4)
    assert core.saves == 0  # Final sensor feedback is drained before closing.
    follower.tick(now=10.7)
    assert core.saves == 1
    clock[0] = 11
    assert not gate.tick(True, now=11)
    assert gate.rollout_id != first
    follower.tick(now=11)
    assert core.starts == 2


def test_recorder_saves_on_actor_disconnect_and_does_not_restart_old_id():
    core = Core()
    follower = RolloutRecorder(core, core.save, lambda: False)
    request = dict(active=True, rollout_id='one')
    follower.request(request, now=10)
    follower.tick(now=10)
    follower.tick(now=12.1)
    follower.tick(now=12.4)
    assert core.saves == 1 and core.errors == ['rollout heartbeat expired']
    follower.request(request, now=13)
    follower.tick(now=13)
    assert core.starts == 1


def test_capture_requires_fresh_matching_recorder_and_stops_on_disconnect():
    gate = RolloutCapture(lambda _: None)
    gate.observe(dict(rollout_id='old', recording=True), now=10)
    assert not gate.tick(True, now=10)
    gate.observe(dict(rollout_id=gate.rollout_id, recording=True), now=10.1)
    assert gate.tick(True, now=10.2)
    with pytest.raises(RuntimeError, match='heartbeat expired'):
        gate.tick(True, now=12.2)
    gate.tick(False, now=13)
    assert not gate.tick(True, now=14)
    with pytest.raises(RuntimeError, match='acknowledge'):
        gate.tick(True, now=20)


def test_start_failure_is_reported_without_repeated_directory_creation():
    core = Core()
    def fail():
        core.starts += 1
        raise OSError('disk unavailable')
    core.start = fail
    follower = RolloutRecorder(core, core.save, lambda: False)
    gate = RolloutCapture(lambda request: follower.request(request, now=10))
    gate.tick(True, now=10)
    follower.tick(now=10)
    follower.tick(now=10.1)
    gate.observe(dict(follower.status(), recording=False), now=10.1)
    with pytest.raises(RuntimeError, match='disk unavailable'):
        gate.tick(True, now=10.2)
    assert core.starts == 1


def test_next_rollout_waits_for_previous_async_save():
    core = Core()
    saving = [True]
    follower = RolloutRecorder(core, core.save, lambda: saving[0])
    follower.request(dict(active=True, rollout_id='two'), now=10)
    follower.tick(now=10)
    assert core.starts == 0
    saving[0] = False
    follower.tick(now=10.1)
    assert core.starts == 1


def test_failure_classification_waits_for_save_and_ignores_other_rollout_ids():
    core = Core()
    classified = []
    core.classify = classified.append
    saving = [False]
    follower = RolloutRecorder(core, core.save, lambda: saving[0])
    follower.request(dict(active=True, rollout_id='current'), now=10)
    follower.tick(now=10)
    follower.request(dict(active=False, finished_rollout_id='old', category='fail'), now=10.1)
    assert follower.category == ''
    follower.request(dict(active=False, finished_rollout_id='current', category='fail'), now=10.2)
    follower.tick(now=10.2)
    follower.tick(now=10.5)
    assert core.saves == 1 and classified == []
    saving[0] = True
    follower.tick(now=10.6)
    assert not classified
    saving[0] = False
    follower.tick(now=10.7)
    follower.tick(now=10.8)
    assert classified == ['fail']


@pytest.mark.parametrize('category', ['success', 'fail'])
def test_stopped_policy_episode_is_saved_then_classified_like_teleop(tmp_path, monkeypatch, category):
    import recording_node.recording_node as module
    from monitor_node.monitor_node import MonitorController

    monkeypatch.setattr(module, '_fk', lambda q: ([0.0] * 15, [0.0] * 15))
    monkeypatch.setattr(module.time, 'time', lambda: 1_000_000_000)
    core = RecorderCore(tmp_path, record_tactile=False, rollout_recording=True)
    saved = []
    follower = RolloutRecorder(core, lambda: saved.append(core.stop(True)), lambda: False)
    clock = [10.0]
    capture = RolloutCapture(lambda request: follower.request(request, now=clock[0]))
    try:
        messages = msgset()
        messages['hand'].q_cmd_source = 3
        messages['hand'].is_hil = False
        for key, message in messages.items():
            core.update(key, message, received=1_000_000_000)
        assert not capture.tick(True, now=10)
        follower.tick(now=10)
        capture.observe(dict(follower.status(), recording=True, saving=False), now=10.1)
        assert capture.tick(True, now=10.1)
        assert core.sample(now=1_000_000_000)
        clock[0] = 10.2
        capture.tick(False, now=10.2)  # Actor stopped; no discard command.
        follower.tick(now=10.2)
        assert not saved  # Keep final feedback before closing the sample.
        follower.tick(now=10.5)
        assert len(saved) == 1 and saved[0] is not None
        assert (saved[0] / 'COMPLETE').exists()
        original_data = (saved[0] / 'data.json').read_bytes()
        labels = []
        def publish_label(body):
            request = json.loads(body)
            labels.append(request)
            core.classify(request['category'])
        controller = MonitorController(recording_root=str(tmp_path), publish_label=publish_label)
        controller.update_upstream('record_status', {**core.status(), 'capture_mode':'rollout'})
        result = controller.action({'action':'mark_'+category, 'saved_path':str(saved[0])})
        assert result['accepted']
        assert labels == [{'category':category, 'saved_path':str(saved[0])}]
        destination = tmp_path / category / saved[0].name
        assert destination.is_dir() and not saved[0].exists()
        assert (destination / 'data.json').read_bytes() == original_data
        assert json.loads((destination / 'manifest.json').read_text())['category'] == category
    finally:
        core.writer.close()
