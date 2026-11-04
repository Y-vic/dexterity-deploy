"""All capture modes retain 6-axis contact data without blocking on derived FK."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import numpy as np
import pytest

from recording_node.recording_node import RecorderCore
from recording_node.rate_metrics import cadence
from test_recording_core import msgset


@pytest.mark.parametrize('mode', ['teleop', 'policy', 'hil'])
def test_all_modes_save_force_torque_and_source_identity_at_30_hz(tmp_path, monkeypatch, mode):
    monkeypatch.setattr('recording_node.recording_node._fk', lambda q: ([.1]*15, [.2]*15))
    core = RecorderCore(tmp_path, rollout_recording=mode != 'teleop')
    messages = msgset()
    messages['deform'].left = bytes(5*240*240)
    messages['deform'].right = bytes([1])*(5*240*240)
    count, start = 36, 1_000_000_000
    try:
        core.start()
        core.inference_rollout_id = 'episode' if mode != 'teleop' else ''
        for index in range(count):
            stamp = start+round(index*1e9/30)
            for name, message in messages.items():
                message.timestamp_ns = stamp
                if name == 'camera':
                    for camera in ('ego', 'left_wrist', 'right_wrist'):
                        setattr(message, camera+'_timestamp_ns', stamp)
                if name == 'hand':
                    message.q_cmd_timestamp_ns = stamp
                    message.q_cmd_source = 2 if mode == 'teleop' else 3
                    message.is_hil = mode == 'hil' and 12 <= index < 24
                if name == 'wrench':
                    message.left = (np.arange(30)+index).tolist()
                    message.right = (-np.arange(30)-index).tolist()
                core.update(name, message, received=stamp/1e9)
            if index == 5 and mode != 'teleop':
                for kind in ('requested', 'completed'):
                    event = json.dumps(dict(event=kind, request_id='r', rollout_id='episode',
                        submitted_unix_ns=stamp, completed_unix_ns=stamp,
                        result=dict(actions=dict(dtype='<f4', shape=[25, 56], data=[[.5]*56]*25))))
                    assert core.record_inference(event, event.encode())
            assert core.sample(stamp/1e9), core.last_error
        saved = core.stop()
        assert saved is not None
        with np.load(saved/'sharpa/tactile/force6d.npz', allow_pickle=False) as data:
            assert data['wrench'].shape == (count, 2, 5, 6)
            assert data['force'].shape == data['torque'].shape == (count, 2, 5, 3)
            assert data['force'].dtype == data['torque'].dtype == np.float32
            np.testing.assert_array_equal(data['force'], data['wrench'][..., :3])
            np.testing.assert_array_equal(data['torque'], data['wrench'][..., 3:])
            np.testing.assert_array_equal(data['wrench'][20, 0, 1], np.arange(6, 12)+20)
            np.testing.assert_array_equal(data['wrench'][20, 1, 1], -np.arange(6, 12)-20)
            assert data['valid'].all()
            assert data['timestamp_ns'].tolist() == [start+round(i*1e9/30) for i in range(count)]
        schema = json.loads((saved/'schema.json').read_text())['tactile_wrench']
        assert schema['force_unit'] == 'N' and schema['torque_unit'] == 'N*m'
        rows = json.loads((saved/'data.json').read_text())['data']
        assert len(rows) == count and '_hand_fk' not in rows[0]
        assert rows[20]['states']['tactile']['left_tactile']['ring'] == list(np.arange(6, 12)+20)
        assert rows[20]['metadata']['stamps']['tactile'] == pytest.approx((start+round(20*1e9/30))/1e9)
        assert rows[0]['states']['left_hand_positions']['positions'] == [.1]*15
        assert all(row['metadata']['hand_action_source'] == (2 if mode == 'teleop' else 3) for row in rows)
        if mode != 'teleop':
            assert sum(row['metadata']['is_hil'] for row in rows) == (12 if mode == 'hil' else 0)
            assert (saved/'raw/policy_inference.jsonl').is_file()
        validation = json.loads((saved/'validation.json').read_text())
        assert validation['ok'], validation
        assert validation['target_hz'] == 30.
        assert all(row['hz'] == pytest.approx(30., abs=.001) for row in validation['items'])
        assert all(row['timing']['stable'] for row in validation['items'])
    finally:
        core.writer.close()


def test_one_nonfinite_finger_is_invalid_without_dropping_other_modalities(tmp_path, monkeypatch):
    monkeypatch.setattr('recording_node.recording_node._fk', lambda q: ([0.]*15, [0.]*15))
    core = RecorderCore(tmp_path)
    try:
        messages = msgset()
        messages['wrench'].left = np.arange(30, dtype=float)
        messages['wrench'].left[9] = float('nan')  # ring torque
        core.start()
        for key, message in messages.items():
            core.update(key, message, received=1.)
        assert core.sample(1.)
        saved = core.stop()
        assert saved is not None
        with np.load(saved/'sharpa/tactile/force6d.npz') as data:
            assert data['valid'].sum() == 9 and not data['valid'][0, 0, 1]
            assert not data['wrench'][0, 0, 1].any()
            np.testing.assert_array_equal(data['wrench'][0, 0, 2], np.arange(12, 18))
        row = json.loads((saved/'data.json').read_text())['data'][0]
        assert not row['metadata']['tactile_valid']['wrench']['left'][1]
        assert not core.validation['ok']
    finally:
        core.writer.close()


def test_fk_cannot_block_sampling_or_sensor_updates_and_uses_captured_joints(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    captured = []
    def slow_fk(q):
        captured.append(list(q))
        entered.set()
        assert release.wait(3)
        return [0.]*15, [0.]*15
    monkeypatch.setattr('recording_node.recording_node._fk', slow_fk)
    core = RecorderCore(tmp_path, record_tactile=False)
    try:
        messages = msgset()
        for key, message in messages.items():
            core.update(key, message, received=1.)
        core.start()
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(core.sample, 1.).result(timeout=1)
            assert entered.wait(1)
            messages['hand'].q[:] = [1.]*44
            pool.submit(core.update, 'hand', messages['hand'], 1.1).result(timeout=1)
            assert captured == [[0.]*44]
            assert core.status()['writer_queues']['structured']['pending'] == 1
            release.set()
        saved = core.stop()
        assert saved is not None
        row = json.loads((saved/'data.json').read_text())['data'][0]
        assert row['states']['left_hand']['qpos'] == [0.]*22
    finally:
        release.set()
        core.writer.close()


def test_repeated_deformation_neither_duplicates_large_writes_nor_refreshes_freshness(tmp_path):
    core = RecorderCore(tmp_path)
    try:
        core.start()
        message = msgset()['deform']
        assert core.update_deformation(message, received=1.)
        assert core.update_deformation(message, received=1.2)
        assert core.latest['deform'].received == 1.
        assert core.writer._drain()
        assert core.writer._deformation_frames == 1
        assert not core._fresh(core.latest['deform'], 1.3)
    finally:
        core.stop(False)
        core.writer.close()


def test_async_fk_failure_preserves_raw_force_and_marks_episode_incomplete(tmp_path, monkeypatch):
    def fail_fk(_):
        raise RuntimeError('test FK unavailable')
    monkeypatch.setattr('recording_node.recording_node._fk', fail_fk)
    core = RecorderCore(tmp_path)
    try:
        core.start()
        for key, message in msgset().items():
            core.update(key, message, received=1.)
        assert core.sample(1.)
        assert core.stop() is None
        saved = core.writer.last_result_dir
        assert (saved/'ERROR').exists() and not (saved/'COMPLETE').exists()
        with np.load(saved/'sharpa/tactile/force6d.npz') as force:
            assert force['valid'].all() and force['torque'].shape == (1, 2, 5, 3)
        assert 'test FK unavailable' in (saved/'RECORDING_ERRORS.json').read_text()
    finally:
        core.writer.close()


@pytest.mark.parametrize('hz,stable', [(30, True), (29.8, True), (29, False), (25, False)])
def test_source_frequency_is_measured_against_30_hz(hz, stable):
    stamps = 1_000_000_000+np.rint(np.arange(300)/hz*1e9).astype(np.int64)
    result = cadence(stamps, 30, (stamps[-1]-stamps[0])/1e9+1/30)
    assert result['stable'] == stable


def test_repeats_and_long_gap_cannot_hide_behind_nominal_fps():
    stamps = 1_000_000_000+np.rint(np.arange(300)/30*1e9).astype(np.int64)
    repeated = stamps.copy(); repeated[1:] = repeated[0]
    assert not cadence(repeated, 30, 10.)['stable']
    gaps = stamps.copy(); gaps[100] = gaps[99]  # next gap is 66.7 ms
    result = cadence(gaps, 30, 10.)
    assert result['source_hz'] > 29.4 and result['gap_count'] == 1 and not result['stable']


def test_only_the_part_of_a_source_gap_inside_recording_counts_as_a_stall():
    origin = 1_000_000_000
    stamps = origin+np.array([-57, -2, 31, 67])*1_000_000
    assert not cadence(stamps, 30, 4/30)['stable']
    result = cadence(stamps, 30, 4/30, window_start_ns=origin)
    assert result['stable'] and result['boundary_gaps_clipped'] == 2
    # A gap spanning 70 ms of actual capture is still a failure, even if it
    # starts before recording. Boundary clipping cannot conceal a real stall.
    stamps = origin+np.array([-57, 70, 100, 133])*1_000_000
    assert not cadence(stamps, 30, 4/30, window_start_ns=origin)['stable']
