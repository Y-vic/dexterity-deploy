import json
import threading
import time

import numpy as np
import pytest

from recording_node.alignment import TimestampAligner
from recording_node.recording_node import RecorderCore
from test_recording_core import msgset


def test_burst_arrivals_preserve_grid_unique_sources_and_paired_sharpa():
    names = ("ur", "hand", "tau", "wrench", "deform")
    aligner = TimestampAligner(names, 30)
    origin = 1_000_000_000
    aligner.start(origin)
    for i in range(8):
        stamp = origin + round(i * 1e9 / 30)
        for name in names:
            # Delayed delivery followed by a burst; source clocks keep advancing.
            received = stamp + (45_000_000 if i == 2 else 1_000_000)
            aligner.push(name, stamp, received, (name, i))
    rows = []
    for i in range(8):
        row = aligner.pop_ready(origin + round(i * 1e9 / 30) + 50_000_000)
        rows.append(row)
    assert [row[0] for row in rows] == [origin + round(i * 1e9 / 30) for i in range(8)]
    for name in names:
        assert [row[1][name].value for row in rows] == [(name, i) for i in range(8)]
    assert not aligner.status()["missing"]


def test_late_missing_and_repeated_sources_never_become_duplicate_valid_rows():
    aligner = TimestampAligner(("hand", "tau"), 30)
    aligner.start(1_000_000_000)
    for name in aligner.names:
        assert aligner.push(name, 1_000_000_000, 1_000_000_001, name)
        assert not aligner.push(name, 1_000_000_000, 1_010_000_000, "repeat")
    assert aligner.pop_ready(1_000_000_001)[1]["hand"].value == "hand"
    assert aligner.pop_ready(1_060_000_000) is None
    row = aligner.pop_ready(1_100_000_000)
    assert row[1] == dict(hand=None, tau=None)
    assert aligner.status()["missing"] == dict(hand=1, tau=1)


def test_sharpa_bundle_does_not_pair_tau_from_another_source_frame():
    aligner = TimestampAligner(("tau", "hand"), 30)
    aligner.start(1_000_000_000)
    aligner.push("hand", 990_000_000, 991_000_000, "paired-command")
    aligner.push("tau", 1_010_000_000, 1_011_000_000, "different-state")
    assert aligner.pop_ready(1_050_000_000) is None
    _, selected = aligner.pop_ready(1_067_000_000)
    assert selected["hand"].value == "paired-command"
    assert selected["tau"] is None


def test_overflow_is_counted_and_stop_never_consumes_post_stop_data():
    aligner = TimestampAligner(("hand",), 30, capacity=4)
    aligner.start(1_000_000_000)
    for i in range(5):
        aligner.push("hand", 1_000_000_000+i*33_333_333, 1_000_000_000+i*33_333_333, i)
    assert aligner.status()["overflows"] == {"hand": 1}
    assert aligner.pop_ready(1_200_000_000, end_ns=1_010_000_000, force=True)[1]["hand"] is None
    assert aligner.pop_ready(1_200_000_000, end_ns=1_010_000_000, force=True) is None


def test_clock_phase_jitter_consumes_oldest_eligible_camera_without_skipping():
    aligner = TimestampAligner(("ego",), 30)
    aligner.start(1_000_000_000)
    for stamp in (983_000_000, 1_016_000_000, 1_050_000_000):
        aligner.push("ego", stamp, stamp, stamp)
    assert aligner.pop_ready(1_030_000_000)[1]["ego"].timestamp_ns == 983_000_000
    assert aligner.pop_ready(1_060_000_000)[1]["ego"].timestamp_ns == 1_016_000_000
    assert aligner.pop_ready(1_090_000_000)[1]["ego"].timestamp_ns == 1_050_000_000


def test_stop_drains_in_flight_source_frames_before_closing_writer(tmp_path, monkeypatch):
    import recording_node.recording_node as module
    monkeypatch.setattr(module, "_fk", lambda q: ([0.]*15, [0.]*15))
    core = RecorderCore(tmp_path, record_tactile=False, alignment_wait_frames=2)
    core.start()
    stamp = core.aligner.origin_ns
    arrived = threading.Event()

    def deliver():
        time.sleep(.01)
        for key, msg in msgset().items():
            if key in ("deform", "wrench"):
                continue
            if key == "camera":
                for name in ("ego", "left_wrist", "right_wrist"):
                    setattr(msg, name + "_timestamp_ns", stamp)
            else:
                msg.timestamp_ns = stamp
            if key == "hand":
                msg.q_cmd_timestamp_ns = stamp-1000
            core.update(key, msg)
        arrived.set()

    producer = threading.Thread(target=deliver)
    producer.start()
    try:
        saved = core.stop()
        assert arrived.is_set() and core.frames == 1 and saved is not None
        assert core.aligner.status()["missing"] == {}
        assert core.writer.camera_frame_counts() == dict(ego=1, left_wrist=1, right_wrist=1)
    finally:
        producer.join()
        core.writer.close()


def test_start_preserves_cached_deformation_used_by_first_aligned_row(tmp_path, monkeypatch):
    import recording_node.recording_node as module
    now = 1_800_000_000_000_000_000
    monkeypatch.setattr(module.time, "time_ns", lambda: now)
    monkeypatch.setattr(module.time, "time", lambda: now/1e9)
    monkeypatch.setattr(module, "_fk", lambda q: ([0.]*15, [0.]*15))
    core = RecorderCore(tmp_path, alignment_wait_frames=2)
    messages = msgset()
    try:
        for name, msg in messages.items():
            if name == "camera":
                continue
            msg.timestamp_ns = now - 1_000_000
            if name == "hand":
                msg.q_cmd_timestamp_ns = now - 2_000_000
            core.update(name, msg, received=(now-500_000)/1e9)
        core.start()
        camera = messages["camera"]
        for name in ("ego", "left_wrist", "right_wrist"):
            setattr(camera, name + "_timestamp_ns", now)
        core.update_camera(camera, received=now/1e9)
        assert core.flush_aligned(now+1_000_000) == 1
        monkeypatch.setattr(core, "_alignment_now_ns", lambda: now)
        saved = core.stop()
        assert core.frames == 1
        with np.load(saved/"sharpa/tactile/deform_index.npz") as data:
            assert data["source_timestamp_ns"].tolist() == [now-1_000_000]
    finally:
        core.writer.close()


@pytest.mark.parametrize("held_command", [False, True])
def test_aligned_core_keeps_hil_actions_timestamps_and_actual_emit_times(tmp_path, monkeypatch, held_command):
    import recording_node.recording_node as module
    monkeypatch.setattr(module, "_fk", lambda q: ([0.]*15, [0.]*15))
    now = [1_800_000_000_000_000_000]
    monkeypatch.setattr(module.time, "time_ns", lambda: now[0])
    monkeypatch.setattr(module.time, "time", lambda: now[0]/1e9)
    core = RecorderCore(tmp_path, record_tactile=False, rollout_recording=True, alignment_wait_frames=2)
    try:
        core.start()
        origin = core.aligner.origin_ns
        for i in range(4):
            stamp = origin + round(i * 1e9/30)
            messages = msgset()
            for key, msg in messages.items():
                if key == "camera":
                    for name in ("ego", "left_wrist", "right_wrist"):
                        setattr(msg, name + "_timestamp_ns", stamp)
                else:
                    msg.timestamp_ns = stamp
                if key == "hand":
                    msg.q = [float(i)]*44
                    msg.q_cmd = [(.5 if held_command else float(i)+.5)]*44
                    # A valid held command is not a repeated observation frame.
                    msg.q_cmd_timestamp_ns = (origin if held_command else stamp) - 1_000_000
                    msg.q_cmd_source = 3
                    msg.is_hil = not held_command and i in (1, 2)
                if key not in ("deform", "wrench"):
                    core.update(key, msg, received=(stamp + 1_000_000)/1e9)
        for i in range(4):
            now[0] = origin + round(i * 1e9/30) + 50_000_000
            core.flush_aligned(now[0])
        assert core.attempts == core.frames == 4
        # Freeze the stop boundary at the last requested target; no future rows.
        monkeypatch.setattr(core, "_alignment_now_ns", lambda: origin + 100_000_000)
        saved = core.stop()
        assert saved is not None
        with np.load(saved/"timeline/clock.npz") as clock:
            assert clock["time_unix_ns"].tolist() == [origin + round(i*1e9/30) for i in range(4)]
            assert clock["is_hil"].tolist() == ([False]*4 if held_command else [False, True, True, False])
            delays = clock["sampled_unix_ns"]-clock["time_unix_ns"]
            assert np.all((delays >= 0) & (delays <= 50_000_000))
        with np.load(saved/"sharpa/joints_44.npz") as joints:
            np.testing.assert_array_equal(joints["q"][:, 0], range(4))
            np.testing.assert_array_equal(joints["q_cmd"][:, 0], np.full(4, .5) if held_command else np.arange(4)+.5)
        assert json.loads((saved/"validation.json").read_text())["ok"]
        action = next(r for r in json.loads((saved/"validation.json").read_text())["items"]
                      if r["key"] == "hand_action")
        assert action["ok"] and not action["timing"]["cadence_required"]
    finally:
        core.writer.close()
