import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from recording_node.demo_node import DemoService, normalize_recording_root, select_d435
from recording_node.demo_overlay import (
    SensorHistory,
    TorqueHistory,
    finger_square_sums,
    render_overlay,
)
from recording_node.demo_recorder import acquisition_timing, DemoSession, SENSOR_TOPICS


def measurements(stamp=1_000_000_000):
    history = SensorHistory()
    hand = SimpleNamespace(q=np.ones(44), left_valid=True, right_valid=True, timestamp_ns=stamp)
    tau = SimpleNamespace(left=np.arange(1, 23), right=-np.arange(23, 45),
                          left_valid=True, right_valid=True, timestamp_ns=stamp)
    wrench = SimpleNamespace(left=np.arange(30), right=np.arange(30) + 100,
                             left_valid=[True] * 5, right_valid=[True] * 5, timestamp_ns=stamp)
    deform = SimpleNamespace(left=np.repeat(np.arange(5), 240 * 240),
                             right=np.repeat(np.arange(5) + 10, 240 * 240),
                             left_valid=[True] * 5, right_valid=[True] * 5, timestamp_ns=stamp)
    for key, msg in (("hand", hand), ("tau", tau), ("wrench", wrench), ("deform", deform)):
        history.update(key, msg, stamp)
    return history


def test_finger_groups_include_every_joint_once_without_sqrt():
    q = np.arange(1, 45, dtype=float)
    result = finger_square_sums(q)
    assert sum(map(sum, result)) == np.dot(q, q)
    assert result[0][0] == 55
    assert result[1][4] == sum(i * i for i in range(40, 45))
    assert finger_square_sums(np.ones(44)) == [[5, 4, 4, 4, 5]] * 2
    with pytest.raises(ValueError):
        finger_square_sums(np.ones(22))
    with pytest.raises(ValueError):
        finger_square_sums([float("nan")] * 44)


def test_tactile_order_matches_joint_order_and_rejects_stale_and_future():
    history = measurements()
    metrics, planes = history.snapshot(1_010_000_000)
    assert metrics["fingers"][0]["wrench"] == list(range(24, 30))
    assert metrics["fingers"][5]["wrench"] == list(range(124, 130))
    assert planes[0][0, 0] == 4
    assert planes[5][0, 0] == 14
    assert planes[-1][0, 0] == 10
    for stamp in (999_999_999, 1_300_000_000):
        metrics, planes = history.snapshot(stamp)
        assert all(f["joint_square_sum_rad2"] is None and f["wrench"] is None for f in metrics["fingers"])
        assert planes == [None] * 10


def test_stale_source_is_invalid_even_after_recent_receive():
    history = SensorHistory()
    history.update("hand", SimpleNamespace(q=np.ones(44), timestamp_ns=1,
                                           left_valid=True, right_valid=True), 1_000_000_000)
    metrics, _ = history.snapshot(1_000_000_001)
    assert metrics["fingers"][0]["joint_square_sum_rad2"] is None


def test_torque_squares_use_each_hands_joint_groups_and_validity():
    history = measurements()
    metrics, _ = history.snapshot(1_010_000_000)
    values = [f["joint_torque_square_sum_nm2"] for f in metrics["fingers"]]
    assert values[0] == 55
    assert values[5] == sum(i*i for i in range(23, 28))
    assert sum(values) == sum(i*i for i in range(1, 45))
    history.update("tau", SimpleNamespace(left=[2]*22, right=[3]*22,
                                          left_valid=False, right_valid=True,
                                          timestamp_ns=1_020_000_000), 1_020_000_000)
    metrics, _ = history.snapshot(1_030_000_000)
    assert all(f["joint_torque_square_sum_nm2"] is None for f in metrics["fingers"][:5])
    assert [f["joint_torque_square_sum_nm2"] for f in metrics["fingers"][5:]] == [45,36,36,36,45]
    metrics, _ = history.snapshot(1_500_000_000)
    assert all(f["joint_torque_square_sum_nm2"] is None for f in metrics["fingers"])


def test_select_d435_never_selects_wrist_and_requires_unambiguous_device():
    devices = [{"name": "Intel RealSense D405", "serial": "wrist"},
               {"name": "Intel RealSense D435", "serial": "demo"}]
    assert select_d435(devices) == "demo"
    with pytest.raises(RuntimeError):
        select_d435(devices, "wrist")
    with pytest.raises(RuntimeError):
        select_d435(devices, excluded=("demo",))
    devices.append({"name": "Intel RealSense D435I", "serial": "other"})
    with pytest.raises(RuntimeError):
        select_d435(devices)
    assert select_d435(devices, "other") == "other"


def test_demo_start_rejects_missing_camera(tmp_path):
    service = DemoService(str(tmp_path))
    with pytest.raises(ValueError):
        service.command("start")
    assert not list(tmp_path.iterdir())


def test_recording_root_requires_safe_absolute_path(tmp_path):
    assert normalize_recording_root(str(tmp_path)) == tmp_path.resolve()
    with pytest.raises(ValueError, match="absolute"):
        normalize_recording_root("relative/demo")
    with pytest.raises(ValueError, match="filesystem root"):
        normalize_recording_root("/")


def test_delete_removes_only_the_most_recent_timestamp_directory(tmp_path):
    root = tmp_path / "recordings"
    saved = root / "2026-09-11_16-30-00.123456"
    saved.mkdir(parents=True)
    (saved / "demo.mp4").write_bytes(b"video")
    service = DemoService(str(root))
    service.result = {"path": str(saved), "frames": 10}

    status = service.command("delete")

    assert not saved.exists()
    assert status["result"] is None
    assert status["deleted_path"] == str(saved)
    with pytest.raises(ValueError, match="no recently saved"):
        service.command("delete")


def test_delete_refuses_a_result_outside_current_root(tmp_path):
    root = tmp_path / "recordings"
    outside = tmp_path / "other" / "2026-09-11_16-30-00.123456"
    outside.mkdir(parents=True)
    service = DemoService(str(root))
    service.result = {"path": str(outside)}

    with pytest.raises(ValueError, match="outside"):
        service.command("delete")
    assert outside.exists()


def test_rgb_video_and_original_ros_messages_round_trip(tmp_path):
    rosbag = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.utilities import get_message

    session = DemoSession(tmp_path, {"serial": "test", "streams": ["color"]})
    expected = {}
    for topic, type_name in SENSOR_TOPICS.items():
        expected[topic] = serialize_message(get_message(type_name)())
        assert session.enqueue(topic, (1_000_000_000, expected[topic]))
        now = time.time_ns()
        session.observe(topic, SimpleNamespace(timestamp_ns=now, **{
            key: value for name in ("ego", "left_wrist", "right_wrist")
            for key, value in ((name + "_timestamp_ns", now), (name + "_valid", True), (name, b"jpeg"))
        }), now)
    rgb = np.full((480, 640, 3), (45, 130, 190), np.uint8)
    metrics, planes = measurements().snapshot(1_010_000_000)
    for i in range(3):
        assert session.enqueue("d435", (rgb, metrics, planes, {"color_frame_number": i}))
    result = session.stop()
    path = Path(result["path"])
    assert result["complete"] is True
    assert result["frames"] == 3
    assert len((path / "frames.jsonl").read_text().splitlines()) == 3
    assert not (path / "depth").exists()
    assert not (path / "imu.jsonl").exists()
    for filename, width, height in (("d435_rgb.mp4", 640, 480), ("demo.mp4", 640, 480)):
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path / filename)]))
        stream = probe["streams"][0]
        assert (stream["width"], stream["height"], int(stream["nb_frames"])) == (width, height, 3)
    reader = rosbag.SequentialReader()
    reader.open(rosbag.StorageOptions(uri=str(path / "sensors"), storage_id="sqlite3"),
                rosbag.ConverterOptions("cdr", "cdr"))
    actual = {}
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        actual[topic] = data
        assert stamp == 1_000_000_000
    assert actual == expected


def test_overflow_and_missing_sensors_cannot_be_reported_complete(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class SlowBag:
        def write(self, *_):
            entered.set()
            assert release.wait(3)

    session = DemoSession(tmp_path, {}, bag_factory=lambda _: SlowBag(), queue_size=1)
    session.enqueue("/ur/state", (1, b"data"))
    assert entered.wait(1)
    assert session.enqueue("/ur/state", (2, b"data"))
    assert not session.enqueue("/ur/state", (3, b"data"))
    release.set()
    result = session.stop()
    assert result["complete"] is False
    assert result["dropped"] == {"/ur/state": 1}
    assert result["sensor_messages"]["/ur/state"] == 2
    assert "/sharpa/state" in result["missing_topics"]


def test_overlay_preserves_camera_and_builds_two_by_six_side_panel():
    rgb = np.full((480, 640, 3), (45, 130, 190), np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)
    metrics, planes = measurements().snapshot(1_010_000_000)
    torque_history = TorqueHistory()
    torque_history.update(metrics)
    image = np.asarray(render_overlay(rgb, metrics, planes, torque_history))
    assert image.shape == (480, 640, 3)
    # The 160-pixel panel displaces an equal 80 pixels from both RGB edges.
    assert np.array_equal(image[100, 100], rgb[100, 180])
    assert np.std(image[:, 480:]) > 10
    assert len(torque_history.series("left")[0][1]) == 5
    assert len(torque_history.series("right")[0][1]) == 5


def test_acquisition_timing_slows_playback_instead_of_accelerating_drops():
    duration, scale = acquisition_timing(803, 30, 1_000_000_000, 36_613_333_333)
    assert duration == pytest.approx(35.646666666)
    assert scale == pytest.approx(duration / (803 / 30))
    assert 803 / duration == pytest.approx(22.526, rel=1e-3)


def test_demo_disabled_ignores_sensors_and_disabling_saves_recording(tmp_path, monkeypatch):
    service = DemoService(str(tmp_path))
    assert service.status()['enabled'] is False
    monkeypatch.setattr(service.history, 'update', lambda *a: pytest.fail('disabled sensor work'))
    service.sensor('/sharpa/state', object(), 123, lambda _: pytest.fail('disabled serialization'))
    service.command('enable')
    assert service.status()['enabled'] is True
    finished = threading.Event()
    class Session:
        started_ns = time.time_ns()
        frames = 0
        dropped = {}
        errors = []
        def stop(self):
            finished.set()
            return {'path': str(tmp_path / 'saved'), 'complete': True}
    service.session = Session()
    result = service.command('disable')
    assert result['enabled'] is False and result['connected'] is False
    assert finished.wait(1)
    service.save_thread.join(1)
    assert service.result['complete']
    assert service.session is None


def test_demo_does_not_open_camera_until_enabled_and_closes_on_disable(tmp_path, monkeypatch):
    import sys
    from ur_common import drivers
    monkeypatch.setattr(drivers, 'ensure_optional_runtime_paths', lambda: None)
    opened, stopped = threading.Event(), threading.Event()
    attempts = []
    intrinsics = SimpleNamespace(width=64, height=36, fx=1, fy=1, ppx=0, ppy=0, coeffs=[], model='none')
    profile = SimpleNamespace(get_stream=lambda _: SimpleNamespace(as_video_stream_profile=lambda:
        SimpleNamespace(get_intrinsics=lambda: intrinsics)))
    class Pipeline:
        def start(self, config): opened.set(); return profile
        def wait_for_frames(self, timeout):
            time.sleep(.01)
            return SimpleNamespace(get_color_frame=lambda: None)
        def stop(self): stopped.set()
    def context():
        attempts.append('context')
        return SimpleNamespace(query_devices=lambda: [SimpleNamespace(get_info=lambda key: 'D435' if key == 'name' else 'demo')])
    rs = SimpleNamespace(context=context, pipeline=Pipeline,
        config=lambda: SimpleNamespace(enable_device=lambda *a: None, enable_stream=lambda *a: None),
        camera_info=SimpleNamespace(name='name', serial_number='serial'),
        stream=SimpleNamespace(color='color'), format=SimpleNamespace(rgb8='rgb8'))
    monkeypatch.setitem(sys.modules, 'pyrealsense2', rs)
    service = DemoService(str(tmp_path), width=64, height=36)
    service.thread.start()
    try:
        assert not opened.wait(.05)
        assert attempts == []
        service.command('enable')
        assert opened.wait(1)
        service.command('disable')
        assert stopped.wait(1)
        assert not service.capture_enabled.is_set()
        time.sleep(.05)
        assert attempts == ['context']
    finally:
        service.close()
    assert not service.thread.is_alive()
