from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from ur_common import drivers


def test_zed_retrieve_failure_does_not_encode_previous_image():
    camera = drivers.ZedEgoCamera()
    camera.sl = SimpleNamespace(ERROR_CODE=SimpleNamespace(SUCCESS=0),
                                VIEW=SimpleNamespace(LEFT=1))
    camera.runtime = object()
    camera.image = SimpleNamespace(get_data=lambda: pytest.fail("must not read stale pixels"))
    camera.camera = SimpleNamespace(grab=lambda runtime: 0,
                                    retrieve_image=lambda image, view: 17)
    with pytest.raises(drivers.HardwareConnectionError, match="retrieve_image failed: 17"):
        camera._grab_rgb()


def test_camera_diagnostics_distinguishes_queue_losses():
    camera = drivers._ThreadedCamera()
    camera.dropped_raw_count = 2
    camera.dropped_encoded_count = 3
    camera.capture_error_count = 1
    counters = camera.diagnostics()
    assert counters["dropped_raw_count"] == 2
    assert counters["dropped_encoded_count"] == 3
    assert counters["capture_error_count"] == 1
    assert counters["raw_queue_depth"] == counters["encoded_queue_depth"] == 0


class _FakeFrame:
    def __init__(self, data=None):
        self._data = data

    def get_data(self):
        return self._data


class _FakeFrames:
    def __init__(self, color_frame):
        self._color_frame = color_frame

    def get_color_frame(self):
        return self._color_frame


class _FakePipeline:
    def __init__(self, raw_frame, wait_error=None):
        self.raw_frame = raw_frame
        self.wait_error = wait_error
        self.started_with = None
        self.stopped = False
        self.wait_timeout_ms = None

    def start(self, config):
        self.started_with = config

    def wait_for_frames(self, timeout_ms):
        self.wait_timeout_ms = timeout_ms
        if self.wait_error is not None:
            raise self.wait_error
        return _FakeFrames(self.raw_frame)

    def stop(self):
        self.stopped = True


class _FakeConfig:
    def __init__(self):
        self.serial = None
        self.stream = None

    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, *stream):
        self.stream = stream


def test_realsense_wrist_uses_bgr8_without_redundant_color_conversion(monkeypatch):
    bgr = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    raw_frame = _FakeFrame(bgr)
    pipeline = _FakePipeline(raw_frame)
    config = _FakeConfig()
    fake_rs = SimpleNamespace(
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
        pipeline=lambda: pipeline,
        config=lambda: config,
    )
    monkeypatch.setattr(drivers, "ensure_optional_runtime_paths", lambda: ())
    monkeypatch.setattr(
        drivers.importlib,
        "import_module",
        lambda name: fake_rs if name == "pyrealsense2" else None,
    )

    camera = drivers.RealSenseWristCamera(
        serial_number="wrist-serial",
        height=2,
        width=3,
        frequency_hz=30,
    )
    camera.connect()

    assert config.serial == "wrist-serial"
    assert config.stream == ("color", 3, 2, "bgr8", 30)
    assert np.array_equal(camera._grab_rgb(), bgr)
    assert pipeline.wait_timeout_ms == 5000
    assert camera._frame_timestamp_ns > 0

    camera.close()
    assert pipeline.stopped


def test_realsense_wrist_keeps_pipeline_after_timeout(monkeypatch):
    bgr = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    raw_frame = _FakeFrame(bgr)
    pipeline = _FakePipeline(raw_frame, RuntimeError("Frame didn't arrive"))

    fake_rs = SimpleNamespace(
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
        pipeline=lambda: pipeline,
        config=_FakeConfig,
    )
    monkeypatch.setattr(drivers, "ensure_optional_runtime_paths", lambda: ())
    monkeypatch.setattr(
        drivers.importlib,
        "import_module",
        lambda name: fake_rs if name == "pyrealsense2" else None,
    )

    camera = drivers.RealSenseWristCamera(
        serial_number="wrist-serial",
        height=2,
        width=3,
        frequency_hz=30,
    )
    camera.connect()

    with pytest.raises(drivers.HardwareConnectionError, match="capture timed out"):
        camera._grab_rgb()

    assert not pipeline.stopped
    pipeline.wait_error = None
    assert np.array_equal(camera._grab_rgb(), bgr)
    camera.close()
    assert pipeline.stopped


def test_threaded_camera_queues_frames_without_blocking_capture(monkeypatch):
    encode_started = threading.Event()
    release_encode = threading.Event()

    def slow_encode(rgb, quality):
        encode_started.set()
        release_encode.wait(timeout=1.0)
        return bytes((int(rgb[0, 0, 0]), quality))

    class FastCamera(drivers._ThreadedCamera):
        def _grab_rgb(self):
            if self.capture_count >= 8:
                self._stop.wait(0.001)
                return None
            return np.full((2, 2, 3), self.capture_count, dtype=np.uint8)

    monkeypatch.setattr(drivers, "_encode_jpeg", slow_encode)
    camera = FastCamera(jpeg_quality=90, process_encode=False)
    camera.start()
    assert encode_started.wait(timeout=1.0)

    deadline = time.monotonic() + 1.0
    while camera.capture_count < 8 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert camera.capture_count == 8
    assert camera.encoded_count == 0
    # Inference already has the newest capture while JPEG is still blocked.
    stamp, raw = camera.read_raw_latest()
    assert stamp > 0
    np.testing.assert_array_equal(raw, np.full((2, 2, 3), 7, dtype=np.uint8))
    assert camera.dropped_raw_count == 0
    release_encode.set()
    deadline = time.monotonic() + 1.0
    while camera.encoded_count < 8 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert camera.encoded_count == 8
    assert camera.read() is not None
    assert len(camera.read_pending()) == 8
    camera.close()


def test_threaded_camera_drops_oldest_frame_when_queue_is_full(monkeypatch):
    encode_started = threading.Event()
    release_encode = threading.Event()

    def slow_encode(rgb, quality):
        encode_started.set()
        release_encode.wait(timeout=1.0)
        return bytes((int(rgb[0, 0, 0]), quality))

    class FastCamera(drivers._ThreadedCamera):
        def _grab_rgb(self):
            if self.capture_count >= 6:
                self._stop.wait(0.001)
                return None
            return np.full((2, 2, 3), self.capture_count, dtype=np.uint8)

    monkeypatch.setattr(drivers, "_encode_jpeg", slow_encode)
    camera = FastCamera(jpeg_quality=90, raw_queue_size=2, process_encode=False)
    camera.start()
    assert encode_started.wait(timeout=1.0)

    deadline = time.monotonic() + 1.0
    while camera.capture_count < 6 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert camera.capture_count == 6
    assert camera.dropped_raw_count >= 3
    release_encode.set()
    expected_encoded = camera.capture_count - camera.dropped_raw_count
    deadline = time.monotonic() + 1.0
    while camera.encoded_count < expected_encoded and time.monotonic() < deadline:
        time.sleep(0.001)
    assert camera.encoded_count == expected_encoded
    assert camera.encoded_count + camera.dropped_raw_count == camera.capture_count
    camera.close()


def test_parallel_encoders_publish_every_frame_in_capture_order(monkeypatch):
    release_first = threading.Event()
    second_finished = threading.Event()

    def out_of_order_encode(rgb, quality):
        value = int(rgb[0, 0, 0])
        if value == 0:
            release_first.wait(timeout=1.0)
        else:
            second_finished.set()
        return bytes((value, quality))

    class TwoFrameCamera(drivers._ThreadedCamera):
        def _grab_rgb(self):
            if self.capture_count >= 2:
                self._stop.wait(0.001)
                return None
            return np.full((2, 2, 3), self.capture_count, dtype=np.uint8)

    monkeypatch.setattr(drivers, "_encode_jpeg", out_of_order_encode)
    camera = TwoFrameCamera(
        jpeg_quality=90,
        process_encode=False,
        encode_workers=2,
    )
    camera.start()
    assert second_finished.wait(timeout=1.0)
    # Sequence 1 completed first, but it must wait for sequence 0 instead of
    # replacing it in a latest-frame slot.
    assert camera.read() is None

    release_first.set()
    deadline = time.monotonic() + 1.0
    while camera.encoded_count < 2 and time.monotonic() < deadline:
        time.sleep(0.001)

    frames = camera.read_pending()
    assert [frame.jpeg[0] for frame in frames] == [0, 1]
    assert camera.read_pending() == []
    camera.close()
