"""Exercise the real spawn/pipe boundary without opening any hardware."""

import ctypes
import multiprocessing
import os
import time

import numpy as np
import pytest

from ur_common import isolated_camera
from ur_common.drivers import HardwareConnectionError, HardwareDependencyError


class SyntheticCamera:
    def __init__(self, **options):
        self.options = options
        self.sequence = 0

    def connect(self):
        if self.options.get("open_error"):
            raise HardwareDependencyError("synthetic missing SDK")
        if self.options.get("hang_open"):
            time.sleep(30)

    def _grab_frame(self):
        started = time.monotonic_ns()
        entered = self.options.get("entered")
        if entered is not None:
            entered.set()
            # Like pyzed 4.2 grab(), PyDLL holds the interpreter lock throughout
            # this blocking native call. It must not hold the parent's GIL.
            ctypes.PyDLL(None).usleep(500_000)
        if self.options.get("exit_capture"):
            os._exit(7)
        if self.options.get("capture_error") and self.sequence == 0:
            self.sequence += 1
            raise RuntimeError("synthetic grab failed")
        self.sequence += 1
        image = np.full((270, 480, 3), self.sequence % 256, np.uint8)
        if self.options.get("bad_pixels"):
            image = image.astype(np.float32)
        return image, 1_000_000_000 + self.sequence, started, time.monotonic_ns()

    def close(self):
        pass


@pytest.fixture
def camera(monkeypatch):
    original_import = isolated_camera.importlib.import_module
    monkeypatch.setattr(isolated_camera.importlib, "import_module",
                        lambda name: object() if name == "pyzed.sl" else original_import(name))
    monkeypatch.setattr(isolated_camera, "ensure_optional_runtime_paths", lambda: ())
    result = isolated_camera.IsolatedZedEgoCamera()
    result._camera_factory = SyntheticCamera
    yield result
    result.close()


def test_spawn_preserves_pixels_source_timestamps_and_order_across_reconnect(camera):
    for _ in range(2):
        camera.connect()
        assert camera.diagnostics()["capture_process_pid"] != os.getpid()
        assert camera.diagnostics()["capture_process_alive"]
        for sequence in range(1, 4):
            image, stamp, started, ended = camera._grab_frame()
            assert stamp == 1_000_000_000 + sequence
            assert 0 < started <= ended <= time.monotonic_ns()
            np.testing.assert_array_equal(image, np.full((270, 480, 3), sequence, np.uint8))
        camera.close()
        assert not camera.diagnostics()["capture_process_alive"]
        assert camera.read_raw_latest() is None


def test_blocking_sdk_call_does_not_block_parent_python(camera):
    entered = multiprocessing.get_context("spawn").Event()
    camera._capture_options = {"entered": entered}
    camera.connect()
    assert entered.wait(2)
    # Python in the ROS/publisher process remains runnable during the grab.
    started = time.monotonic()
    for _ in range(10):
        time.sleep(0.005)
    assert time.monotonic() - started < 0.3
    image, stamp, _, _ = camera._grab_frame()
    assert stamp == 1_000_000_001 and image[0, 0, 0] == 1


def test_worker_failure_never_reuses_a_previous_frame(camera):
    camera._capture_options = {"exit_capture": True}
    camera.connect()
    with pytest.raises(HardwareConnectionError, match="disconnected|exited"):
        camera._grab_frame()
    assert camera.read_raw_latest() is None


def test_shared_slot_reuse_never_changes_previously_received_pixels(camera):
    camera.connect()
    images = [camera._grab_frame()[0] for _ in range(25)]
    for sequence, pixels in enumerate(images, 1):
        assert np.all(pixels == sequence)
    assert camera.diagnostics()["capture_transport"] == "acknowledged_shared_memory"


def test_invalid_pixels_are_reported_without_publishing_a_frame(camera):
    camera._capture_options = {"bad_pixels": True}
    camera.connect()
    with pytest.raises(HardwareConnectionError, match="invalid or oversized"):
        camera._grab_frame()


def test_transient_capture_error_is_reported_and_next_frame_is_preserved(camera):
    camera._capture_options = {"capture_error": True}
    camera.connect()
    with pytest.raises(HardwareConnectionError, match="synthetic grab failed"):
        camera._grab_frame()
    assert camera._grab_frame()[1] == 1_000_000_002


@pytest.mark.parametrize("options, error, message", [
    ({"open_error": True}, HardwareDependencyError, "synthetic missing SDK"),
    ({"hang_open": True}, HardwareConnectionError, "timed out"),
])
def test_failed_connect_cleans_up_child(camera, options, error, message):
    camera._capture_options = options
    camera._connect_timeout_s = 1.0
    with pytest.raises(error, match=message):
        camera.connect()
    assert camera._capture_process is None
    assert camera._capture_connection is None


def test_close_unblocks_full_frame_pipe_and_capture_thread(camera):
    camera.connect()
    # With no consumer the bounded pipe fills; shutdown must still reap child.
    process = camera._capture_process
    started = time.monotonic()
    camera.close()
    assert time.monotonic() - started < 3
    assert process._closed
    camera.connect()
    camera.configure_raw_recording()
    camera.start()
    deadline = time.monotonic() + 2
    while camera.read_raw_latest() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert camera.read_raw_latest()[0] > 0
    camera.close()
    assert camera._capture_thread is None
    assert camera.read_raw_pending() == []
