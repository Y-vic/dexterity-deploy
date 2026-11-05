from types import SimpleNamespace

from camera_node.camera_node import (
    CameraSetCore,
    CompleteCameraSetGate,
    FrameProgressWatchdog,
    _camera_frame,
    zed_device_diagnostic,
)


def frame(timestamp_ns):
    return timestamp_ns, b"jpeg"


def test_complete_set_waits_for_every_view_to_advance():
    gate = CompleteCameraSetGate()
    first = CameraSetCore.make(
        {name: frame(10) for name in CameraSetCore.NAMES}, now_ns=10
    )
    repeated_wrist = CameraSetCore.make(
        {"ego": frame(20), "left_wrist": frame(20), "right_wrist": frame(10)},
        now_ns=20,
    )
    second = CameraSetCore.make(
        {name: frame(20) for name in CameraSetCore.NAMES}, now_ns=20
    )

    assert gate.accept(first)
    assert not gate.accept(repeated_wrist)
    assert gate.accept(second)


def test_incomplete_set_is_published_for_status_visibility():
    gate = CompleteCameraSetGate()
    incomplete = CameraSetCore.make(
        {"ego": frame(10), "left_wrist": frame(10), "right_wrist": None},
        now_ns=10,
    )

    assert gate.accept(incomplete)


def test_frame_progress_watchdog_detects_no_frame_and_stale_frame():
    watchdog = FrameProgressWatchdog(timeout_s=3.0)
    watchdog.reset(now=10.0)

    assert not watchdog.stalled(0, now=12.9)
    assert watchdog.stalled(0, now=13.0)

    watchdog.reset(now=20.0, timestamp_ns=100)
    assert not watchdog.stalled(200, now=22.0)
    assert not watchdog.stalled(200, now=24.9)
    assert watchdog.stalled(200, now=25.0)


def test_hardware_ego_never_falls_back_to_file(tmp_path):
    source = tmp_path / "ego.jpg"
    source.write_bytes(b"not-zed")

    assert _camera_frame("ego", None, str(source), use_hardware=True) is None
    assert _camera_frame("ego", None, str(source), use_hardware=False)[1] == b"not-zed"


def test_zed_diagnostic_explains_unavailable_main_video_interface():
    device = SimpleNamespace(
        camera_model="ZED-M",
        camera_state="NOT AVAILABLE",
        serial_number=0,
        path="",
    )
    sl = SimpleNamespace(
        Camera=SimpleNamespace(
            get_sdk_version=lambda: "4.2.5",
            get_device_list=lambda: [device],
        )
    )

    diagnostic = zed_device_diagnostic(SimpleNamespace(sl=sl))

    assert "SDK 4.2.5" in diagnostic
    assert "state=NOT AVAILABLE" in diagnostic
    assert "serial=0" in diagnostic
    assert "path=<empty>" in diagnostic
    assert "MCU/HID entry alone cannot provide video" in diagnostic
    assert "USB 3.x" in diagnostic
    assert "no fallback ego camera" in diagnostic
