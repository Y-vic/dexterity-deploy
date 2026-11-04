import json
from pathlib import Path
import struct
import time

from ur_common import acquisition_diagnostics as diag
from ur_common.usb_traffic import aggregate_header, atomic_json
from ur_common.drivers import _ThreadedCamera


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_usb_topology_speed_and_counter_reset(tmp_path):
    device = tmp_path / "sys/bus/usb/devices/2-3"
    for name, value in {"idVendor": "8086", "speed": "5000", "devnum": "4", "urbnum": "20", "serial": "camera"}.items():
        write(device / name, value)
    probe = diag.HostProbe(tmp_path / "sys", tmp_path / "proc", tmp_path / "absent")
    first = probe.sample()
    assert first["usb_devices"][0]["speed"] == "5000"
    assert first["usb_devices"][0]["urb_counter_delta"] is None
    write(device / "urbnum", "35")
    assert probe.sample()["usb_devices"][0]["urb_counter_delta"] == 15
    write(device / "devnum", "5")
    write(device / "urbnum", "2")
    assert probe.sample()["usb_devices"][0]["urb_counter_delta"] is None
    assert not first["usbmon"]["available"]


def test_cpu_iowait_and_optional_usbmon_freshness(tmp_path):
    probe = diag.HostProbe(tmp_path / "sys", tmp_path / "proc", tmp_path / "usb.json")
    write(tmp_path / "proc/stat", "cpu 10 0 10 100 5 0 0 0\n")
    probe.sample()
    write(tmp_path / "proc/stat", "cpu 20 0 20 150 35 0 0 0\n")
    atomic_json(tmp_path / "usb.json", {"monotonic_ns": time.monotonic_ns()})
    second = probe.sample()
    assert second["cpu"]["cpu"] == {"busy_percent": 20, "iowait_percent": 30}
    assert second["usbmon"]["available"]
    atomic_json(tmp_path / "usb.json", {"monotonic_ns": time.monotonic_ns()-10_000_000_000})
    assert not probe.sample()["usbmon"]["available"]


def test_usbmon_counts_completions_not_submissions_and_never_payload():
    counters = {}
    header = bytearray(64)
    struct.pack_into("=BBBBH", header, 8, ord("S"), 3, 0x81, 4, 2)
    struct.pack_into("=I", header, 32, 1024)
    aggregate_header(header, counters)
    assert counters["2:4:1:in:3"]["actual_bytes"] == 0
    header[8] = ord("C")
    aggregate_header(header, counters)
    assert counters["2:4:1:in:3"]["actual_bytes"] == 1024
    struct.pack_into("=i", header, 28, -71)
    aggregate_header(header, counters)
    assert counters["2:4:1:in:3"]["errors"] == 1
    header[9] = 0
    struct.pack_into("=i", header, 40, 2)
    aggregate_header(header, counters)
    assert counters["2:4:1:in:0"]["iso_packet_errors"] == 2


def test_continuous_log_includes_idle_and_recording_without_kernel_permission(tmp_path, monkeypatch):
    class Kernel:
        def snapshot(self):
            return {"stderr": ["permission denied"], "events": []}
        def close(self):
            pass
    monkeypatch.setattr(diag, "KernelUsbLog", Kernel)
    monkeypatch.setattr(diag.HostProbe, "sample", lambda self: {"usb_devices": []})
    camera = _ThreadedCamera()
    camera.record_raw_publish(100, 110, 200, 0.5)
    monitor = diag.AcquisitionMonitor(
        lambda: {'ego': camera.diagnostics(include_frame_timings=True)}, tmp_path, interval=.01)
    try:
        deadline = time.monotonic()+2
        while not monitor.latest() and time.monotonic() < deadline:
            time.sleep(.005)
        assert monitor.latest()["recording"]["recording"] is False
        monitor.recording_status({"recording": True, "episode_id": "test"})
        deadline = time.monotonic()+2
        while not monitor.latest()["recording"].get("recording") and time.monotonic() < deadline:
            time.sleep(.005)
        assert monitor.latest()["recording"]["recording"] is True
    finally:
        monitor.close()
    rows = [json.loads(line) for line in monitor.path.read_text().splitlines()]
    assert {r["recording"]["recording"] for r in rows} == {False, True}
    assert rows[-1]["kernel_usb"]["stderr"] == ["permission denied"]
    events = [e for row in rows for e in row['cameras']['ego']['frame_timing_events']]
    assert events == [dict(stage='raw_published', source_timestamp_ns=100,
                           at_ns=110, monotonic_ns=200, pack_write_ms=0.5)]
