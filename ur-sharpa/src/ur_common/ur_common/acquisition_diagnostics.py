"""Low-rate host/USB evidence, independent of camera capture and ROS callbacks."""

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


class HostProbe:
    def __init__(self, sys_root="/sys", proc_root="/proc",
                 usbmon_path="/tmp/ur_sharpa_usbmon.json"):
        self.sys = Path(sys_root)
        self.proc = Path(proc_root)
        self.usbmon_path = Path(usbmon_path)
        self._previous_usb = {}
        self._previous_cpu = {}
        self._previous_disks = {}
        self._previous_net = {}
        self._previous_at = None

    def sample(self):
        now = time.monotonic()
        elapsed = None if self._previous_at is None else now - self._previous_at
        usb = []
        current_usb = {}
        for path in sorted((self.sys / "bus/usb/devices").glob("*")):
            if not (path / "idVendor").exists():
                continue
            row = {key: read_text(path / key) for key in (
                "product", "manufacturer", "serial", "idVendor", "idProduct",
                "speed", "busnum", "devnum", "urbnum", "power/runtime_status")}
            row["port"] = path.name
            row["sysfs_path"] = str(path.resolve())
            row["host_controller"] = next((p for p in reversed(path.resolve().parts)
                                            if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", p)), None)
            key = (path.name, row["devnum"], row["serial"])
            count = int(row["urbnum"]) if row["urbnum"] and row["urbnum"].isdigit() else None
            previous = self._previous_usb.get(key)
            row["urb_counter_delta"] = (count - previous if count is not None and previous is not None
                                          and count >= previous else None)
            # URB counts are not transferred bytes or bus utilization.
            current_usb[key] = count
            usb.append(row)
        cpu = {}
        current_cpu = {}
        for line in (read_text(self.proc / "stat") or "").splitlines():
            fields = line.split()
            if not fields or not re.fullmatch(r"cpu\d*", fields[0]):
                continue
            values = list(map(int, fields[1:9]))
            total, idle = sum(values), values[3]
            wait = values[4] if len(values) > 4 else 0
            current_cpu[fields[0]] = (total, idle, wait)
            previous = self._previous_cpu.get(fields[0])
            if previous and total > previous[0]:
                delta = total - previous[0]
                cpu[fields[0]] = {"busy_percent": 100 * (delta - (idle-previous[1]) - (wait-previous[2])) / delta,
                                  "iowait_percent": 100 * (wait-previous[2]) / delta}
        disks = {}
        current_disks = {}
        for line in (read_text(self.proc / "diskstats") or "").splitlines():
            fields = line.split()
            if len(fields) < 14 or fields[2].startswith(("loop", "ram")):
                continue
            name = fields[2]
            values = (int(fields[5])*512, int(fields[9])*512, int(fields[12]))
            current_disks[name] = values
            previous = self._previous_disks.get(name)
            if previous and elapsed and all(a >= b for a, b in zip(values, previous)):
                disks[name] = {"read_bytes_s": (values[0]-previous[0])/elapsed,
                               "write_bytes_s": (values[1]-previous[1])/elapsed,
                               "io_busy_ms": values[2]-previous[2],
                               "sysfs_path": str((self.sys / "class/block" / name).resolve())}
        network = {}
        current_net = {}
        for path in (self.sys / "class/net").glob("*"):
            values = {key: int(read_text(path / "statistics" / key) or 0)
                      for key in ("rx_bytes", "tx_bytes", "rx_errors", "tx_errors", "rx_dropped", "tx_dropped")}
            current_net[path.name] = values
            old = self._previous_net.get(path.name)
            network[path.name] = {"counters": values, "sysfs_device": str((path / "device").resolve())}
            if old and elapsed:
                network[path.name]["bytes_per_second"] = {
                    key: max(0, values[key]-old[key])/elapsed for key in ("rx_bytes", "tx_bytes")}
        usbmon = {"available": False, "reason": "optional USB payload monitor not running"}
        try:
            if self.usbmon_path.stat().st_size > 1024 * 1024:
                raise ValueError("USB monitor summary exceeds size limit")
            monitor = json.loads(self.usbmon_path.read_text())
            age = (time.monotonic_ns() - int(monitor["monotonic_ns"])) / 1e9
            usbmon = {"available": 0 <= age <= 5, "age_s": age, "summary": monitor}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            usbmon["reason"] = str(exc)
        result = {"sample_interval_s": elapsed, "usb_devices": usb, "cpu": cpu,
                  "loadavg": read_text(self.proc / "loadavg"),
                  "pressure": {name: read_text(self.proc / "pressure" / name) for name in ("cpu", "io", "memory")},
                  "memory": {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                             for line in (read_text(self.proc / "meminfo") or "").splitlines()
                             if line.startswith(("MemAvailable:", "Dirty:", "Writeback:"))},
                  "disk_io": disks, "network": network, "usbmon": usbmon}
        self._previous_at, self._previous_usb = now, current_usb
        self._previous_cpu, self._previous_disks = current_cpu, current_disks
        self._previous_net = current_net
        return result


class KernelUsbLog:
    """Best-effort kernel events; report permission errors, never run sudo."""
    def __init__(self):
        self._lock = threading.Lock()
        self._events = deque(maxlen=128)
        self._stderr = deque(maxlen=8)
        self.process = None
        self._threads = []
        self._overrun = 0
        try:
            self.process = subprocess.Popen(
                ["journalctl", "-k", "-f", "--since=-30 seconds", "--no-pager", "-o", "json"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for stream, error in ((self.process.stdout, False), (self.process.stderr, True)):
                thread = threading.Thread(target=self._read, args=(stream, error), daemon=True)
                thread.start()
                self._threads.append(thread)
        except OSError as exc:
            self._stderr.append(str(exc))

    def _read(self, stream, error):
        for line in stream:
            if error:
                with self._lock:
                    self._stderr.append(line.strip()[:1000])
                continue
            try:
                row = json.loads(line)
                message = str(row.get("MESSAGE", ""))
                if re.search(r"usb|xhci|uvcvideo|bandwidth", message, re.IGNORECASE):
                    with self._lock:
                        if len(self._events) == self._events.maxlen:
                            self._overrun += 1
                        self._events.append({"kernel_realtime_us": row.get("__REALTIME_TIMESTAMP"),
                                             "kernel_monotonic_us": row.get("__MONOTONIC_TIMESTAMP"),
                                             "message": message[:2000]})
            except (ValueError, TypeError):
                continue

    def snapshot(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return {"process_running": self.process is not None and self.process.poll() is None,
                    "stderr": list(self._stderr), "events": events,
                    "event_buffer_overruns": self._overrun}

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for thread in self._threads:
                thread.join(timeout=1)
            self.process.stdout.close()
            self.process.stderr.close()


class AcquisitionMonitor:
    """Continuous before/during/after recording log, sampled off the executor."""
    def __init__(self, cameras, directory="", *, interval=1.0):
        self._cameras = cameras
        self.interval = float(interval)
        if self.interval <= 0:
            raise ValueError("diagnostic interval must be positive")
        base = Path(directory).expanduser() if directory else Path(os.environ.get("ROS_LOG_DIR", str(Path.home()/".ros/log")))
        self.path = base / ("acquisition_" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}.jsonl")
        self.probe = HostProbe()
        self.kernel = KernelUsbLog()
        self._lock = threading.Lock()
        self._recording = {"recording": False}
        self._latest = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="acquisition-host-log", daemon=True)
        self._thread.start()

    def recording_status(self, status):
        with self._lock:
            self._recording = {key: status.get(key) for key in
                               ("recording", "saving", "episode_id", "recording_root", "tactile_enabled", "video_encoding")}
            self._recording["observed_monotonic_ns"] = time.monotonic_ns()

    def latest(self):
        with self._lock:
            return dict(self._latest)

    def _run(self):
        previous = {}
        previous_at = None
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                cameras = self._cameras()
                if previous_at is not None:
                    for name, row in cameras.items():
                        old = previous.get(name, {})
                        for key in ("capture_count", "encoded_count"):
                            delta = row[key] - old.get(key, row[key])
                            row[key.replace("_count", "_rate_hz")] = delta/(started-previous_at) if delta >= 0 else None
                previous = {name: dict(row) for name, row in cameras.items()}
                previous_at = started
                with self._lock:
                    recording = dict(self._recording)
                payload = {"timestamp_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                           "cameras": cameras, "recording": recording, "host": self.probe.sample(),
                           "kernel_usb": self.kernel.snapshot(), "continuous_log": str(self.path)}
                payload["diagnostic_sample_ms"] = (time.monotonic() - started) * 1000
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    # Bounded continuous logs, retaining up to four segments.
                    if self.path.exists() and self.path.stat().st_size > 32 * 1024 * 1024:
                        for i in (2, 1):
                            old = Path(str(self.path) + f".{i}")
                            if old.exists():
                                old.replace(Path(str(self.path) + f".{i+1}"))
                        self.path.replace(Path(str(self.path) + ".1"))
                    with self.path.open("a") as handle:
                        handle.write(json.dumps(payload, allow_nan=False) + "\n")
                except OSError as exc:
                    payload["continuous_log_error"] = str(exc)
                with self._lock:
                    self._latest = payload
            except Exception as exc:
                with self._lock:
                    self._latest = {"timestamp_ns": time.time_ns(), "diagnostic_error": str(exc)}
            self._stop.wait(max(0.0, self.interval - (time.monotonic()-started)))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self.kernel.close()
