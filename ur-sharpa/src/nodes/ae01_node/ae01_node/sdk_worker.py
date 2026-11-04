"""Isolated Avatar SDK reader. Never constructs a WaveService or sends hand commands."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from .feedback import ForceRelay

PACKET_PREFIX = "AE01_FRAME "
TRANSPORTS = {"wireless": "usb_serial", "wired": "ethernet_udp", "usb": "usb_vendor"}
RAW_IMU_FIELDS = ("imu_gyro_x", "imu_gyro_y", "imu_gyro_z", "imu_acc_x", "imu_acc_y", "imu_acc_z",
                  "imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z")


def prepare_config(path, transport):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or "backend" in config:
        raise ValueError("AE01 SDK config must be a hardware config object without backend")
    config.update(transport_link=TRANSPORTS[transport], enable_wave=False,
                  enable_wave_rk=False, wave_enabled=False, enable_localhost_glove=False)
    config.pop("transport_link_options", None)
    rate = config.get("casadi_common", {}).get("retarget_update_rate_hz", 0)
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("AE01 ROBOT requires casadi_common.retarget_update_rate_hz > 0")
    return path, config


def vendor_processes(proc_root=Path("/proc")):
    conflicts = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode().split("\0")
            if any(Path(arg).name in ("avatar-backend", "avatar_teleop.py", "avatar_example.py") for arg in args if arg):
                conflicts.append(entry.name)
        except (OSError, UnicodeError):
            continue
    return conflicts


def frame_stamp(hand):
    stamp = hand.header.stamp
    sec, ns = int(stamp.sec), int(stamp.nanosec)
    if sec < 0 or not 0 <= ns < 1_000_000_000:
        raise ValueError("invalid SDK frame stamp")
    return sec*1_000_000_000 + ns


class SdkReader:
    """Advance receive ages only when SDK source stamps advance, including stationary hands."""
    def __init__(self, api, config_path, config, clock=time.monotonic):
        self.api, self.clock = api, clock
        # A device init has its own transport defaults. Preserve the manager's
        # effective transport/retarget/resources instead of reapplying "{}".
        device_config = dict(config)
        device_config["enable_retarget"] = True
        self.device_config = json.dumps(device_config)
        self.sdk = api.AvatarSDK.get_instance()
        self.devices, self.cache = {}, {}
        self.initialized, self.started = set(), set()
        self.retry_after = {}
        self.on_release = None
        self.closed = False
        try:
            self._check(self.sdk.initialize(json.dumps(config), str(config_path)), "initialize")
        except Exception:
            self.sdk.destroy()
            raise

    def _check(self, code, operation):
        if code != self.api.ErrorCode.SUCCESS:
            raise RuntimeError(f"Avatar SDK {operation}: {code}")

    def _release(self, side, clear_cache=False):
        if self.on_release is not None:
            self.on_release(side)
        dev = self.devices.pop(side, None)
        started = side in self.started
        self.initialized.discard(side)
        self.started.discard(side)
        self.retry_after.pop(side, None)
        if clear_cache:
            self.cache.pop(side, None)
        elif side in self.cache:
            # Keep source watermarks, but require new frames after any disconnect/fault.
            self.cache[side].update(q=None, raw_seen_s=0.0, robot_seen_s=0.0)
        if dev is not None and started:
            # get_device() returns a manager-owned handle. Destroying it during
            # asynchronous identity discovery invalidates subsequent retries.
            dev.stop()

    def _frame(self, dev, category, kind, row):
        code, frame = dev.fetch_data(category)
        if code != self.api.ErrorCode.SUCCESS or frame is None:
            return
        hand = getattr(frame, kind)
        stamp = frame_stamp(hand)
        # fetch_data(ROBOT) returns a cache. Re-reading it is not a new input.
        if stamp <= row.get(kind + "_timestamp_ns", 0):
            return
        q = list(hand.joint.position)
        if kind == "raw" and len(q) == 32:
            # SDK 1.7.5 / 2.4.1A appends IMU fields to the RAW Joint payload.
            # These are sensor fields, never part of the 22-DOF robot target.
            expected = [f"joint_{i}" for i in range(22)] + list(RAW_IMU_FIELDS)
            if list(hand.joint.name) != expected:
                raise ValueError("AE01 32-field RAW frame has an unknown joint/IMU layout")
            q = q[:22]
        if len(q) != 22 or not all(math.isfinite(v) for v in q):
            raise ValueError(f"AE01 {kind} frame requires 22 finite joint values; received {len(q)}")
        row[kind + "_timestamp_ns"] = stamp
        row[kind + "_seen_s"] = self.clock()
        if kind == "robot":
            row.update(q=q, names=list(hand.joint.name))

    def poll(self):
        infos = list(self.sdk.list_device_info())
        sides = {}
        for side, enum in (("left", self.api.DeviceSide.LEFT), ("right", self.api.DeviceSide.RIGHT)):
            rows = [d for d in infos if d.device_type == self.api.DeviceType.GLOVE and d.device_side == enum and d.online]
            if len(rows) != 1:
                self._release(side)
                sides[side] = dict(online=False, error="同侧发现多个 AE01 手套" if rows else "手套未在线")
                continue
            info = rows[0]
            row = self.cache.setdefault(side, {})
            if row.get("sn", info.sn) != info.sn:
                self._release(side, clear_cache=True)
                row = self.cache.setdefault(side, {})
            row.update(online=True, sn=info.sn, hz=float(info.hz), robot_hz=float(info.robot_hz),
                       ready=bool(info.retarget_bootstrap_ready), fault_code=int(info.error_code), error="")
            try:
                # Device diagnostics contain faults beyond the compatibility summary code.
                faults = {int(k) for k in info.diagnostics if int(k)}
                if info.error_code:
                    faults.add(int(info.error_code))
                if faults:
                    raise RuntimeError("AE01 fault: " + ", ".join(f"0x{f:04X}" for f in sorted(faults)))
                if side not in self.devices:
                    dev = self.sdk.get_device(self.api.DeviceType.GLOVE, enum)
                    if dev is None:
                        raise RuntimeError("SDK glove handle unavailable")
                    self.devices[side] = dev
                dev = self.devices[side]
                if side not in self.started and self.clock() < self.retry_after.get(side, 0):
                    row["error"] = "等待 SDK 完成手套初始化"
                    sides[side] = dict(row)
                    continue
                if side not in self.initialized:
                    self.retry_after[side] = self.clock() + .5
                    self._check(dev.init(self.device_config), f"{side}.init")
                    self.initialized.add(side)
                if side not in self.started:
                    self.retry_after[side] = self.clock() + .5
                    self._check(dev.start(), f"{side}.start")
                    self.started.add(side)
                self._frame(dev, self.api.DeviceDataCategery.RAW, "raw", row)
                self._frame(dev, self.api.DeviceDataCategery.ROBOT, "robot", row)
            except Exception as exc:
                row.update(q=None, raw_seen_s=0.0, robot_seen_s=0.0, error=str(exc))
            sides[side] = dict(row)
        return dict(monotonic_s=self.clock(), sides=sides)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            for side in list(self.devices):
                try:
                    self._release(side)
                except Exception:
                    pass  # Release every handle even if one SDK stop/destroy fails.
        finally:
            self.sdk.destroy()


def emit(packet):
    print(PACKET_PREFIX + json.dumps(packet, separators=(",", ":"), allow_nan=False), flush=True)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", default="/opt/avatar-sdk")
    parser.add_argument("--config", required=True)
    parser.add_argument("--transport", choices=TRANSPORTS, default="wireless")
    parser.add_argument("--parent-pid", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--feedback', action='store_true', help='Receive gated force data from the ROS parent')
    parser.add_argument('--feedback-max-age-s', type=float, default=.25)
    parser.add_argument('--feedback-future-tolerance-s', type=float, default=.05)
    opts = parser.parse_args(args)
    reader = None
    try:
        if opts.parent_pid:
            # Do not leave a serial-owning SDK child behind if the ROS process dies.
            if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
                raise RuntimeError("cannot register AE01 parent-death signal")
            if os.getppid() != opts.parent_pid:
                raise RuntimeError("AE01 ROS parent exited before SDK startup")
        # A second process must not compete with Avatar for the serial receiver.
        with open(f"/tmp/ur_sharpa_ae01_{os.getuid()}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            conflicts = vendor_processes()
            if conflicts:
                raise RuntimeError("请先关闭 Sharpa Avatar / avatar-backend / SDK 示例；占用进程 PID: " + ", ".join(conflicts))
            path, config = prepare_config(opts.config, opts.transport)
            python_root = Path(opts.sdk_root).expanduser().resolve() / "lib/python"
            if not (python_root / "avatar_sdk/__init__.py").is_file():
                raise FileNotFoundError(f"Avatar SDK Python package missing at {python_root}")
            sys.path.insert(0, str(python_root))
            import avatar_sdk as api
            native_log = os.environ.get("AE01_SDK_LOG_PATH", "")
            api.setup_cpp_logging(native_log, False, "DEBUG" if native_log else "WARNING")
            running = True

            def stop(_sig, _frame):
                nonlocal running
                running = False

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            reader = SdkReader(api, path, config)
            relay = None
            try:
                if opts.feedback:
                    relay = ForceRelay(api, opts.feedback_max_age_s, opts.feedback_future_tolerance_s)
                    reader.on_release = relay.detach
                    relay.start(sys.stdin)
                while running:
                    packet = reader.poll()
                    if relay is not None:
                        relay.update(packet['sides'], {side: reader.devices[side] for side in reader.started})
                        packet['feedback'] = relay.status()
                    emit(packet)
                    time.sleep(1 / 60)
            finally:
                try:
                    if relay is not None:
                        relay.close()
                finally:
                    reader.close()
        return 0
    except Exception as exc:
        emit(dict(monotonic_s=time.monotonic(), sides={}, error=str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
