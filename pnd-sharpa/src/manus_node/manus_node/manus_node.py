#!/usr/bin/env python3
"""Own Manus acquisition/retargeting and publish Sharpa retargeted joints."""

from __future__ import annotations

import os
import pty
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from manus_node.common import (
    age_ms,
    as_bool,
    json_msg,
    sdk_path,
)
from manus_node.zmq_tools import CtypesZmqSubscriber, NonBlockingSubscriber


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


@dataclass
class StreamCounter:
    name: str
    address: str
    count: int = 0
    hz: float = 0.0
    window_count: int = 0
    last_message_time: float | None = None
    last_error: str = ""


@dataclass
class ManagedProcess:
    label: str
    command: list[str]
    cwd: str
    env: dict[str, str]
    use_pty: bool = False
    process: subprocess.Popen | None = None
    process_group_id: int | None = None
    pty_master_fd: int | None = None
    last_start_time: float | None = None
    restart_count: int = 0
    last_returncode: int | None = None
    last_output: str = ""
    last_error: str = ""
    next_start_time: float = 0.0
    blocked: bool = False
    recent_output: deque[str] = field(default_factory=lambda: deque(maxlen=20))

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


def _default_sdk_root() -> str:
    return sdk_path("sharpa-manus-sdk")


def _default_retarget_script() -> str:
    return sdk_path(
        "sharpa-manus-sdk",
        "retargeting_alg_release_V4.0",
        "retargeting_manus_demo_multiprocess.py",
    )


def _default_proto_path() -> str:
    return sdk_path(
        "sharpa-manus-sdk",
        "retargeting_alg_release_V4.0",
        "include",
        "proto_hand",
    )


def _default_retarget_python() -> str:
    for candidate in (
        os.environ.get("SHARPA_MANUS_RETARGET_PYTHON", ""),
        sdk_path("sharpa-manus-sdk", ".venv", "bin", "python"),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return sys.executable


def _default_operator_home() -> str:
    return (
        os.environ.get("PND_TELEOP_USER_HOME")
        or os.environ.get("SHARPA_MANUS_HOME")
        or "/home/pnd-humanoid"
    )


class ManusNode(Node):
    """Run Manus retargeting and expose its output as a ROS JointState."""

    def __init__(self) -> None:
        super().__init__("manus")

        sdk_root_default = _default_sdk_root()
        retarget_dir_default = os.path.dirname(_default_retarget_script())
        self.declare_parameter("sdk_root", sdk_root_default)
        self.declare_parameter(
            "manus_client_path",
            os.path.join(sdk_root_default, "client", "SharpaManusClient.out"),
        )
        self.declare_parameter("retarget_script", _default_retarget_script())
        self.declare_parameter("retarget_python", _default_retarget_python())
        self.declare_parameter("proto_path", _default_proto_path())
        self.declare_parameter("mocap_address", "tcp://127.0.0.1:2044")
        self.declare_parameter("hand_action_bind_address", "tcp://*:6668")
        self.declare_parameter("hand_action_monitor_address", "tcp://127.0.0.1:6668")
        self.declare_parameter("output_topic", "/sharpa_command_joint_states")
        self.declare_parameter("status_topic", "/manus/status")
        self.declare_parameter("poll_period", 0.005)
        self.declare_parameter("status_period", 0.5)
        self.declare_parameter("max_monitor_messages_per_poll", 8)
        self.declare_parameter("max_bridge_messages_per_poll", 8)
        self.declare_parameter("restart_on_exit", True)
        self.declare_parameter("restart_delay", 2.0)
        self.declare_parameter("start_manus_client", True)
        self.declare_parameter("start_retarget", True)
        self.declare_parameter("cleanup_stale_processes", True)
        self.declare_parameter("stream_watchdog_enabled", True)
        self.declare_parameter("mocap_startup_timeout", 60.0)
        self.declare_parameter("mocap_stale_timeout", 3.0)
        self.declare_parameter("stream_restart_cooldown", 5.0)
        self.declare_parameter("manus_fault_startup_grace", 8.0)
        self.declare_parameter("manus_fault_restart_cooldown", 60.0)
        self.declare_parameter("manus_usb_reset_enabled", True)
        self.declare_parameter("manus_usb_reset_cooldown", 60.0)
        self.declare_parameter("manus_usb_vendor_id", "3325")
        self.declare_parameter("manus_license_usb_vendor_id", "1915")
        self.declare_parameter("manus_license_usb_product_id", "83fd")
        self.declare_parameter("filter_alpha", 1.0)
        self.declare_parameter("joint_name_prefix", "")
        self.declare_parameter("operator_home", _default_operator_home())
        self.declare_parameter(
            "manus_config_home", os.path.join(_default_operator_home(), ".config")
        )
        self.declare_parameter("retarget_cwd", retarget_dir_default)

        self.sdk_root = str(self.get_parameter("sdk_root").value)
        self.manus_client_path = str(self.get_parameter("manus_client_path").value)
        self.retarget_script = str(self.get_parameter("retarget_script").value)
        self.retarget_python = str(self.get_parameter("retarget_python").value)
        self.proto_path = str(self.get_parameter("proto_path").value)
        self.mocap_address = str(self.get_parameter("mocap_address").value)
        self.hand_action_bind_address = str(
            self.get_parameter("hand_action_bind_address").value
        )
        self.hand_action_monitor_address = str(
            self.get_parameter("hand_action_monitor_address").value
        )
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.poll_period = float(self.get_parameter("poll_period").value)
        self.status_period = float(self.get_parameter("status_period").value)
        self.max_monitor_messages_per_poll = int(
            self.get_parameter("max_monitor_messages_per_poll").value
        )
        self.max_bridge_messages_per_poll = int(
            self.get_parameter("max_bridge_messages_per_poll").value
        )
        self.restart_on_exit = as_bool(self.get_parameter("restart_on_exit").value)
        self.restart_delay = float(self.get_parameter("restart_delay").value)
        self.start_manus_client = as_bool(
            self.get_parameter("start_manus_client").value
        )
        self.start_retarget = as_bool(self.get_parameter("start_retarget").value)
        self.cleanup_stale_processes = as_bool(
            self.get_parameter("cleanup_stale_processes").value
        )
        self.stream_watchdog_enabled = as_bool(
            self.get_parameter("stream_watchdog_enabled").value
        )
        self.mocap_startup_timeout = float(
            self.get_parameter("mocap_startup_timeout").value
        )
        self.mocap_stale_timeout = float(
            self.get_parameter("mocap_stale_timeout").value
        )
        self.stream_restart_cooldown = float(
            self.get_parameter("stream_restart_cooldown").value
        )
        self.manus_fault_startup_grace = float(
            self.get_parameter("manus_fault_startup_grace").value
        )
        self.manus_fault_restart_cooldown = float(
            self.get_parameter("manus_fault_restart_cooldown").value
        )
        self.manus_usb_reset_enabled = as_bool(
            self.get_parameter("manus_usb_reset_enabled").value
        )
        self.manus_usb_reset_cooldown = float(
            self.get_parameter("manus_usb_reset_cooldown").value
        )
        self.manus_usb_vendor_id = str(
            self.get_parameter("manus_usb_vendor_id").value
        )
        self.manus_license_usb_vendor_id = str(
            self.get_parameter("manus_license_usb_vendor_id").value
        )
        self.manus_license_usb_product_id = str(
            self.get_parameter("manus_license_usb_product_id").value
        )
        self.filter_alpha = float(self.get_parameter("filter_alpha").value)
        self.joint_name_prefix = str(self.get_parameter("joint_name_prefix").value)
        self.operator_home = str(self.get_parameter("operator_home").value)
        self.manus_config_home = str(self.get_parameter("manus_config_home").value)
        self.retarget_cwd = str(self.get_parameter("retarget_cwd").value)

        if self.poll_period <= 0.0:
            raise ValueError("poll_period must be positive")
        if self.status_period <= 0.0:
            raise ValueError("status_period must be positive")
        if self.max_monitor_messages_per_poll <= 0:
            raise ValueError("max_monitor_messages_per_poll must be positive")
        if self.max_bridge_messages_per_poll <= 0:
            raise ValueError("max_bridge_messages_per_poll must be positive")

        self.sharpa_hand_pb2 = self._load_proto_module(self.proto_path)
        self.retarget_pub = self.create_publisher(JointState, self.output_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.mocap_counter = StreamCounter("mocap_keypoints", self.mocap_address)
        self.hand_action_counter = StreamCounter(
            "hand_action", self.hand_action_monitor_address
        )
        self.hand_action_bridge_counter = StreamCounter(
            "retargeted_joints", self.hand_action_monitor_address
        )
        self.window_time = time.monotonic()
        self.last_error = ""
        self.last_bridge_error = ""
        self.bridge_backend = ""
        self.last_joint_count = 0
        self.published = 0
        self.bridge_window_published = 0
        self.publish_hz = 0.0
        self.last_publish_time: float | None = None
        self.last_mocap_progress_time: float | None = None
        self.watchdog_seen_mocap_count = 0
        self.watchdog_restart_count = 0
        self.last_watchdog_restart_time: float | None = None
        self.last_watchdog_reason = ""
        self.usb_reset_count = 0
        self.last_usb_reset_time: float | None = None
        self.last_usb_reset_status = ""
        self.pipeline_blocked = False
        self.pipeline_block_reason = ""

        self.mocap_subscriber = self._make_monitor(self.mocap_counter)
        self.hand_action_monitor = self._make_monitor(self.hand_action_counter)
        self.hand_action_bridge = self._make_bridge_subscriber()

        self.manus_client = ManagedProcess(
            label="manus_client",
            command=[self.manus_client_path],
            cwd=os.path.join(self.sdk_root, "client"),
            env=self._client_env(),
            use_pty=True,
        )
        self.retarget = ManagedProcess(
            label="retarget",
            command=[
                self.retarget_python,
                self.retarget_script,
                "-mocap_address",
                self.mocap_address,
                "-hand_action_address",
                self.hand_action_bind_address,
                "-filter_alpha",
                str(self.filter_alpha),
            ],
            cwd=self.retarget_cwd,
            env=self._retarget_env(),
        )

        self._cleanup_stale_pipeline_processes()
        manus_preflight_error = self._manus_client_preflight_error()
        if manus_preflight_error:
            self.pipeline_blocked = True
            self.pipeline_block_reason = manus_preflight_error
            self.last_error = f"manus_client: {manus_preflight_error}"
            self.manus_client.last_start_time = time.monotonic()
            self.manus_client.next_start_time = 0.0
            self.manus_client.last_error = manus_preflight_error
            self.manus_client.blocked = True
            self.retarget.last_error = "blocked until Manus SDK client is available"
            self.retarget.blocked = True
            self.get_logger().error(self.last_error)
        elif self.start_manus_client:
            self._start_process(self.manus_client)
        else:
            self.manus_client.last_error = "start_manus_client is false"

        if self.pipeline_blocked:
            pass
        elif self.start_retarget:
            self._start_process(self.retarget)
        else:
            self.retarget.last_error = "start_retarget is false"

        self.create_timer(self.poll_period, self._poll)
        self.create_timer(self.status_period, self._publish_status)
        self.get_logger().info(
            f"Manus node: {self.hand_action_monitor_address} -> {self.output_topic}"
        )

    def _load_proto_module(self, proto_path: str):
        if not os.path.isdir(proto_path):
            raise RuntimeError(f"proto_path does not exist: {proto_path}")
        if proto_path not in sys.path:
            sys.path.insert(0, proto_path)
        import sharpa_hand_pb2  # type: ignore

        return sharpa_hand_pb2

    def _make_monitor(self, counter: StreamCounter) -> CtypesZmqSubscriber | None:
        try:
            return CtypesZmqSubscriber(counter.address, receive_hwm=1)
        except Exception as exc:
            counter.last_error = str(exc)
            self.last_error = f"{counter.name} monitor failed: {exc}"
            self.get_logger().warning(self.last_error)
            return None

    def _make_bridge_subscriber(self) -> NonBlockingSubscriber | None:
        try:
            subscriber = NonBlockingSubscriber(
                self.hand_action_monitor_address, receive_hwm=1
            )
            self.bridge_backend = subscriber.backend
            return subscriber
        except Exception as exc:
            self.last_bridge_error = str(exc)
            self.last_error = f"hand action bridge failed: {exc}"
            self.get_logger().warning(self.last_error)
            return None

    def _client_env(self) -> dict[str, str]:
        env = os.environ.copy()
        client_lib = os.path.join(self.sdk_root, "client", "ManusSDK", "lib")
        self._prepend_env(env, "LD_LIBRARY_PATH", client_lib)
        env["HOME"] = self.operator_home
        env["XDG_CONFIG_HOME"] = self.manus_config_home
        env["USER"] = os.path.basename(self.operator_home.rstrip(os.sep)) or "pnd-humanoid"
        env["LOGNAME"] = env["USER"]
        env.setdefault("TERM", "xterm-256color")
        return env

    def _retarget_env(self) -> dict[str, str]:
        env = os.environ.copy()
        retarget_dir = os.path.dirname(self.retarget_script)
        include_dir = os.path.join(retarget_dir, "include")
        proto_dir = os.path.join(include_dir, "proto_hand")
        self._prepend_env(env, "PYTHONPATH", proto_dir)
        self._prepend_env(env, "PYTHONPATH", include_dir)
        self._prepend_env(env, "PYTHONPATH", retarget_dir)
        self._prepend_env(env, "LD_LIBRARY_PATH", include_dir)
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        env.setdefault("MPLBACKEND", "Agg")
        return env

    @staticmethod
    def _prepend_env(env: dict[str, str], key: str, value: str) -> None:
        current = env.get(key, "")
        env[key] = value if not current else f"{value}:{current}"

    def _start_process(self, managed: ManagedProcess) -> None:
        now = time.monotonic()
        managed.last_start_time = now
        managed.next_start_time = now + self.restart_delay
        managed.last_returncode = None
        managed.last_error = ""
        preflight_error = self._process_preflight_error(managed)
        if preflight_error:
            managed.blocked = True
            managed.last_error = preflight_error
            self.last_error = f"{managed.label}: {managed.last_error}"
            self.get_logger().error(self.last_error)
            return
        path = Path(managed.command[0])
        if not path.exists():
            managed.last_error = f"missing executable: {managed.command[0]}"
            self.last_error = f"{managed.label}: {managed.last_error}"
            self.get_logger().error(self.last_error)
            return
        if not os.path.isdir(managed.cwd):
            managed.last_error = f"missing cwd: {managed.cwd}"
            self.last_error = f"{managed.label}: {managed.last_error}"
            self.get_logger().error(self.last_error)
            return
        try:
            managed.process = (
                self._popen_with_pty(managed)
                if managed.use_pty
                else subprocess.Popen(
                    managed.command,
                    cwd=managed.cwd,
                    env=managed.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )
            )
            managed.process_group_id = os.getpgid(managed.process.pid)
            managed.last_start_time = now
            managed.restart_count += 1
            self._start_output_reader(managed)
            self.get_logger().info(
                f"{managed.label} pid={managed.process.pid} cwd={managed.cwd}"
            )
        except Exception as exc:
            managed.process = None
            managed.last_error = str(exc)
            self.last_error = f"{managed.label} start failed: {exc}"
            self.get_logger().error(self.last_error)

    def _process_preflight_error(self, managed: ManagedProcess) -> str:
        if managed.label != "manus_client":
            return ""
        return self._manus_client_preflight_error()

    def _manus_client_preflight_error(self) -> str:
        errors: list[str] = []
        executable = Path(self.manus_client_path)
        if not executable.exists():
            errors.append(f"missing executable: {self.manus_client_path}")
        elif not os.access(executable, os.X_OK):
            errors.append(f"not executable: {self.manus_client_path}")

        client_lib = Path(self.sdk_root) / "client" / "ManusSDK" / "lib"
        for lib_name in ("libManusSDK_Integrated.so", "libManusSDK.so"):
            lib_path = client_lib / lib_name
            if not lib_path.exists():
                errors.append(f"missing library: {lib_path}")
                continue
            if self._is_git_lfs_pointer(lib_path):
                errors.append(f"library is Git LFS pointer, not binary: {lib_path}")
        return "; ".join(errors)

    @staticmethod
    def _is_git_lfs_pointer(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                return handle.read(64).startswith(b"version https://git-lfs.github.com")
        except OSError:
            return False

    def _popen_with_pty(self, managed: ManagedProcess) -> subprocess.Popen:
        master_fd, slave_fd = pty.openpty()
        try:
            process = subprocess.Popen(
                managed.command,
                cwd=managed.cwd,
                env=managed.env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,
                close_fds=True,
                preexec_fn=os.setsid,
            )
        finally:
            os.close(slave_fd)
        managed.pty_master_fd = master_fd
        return process

    def _start_output_reader(self, managed: ManagedProcess) -> None:
        def read_output() -> None:
            if managed.pty_master_fd is not None:
                self._read_pty_output(managed)
                return
            process = managed.process
            if process is None or process.stdout is None:
                return
            for line in process.stdout:
                self._record_process_output(managed, line)

        threading.Thread(target=read_output, daemon=True).start()

    def _read_pty_output(self, managed: ManagedProcess) -> None:
        fd = managed.pty_master_fd
        if fd is None:
            return
        buffer = ""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                self._record_process_output(managed, line)
        if buffer:
            self._record_process_output(managed, buffer)

    @staticmethod
    def _record_process_output(managed: ManagedProcess, line: str) -> None:
        clean = line.strip()
        if not clean:
            return
        managed.last_output = clean[-400:]
        managed.recent_output.append(clean[-400:])
        lower = clean.lower()
        if "error" in lower or "failed" in lower or "traceback" in lower:
            managed.last_error = clean[-400:]

    def _poll(self) -> None:
        if self.pipeline_blocked:
            self._poll_monitor(self.mocap_subscriber, self.mocap_counter)
            self._poll_monitor(self.hand_action_monitor, self.hand_action_counter)
            self._poll_bridge()
            return
        if self.start_manus_client:
            self._poll_process(self.manus_client)
        if self.start_retarget:
            self._poll_process(self.retarget)
        self._poll_monitor(self.mocap_subscriber, self.mocap_counter)
        self._poll_monitor(self.hand_action_monitor, self.hand_action_counter)
        self._poll_bridge()
        self._check_stream_watchdog()

    def _poll_process(self, managed: ManagedProcess) -> None:
        process = managed.process
        if process is None:
            if self.restart_on_exit and not managed.blocked:
                self._maybe_restart(managed)
            return
        returncode = process.poll()
        if returncode is None:
            return
        managed.last_returncode = returncode
        self._kill_remaining_process_group(managed, signal.SIGTERM)
        self._kill_remaining_process_group(managed, signal.SIGKILL)
        managed.process = None
        self._close_pty(managed)
        message = f"{managed.label} exited returncode={returncode}"
        managed.last_error = message
        self.last_error = message
        self.get_logger().warning(message)
        if self.restart_on_exit and not managed.blocked:
            self._maybe_restart(managed)

    def _maybe_restart(self, managed: ManagedProcess) -> None:
        now = time.monotonic()
        if managed.next_start_time > now:
            return
        if managed.last_start_time is None:
            self._start_process(managed)
            return
        if now - managed.last_start_time >= self.restart_delay:
            self._start_process(managed)

    def _poll_monitor(
        self, subscriber: CtypesZmqSubscriber | None, counter: StreamCounter
    ) -> None:
        if subscriber is None:
            return
        for _ in range(self.max_monitor_messages_per_poll):
            try:
                payload = subscriber.recv_nonblocking()
            except Exception as exc:
                counter.last_error = str(exc)
                return
            if payload is None:
                return
            counter.count += 1
            counter.window_count += 1
            counter.last_message_time = time.monotonic()

    def _poll_bridge(self) -> None:
        if self.hand_action_bridge is None:
            return
        for _ in range(self.max_bridge_messages_per_poll):
            try:
                payload = self.hand_action_bridge.recv_nonblocking()
            except Exception as exc:
                self.last_bridge_error = str(exc)
                return
            if payload is None:
                return
            try:
                hand_action = self.sharpa_hand_pb2.HandAction()
                hand_action.ParseFromString(payload)
                msg = self._to_joint_state(hand_action)
            except Exception as exc:
                self.last_bridge_error = str(exc)
                return
            self.retarget_pub.publish(msg)
            self.published += 1
            self.bridge_window_published += 1
            self.hand_action_bridge_counter.count += 1
            self.hand_action_bridge_counter.window_count += 1
            self.hand_action_bridge_counter.last_message_time = time.monotonic()
            self.last_publish_time = self.hand_action_bridge_counter.last_message_time
            self.last_joint_count = len(msg.name)

    def _to_joint_state(self, hand_action) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = hand_action.header.frame_id or "sharpa_retargeted"
        self._append_joint(msg, hand_action.joint_left)
        self._append_joint(msg, hand_action.joint_right)
        return msg

    def _append_joint(self, msg: JointState, joint) -> None:
        for name, position in zip(joint.name, joint.position):
            msg.name.append(self._target_name(name))
            msg.position.append(float(position))
            msg.velocity.append(0.0)
            msg.effort.append(0.0)

    def _target_name(self, name: str) -> str:
        if self.joint_name_prefix and not name.startswith(self.joint_name_prefix):
            return f"{self.joint_name_prefix}{name}"
        return name

    def _check_stream_watchdog(self) -> None:
        if not self.stream_watchdog_enabled or not self.start_retarget:
            return

        now = time.monotonic()
        if self.mocap_counter.count != self.watchdog_seen_mocap_count:
            self.watchdog_seen_mocap_count = self.mocap_counter.count
            self.last_mocap_progress_time = now
            return

        if self.start_manus_client and not self.manus_client.running():
            return
        if not self.retarget.running():
            return

        latest_start = self._latest_pipeline_start_time()
        if latest_start is None:
            return

        reason = ""
        pipeline_age = now - latest_start
        if self.mocap_counter.count == 0:
            fault_reason = self._manus_fault_reason()
            if fault_reason and pipeline_age >= self.manus_fault_startup_grace:
                reason = fault_reason
            elif pipeline_age >= self.mocap_startup_timeout:
                reason = (
                    "mocap stream did not start within "
                    f"{self.mocap_startup_timeout:.1f}s"
                )
        elif self.last_mocap_progress_time is not None:
            stale_age = now - self.last_mocap_progress_time
            if stale_age >= self.mocap_stale_timeout:
                reason = f"mocap stream stale for {stale_age:.1f}s"

        if not reason:
            return
        cooldown = self.stream_restart_cooldown
        if self._reason_needs_usb_reset(reason):
            cooldown = max(cooldown, self.manus_fault_restart_cooldown)
        if (
            self.last_watchdog_restart_time is not None
            and now - self.last_watchdog_restart_time < cooldown
        ):
            return
        self._restart_pipeline(reason)

    def _latest_pipeline_start_time(self) -> float | None:
        start_times = [
            value
            for value in (self.manus_client.last_start_time, self.retarget.last_start_time)
            if value is not None
        ]
        return max(start_times) if start_times else None

    def _restart_pipeline(self, reason: str) -> None:
        self.last_watchdog_restart_time = time.monotonic()
        self.last_watchdog_reason = reason
        self.watchdog_restart_count += 1
        self.last_error = f"stream watchdog restart: {reason}"
        self.get_logger().warning(self.last_error)
        self._stop_process(self.retarget)
        if self.start_manus_client:
            self._stop_process(self.manus_client)
        self._cleanup_stale_pipeline_processes()
        if self._reason_needs_usb_reset(reason):
            self._reset_manus_usb_devices()
        self.watchdog_seen_mocap_count = self.mocap_counter.count
        self.last_mocap_progress_time = None
        if self.start_manus_client:
            self._start_process(self.manus_client)
        self._start_process(self.retarget)

    def _manus_fault_reason(self) -> str:
        recent = "\n".join(self.manus_client.recent_output).lower()
        if "unknown device type found" in recent:
            return "manus device type detection failed"
        if "no compatible license found" in recent:
            return "manus SDK reported no compatible license"
        if "second stage initialisation failed" in recent:
            return "manus device second stage initialisation failed"
        if "second stage initialization failed" in recent:
            return "manus device second stage initialization failed"
        return ""

    @staticmethod
    def _reason_needs_usb_reset(reason: str) -> bool:
        lower = reason.lower()
        return (
            "manus license" in lower
            or "compatible license" in lower
            or "unknown device" in lower
            or "device type detection" in lower
            or "second stage" in lower
        )

    def _manus_usb_status(self) -> dict:
        devices = self._usb_devices()
        sensor = [
            device
            for device in devices
            if device.get("vendor_id") == self.manus_usb_vendor_id.lower()
        ]
        license_key = [
            device
            for device in devices
            if device.get("vendor_id") == self.manus_license_usb_vendor_id.lower()
            and device.get("product_id") == self.manus_license_usb_product_id.lower()
        ]
        return {
            "sensor_dongles": sensor,
            "license_keys": license_key,
        }

    def _reset_manus_usb_devices(self) -> None:
        if not self.manus_usb_reset_enabled:
            self.last_usb_reset_status = "disabled"
            return
        now = time.monotonic()
        if (
            self.last_usb_reset_time is not None
            and now - self.last_usb_reset_time < self.manus_usb_reset_cooldown
        ):
            remaining = self.manus_usb_reset_cooldown - (now - self.last_usb_reset_time)
            self.last_usb_reset_status = f"cooldown {remaining:.1f}s remaining"
            return

        devices = [
            device
            for device in self._usb_devices()
            if device.get("vendor_id") == self.manus_usb_vendor_id.lower()
            or (
                device.get("vendor_id") == self.manus_license_usb_vendor_id.lower()
                and device.get("product_id")
                == self.manus_license_usb_product_id.lower()
            )
        ]
        if not devices:
            self.last_usb_reset_status = "no Manus USB devices found"
            return

        reset_paths: list[str] = []
        errors: list[str] = []
        for device in devices:
            usb_path = device.get("sysfs_path", "")
            authorized_path = os.path.join(usb_path, "authorized")
            if not os.path.exists(authorized_path):
                continue
            try:
                with open(authorized_path, "w", encoding="utf-8") as file:
                    file.write("0")
                time.sleep(0.5)
                with open(authorized_path, "w", encoding="utf-8") as file:
                    file.write("1")
                reset_paths.append(device.get("name", usb_path))
            except OSError as exc:
                errors.append(f"{device.get('name', usb_path)}: {exc}")

        self.usb_reset_count += 1
        self.last_usb_reset_time = now
        if errors:
            self.last_usb_reset_status = "errors: " + "; ".join(errors)
        elif reset_paths:
            self.last_usb_reset_status = "reset " + ",".join(reset_paths)
        else:
            self.last_usb_reset_status = "no resettable Manus USB devices"
        self.get_logger().warning("Manus USB reset: " + self.last_usb_reset_status)
        time.sleep(2.0)

    def _usb_devices(self) -> list[dict]:
        devices: list[dict] = []
        root = "/sys/bus/usb/devices"
        try:
            entries = os.listdir(root)
        except OSError:
            return devices

        for entry in entries:
            sysfs_path = os.path.join(root, entry)
            vendor_id = self._read_sysfs_text(os.path.join(sysfs_path, "idVendor"))
            product_id = self._read_sysfs_text(os.path.join(sysfs_path, "idProduct"))
            if not vendor_id or not product_id:
                continue
            manufacturer = self._read_sysfs_text(
                os.path.join(sysfs_path, "manufacturer")
            )
            product = self._read_sysfs_text(os.path.join(sysfs_path, "product"))
            serial = self._read_sysfs_text(os.path.join(sysfs_path, "serial"))
            devices.append(
                {
                    "name": entry,
                    "vendor_id": vendor_id.lower(),
                    "product_id": product_id.lower(),
                    "manufacturer": manufacturer,
                    "product": product,
                    "serial": serial,
                    "sysfs_path": sysfs_path,
                }
            )
        return devices

    @staticmethod
    def _read_sysfs_text(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except OSError:
            return ""

    def _publish_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self.window_time
        if elapsed > 0.0:
            self.mocap_counter.hz = self.mocap_counter.window_count / elapsed
            self.hand_action_counter.hz = (
                self.hand_action_counter.window_count / elapsed
            )
            self.hand_action_bridge_counter.hz = (
                self.hand_action_bridge_counter.window_count / elapsed
            )
            self.publish_hz = self.bridge_window_published / elapsed
        self.mocap_counter.window_count = 0
        self.hand_action_counter.window_count = 0
        self.hand_action_bridge_counter.window_count = 0
        self.bridge_window_published = 0
        self.window_time = now

        payload = {
            "node": "manus",
            "output_topic": self.output_topic,
            "sdk_root": self.sdk_root,
            "retarget_python": self.retarget_python,
            "proto_path": self.proto_path,
            "pipeline": {
                "blocked": self.pipeline_blocked,
                "reason": self.pipeline_block_reason,
            },
            "addresses": {
                "mocap": self.mocap_address,
                "hand_action_bind": self.hand_action_bind_address,
                "hand_action_monitor": self.hand_action_monitor_address,
            },
            "manus_environment": {
                "home": self.operator_home,
                "xdg_config_home": self.manus_config_home,
            },
            "polling": {
                "poll_period_s": self.poll_period,
                "status_period_s": self.status_period,
                "max_monitor_messages_per_poll": self.max_monitor_messages_per_poll,
                "max_bridge_messages_per_poll": self.max_bridge_messages_per_poll,
            },
            "manus_usb": self._manus_usb_status(),
            "processes": {
                "manus_client": self._process_status(self.manus_client, now),
                "retarget": self._process_status(self.retarget, now),
            },
            "streams": {
                "mocap_keypoints": self._counter_status(self.mocap_counter, now),
                "hand_action": self._counter_status(self.hand_action_counter, now),
                "retargeted_joints": self._counter_status(
                    self.hand_action_bridge_counter, now
                ),
            },
            "bridge": {
                "backend": self.bridge_backend,
                "published": self.published,
                "publish_hz": round(self.publish_hz, 2),
                "last_publish_age_ms": age_ms(self.last_publish_time, now),
                "last_joint_count": self.last_joint_count,
                "last_error": self.last_bridge_error,
            },
            "watchdog": {
                "enabled": self.stream_watchdog_enabled,
                "mocap_startup_timeout_s": self.mocap_startup_timeout,
                "mocap_stale_timeout_s": self.mocap_stale_timeout,
                "stream_restart_cooldown_s": self.stream_restart_cooldown,
                "manus_fault_startup_grace_s": self.manus_fault_startup_grace,
                "manus_fault_restart_cooldown_s": self.manus_fault_restart_cooldown,
                "manus_usb_reset_enabled": self.manus_usb_reset_enabled,
                "manus_usb_reset_cooldown_s": self.manus_usb_reset_cooldown,
                "usb_reset_count": self.usb_reset_count,
                "last_usb_reset_age_ms": age_ms(self.last_usb_reset_time, now),
                "last_usb_reset_status": self.last_usb_reset_status,
                "restart_count": self.watchdog_restart_count,
                "last_restart_age_ms": age_ms(self.last_watchdog_restart_time, now),
                "last_reason": self.last_watchdog_reason,
                "last_mocap_progress_age_ms": age_ms(
                    self.last_mocap_progress_time, now
                ),
            },
            "last_error": self.last_error,
        }
        self.status_pub.publish(json_msg(payload))

    @staticmethod
    def _process_status(managed: ManagedProcess, now: float) -> dict:
        return {
            "pid": None if managed.process is None else managed.process.pid,
            "pgid": managed.process_group_id,
            "running": managed.running(),
            "last_start_age_ms": age_ms(managed.last_start_time, now),
            "restart_count": managed.restart_count,
            "last_returncode": managed.last_returncode,
            "last_output": managed.last_output,
            "last_error": managed.last_error,
            "next_start_age_ms": age_ms(managed.next_start_time, now),
            "blocked": managed.blocked,
            "recent_output": list(managed.recent_output),
        }

    @staticmethod
    def _counter_status(counter: StreamCounter, now: float) -> dict:
        return {
            "address": counter.address,
            "count": counter.count,
            "hz": round(counter.hz, 2),
            "last_message_age_ms": age_ms(counter.last_message_time, now),
            "last_error": counter.last_error,
        }

    def _stop_process(self, managed: ManagedProcess) -> None:
        process = managed.process
        if process is None:
            self._kill_remaining_process_group(managed, signal.SIGTERM)
            self._kill_remaining_process_group(managed, signal.SIGKILL)
            managed.process_group_id = None
            self._close_pty(managed)
            return
        try:
            self._kill_remaining_process_group(managed, signal.SIGTERM)
            if process.poll() is None:
                process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._kill_remaining_process_group(managed, signal.SIGKILL)
            try:
                process.wait(timeout=1.0)
            except Exception as exc:
                managed.last_error = str(exc)
        except ProcessLookupError:
            pass
        finally:
            self._kill_remaining_process_group(managed, signal.SIGTERM)
            self._kill_remaining_process_group(managed, signal.SIGKILL)
            managed.last_returncode = process.poll()
            managed.process = None
            managed.process_group_id = None
            self._close_pty(managed)

    @staticmethod
    def _kill_remaining_process_group(
        managed: ManagedProcess, sig: signal.Signals
    ) -> None:
        pgid = managed.process_group_id
        if pgid is None:
            process = managed.process
            if process is None:
                return
            try:
                pgid = os.getpgid(process.pid)
                managed.process_group_id = pgid
            except ProcessLookupError:
                return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            managed.last_error = str(exc)

    def _cleanup_stale_pipeline_processes(self) -> None:
        if not self.cleanup_stale_processes:
            return
        pipeline_root = os.path.abspath(self.sdk_root)
        markers = (
            os.path.basename(self.manus_client_path),
            os.path.basename(self.retarget_script),
            "multiprocessing.spawn",
            "multiprocessing.resource_tracker",
        )
        killed_pgids: set[int] = set()
        try:
            proc_entries = os.listdir("/proc")
        except OSError:
            return
        for entry in proc_entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == os.getpid():
                continue
            proc = os.path.join("/proc", entry)
            try:
                with open(os.path.join(proc, "cmdline"), "rb") as file:
                    cmdline = (
                        file.read()
                        .replace(b"\x00", b" ")
                        .decode("utf-8", errors="replace")
                    )
                cwd = os.readlink(os.path.join(proc, "cwd"))
                pgid = os.getpgid(pid)
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                continue
            if pgid in killed_pgids:
                continue
            if not os.path.abspath(cwd).startswith(pipeline_root):
                continue
            if not any(marker and marker in cmdline for marker in markers):
                continue
            try:
                os.killpg(pgid, signal.SIGTERM)
                killed_pgids.add(pgid)
            except (ProcessLookupError, PermissionError):
                continue
        if not killed_pgids:
            return
        time.sleep(0.2)
        for pgid in killed_pgids:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.get_logger().warning(
            "cleaned stale Manus process groups: "
            + ",".join(str(pgid) for pgid in sorted(killed_pgids))
        )

    @staticmethod
    def _close_pty(managed: ManagedProcess) -> None:
        if managed.pty_master_fd is None:
            return
        try:
            os.close(managed.pty_master_fd)
        except OSError:
            pass
        managed.pty_master_fd = None

    def destroy_node(self) -> bool:
        self._stop_process(self.retarget)
        self._stop_process(self.manus_client)
        if self.mocap_subscriber is not None:
            self.mocap_subscriber.close()
        if self.hand_action_monitor is not None:
            self.hand_action_monitor.close()
        if self.hand_action_bridge is not None:
            self.hand_action_bridge.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManusNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
