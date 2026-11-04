"""Manus hand input boundary; retargeting stays an explicit producer."""

from __future__ import annotations

from ur_common.protection_config import setting
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import sys
from pathlib import Path
from typing import Any

import numpy as np


class ManusCore:
    JOINT_NAMES = (
        "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_mcp_aa", "thumb_ip",
        "index_mcp_fe", "index_mcp_aa", "index_pip", "index_dip",
        "middle_mcp_fe", "middle_mcp_aa", "middle_pip", "middle_dip",
        "ring_mcp_fe", "ring_mcp_aa", "ring_pip", "ring_dip",
        "pinky_cmc", "pinky_mcp_fe", "pinky_mcp_aa", "pinky_pip", "pinky_dip",
    )

    @staticmethod
    def parse(payload: Any) -> np.ndarray:
        values = payload.get("q") if isinstance(payload, dict) else payload
        joints = np.asarray(values, dtype=np.float32).reshape(-1)
        if joints.shape != (44,) or not np.isfinite(joints).all():
            raise ValueError("Manus retarget output must be 44 finite joint values")
        return joints

    @classmethod
    def joint_action_side_to_q(cls, action: Any, side: str) -> np.ndarray:
        if side not in ("left", "right"):
            raise ValueError(f"unknown Manus side {side!r}")
        joints = getattr(action, f"joint_{side}")
        names = list(joints.name)
        positions = list(joints.position)
        if len(names) != len(positions):
            raise ValueError(f"Manus {side} joint names and positions have different lengths")

        values = {}
        prefix = f"{side}_"
        for raw_name, position in zip(names, positions):
            name = str(raw_name).strip().lower().replace("-", "_")
            if name.startswith(prefix):
                name = name[len(prefix):]
            if name in values:
                raise ValueError(f"Manus {side} joint is duplicated: {name}")
            values[name] = float(position)

        missing = [name for name in cls.JOINT_NAMES if name not in values]
        if missing:
            raise ValueError(f"Manus {side} joints missing: {', '.join(missing)}")
        result = np.asarray(
            [values[name] for name in cls.JOINT_NAMES], dtype=np.float32
        )
        if result.shape != (22,) or not np.isfinite(result).all():
            raise ValueError(f"Manus {side} retarget output must be 22 finite values")
        return result

    @classmethod
    def joint_action_to_q(cls, action: Any) -> np.ndarray:
        return cls.parse(np.concatenate((
            cls.joint_action_side_to_q(action, "left"),
            cls.joint_action_side_to_q(action, "right"),
        )))


class ManusHealthTracker:
    """Track both raw gloves and retarget output without depending on ROS."""

    EXPECTED_KEYPOINTS = 25

    def __init__(self, startup_grace_s=8.0, raw_timeout_s=2.0,
                 incomplete_timeout_s=1.0, action_timeout_s=2.0):
        self.startup_grace_s = max(0.0, float(startup_grace_s))
        self.raw_timeout_s = max(0.1, float(raw_timeout_s))
        self.incomplete_timeout_s = max(0.1, float(incomplete_timeout_s))
        self.action_timeout_s = max(0.1, float(action_timeout_s))
        self.reset()

    def reset(self, now=None):
        self.started = time.monotonic() if now is None else float(now)
        self.first_raw = 0.0
        self.last_raw = 0.0
        self.last_side_raw = {"left": 0.0, "right": 0.0}
        self.last_action = 0.0
        self.left_keypoints = 0
        self.right_keypoints = 0
        self.incomplete_since = 0.0

    def observe_raw(self, left_count, right_count, now=None):
        now = time.monotonic() if now is None else float(now)
        if self.first_raw <= 0.0:
            self.first_raw = now
        self.last_raw = now
        for side, count in (("left", left_count), ("right", right_count)):
            if int(count) == self.EXPECTED_KEYPOINTS:
                self.last_side_raw[side] = now
        self.left_keypoints = self.side_keypoint_count("left", now)
        self.right_keypoints = self.side_keypoint_count("right", now)
        complete = self.raw_complete(now, self.incomplete_timeout_s)
        if complete:
            self.incomplete_since = 0.0
        elif self.incomplete_since <= 0.0:
            self.incomplete_since = now
        return complete

    def observe_action(self, now=None):
        self.last_action = time.monotonic() if now is None else float(now)

    def raw_complete(self, now=None, max_age_s=0.5):
        now = time.monotonic() if now is None else float(now)
        return bool(
            self.last_raw > 0.0
            and now - self.last_raw < float(max_age_s)
            and self.side_complete("left", now, max_age_s)
            and self.side_complete("right", now, max_age_s)
        )

    def side_complete(self, side, now=None, max_age_s=0.5):
        if side not in ("left", "right"):
            raise ValueError(f"unknown Manus side {side!r}")
        now = time.monotonic() if now is None else float(now)
        last_side_raw = self.last_side_raw[side]
        return bool(
            last_side_raw > 0.0
            and now - last_side_raw < float(max_age_s)
        )

    def side_keypoint_count(self, side, now=None, max_age_s=None):
        max_age_s = (
            self.incomplete_timeout_s if max_age_s is None else float(max_age_s)
        )
        return (
            self.EXPECTED_KEYPOINTS
            if self.side_complete(side, now, max_age_s)
            else 0
        )

    def incomplete_sides(self, now=None, max_age_s=None):
        now = time.monotonic() if now is None else float(now)
        max_age_s = (
            self.incomplete_timeout_s if max_age_s is None else float(max_age_s)
        )
        return [
            side
            for side in ("left", "right")
            if not self.side_complete(side, now, max_age_s)
        ]

    def fault(self, client_alive, retarget_alive, now=None):
        now = time.monotonic() if now is None else float(now)
        if not client_alive:
            return "Manus client exited"
        if not retarget_alive:
            return "Manus retarget exited"
        raw_is_fresh = bool(
            self.last_raw > 0.0 and now - self.last_raw < self.raw_timeout_s
        )
        # The vendor stream can update the two gloves in separate messages.
        # Treat a side as missing only if that side itself has not produced a
        # complete 25-keypoint sample for the full timeout.
        missing_sides = []
        if raw_is_fresh:
            for side in ("left", "right"):
                last_side_raw = self.last_side_raw[side]
                missing_since = last_side_raw or self.first_raw
                if (
                    missing_since > 0.0
                    and now - missing_since > self.incomplete_timeout_s
                ):
                    missing_sides.append(side)
        if missing_sides:
            sides = ", ".join(missing_sides)
            return (
                f"Manus {sides} glove mocap remained incomplete for "
                f"{self.incomplete_timeout_s:.1f}s"
            )
        if now - self.started < self.startup_grace_s:
            return ""
        if not raw_is_fresh:
            return "Manus mocap stream on 2044 is stale"
        if self.last_action <= 0.0 or now - self.last_action >= self.action_timeout_s:
            return "Manus retarget output on 6668 is stale"
        return ""


def _process_tree_groups(root_process_id: int) -> set[int]:
    children = {}
    groups = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            process_id = int(entry.name)
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parent_process_id = int(fields[1])
            process_group_id = int(fields[2])
            children.setdefault(parent_process_id, []).append(process_id)
            groups[process_id] = process_group_id
        except (OSError, ValueError, IndexError):
            continue
    process_ids = [root_process_id]
    descendants = set()
    while process_ids:
        process_id = process_ids.pop()
        if process_id in descendants:
            continue
        descendants.add(process_id)
        process_ids.extend(children.get(process_id, ()))
    return {groups[process_id] for process_id in descendants if process_id in groups}


def terminate_process_group(process, timeout_s: float = 1.0) -> None:
    if process is None:
        return
    process_ended = process.poll() is not None
    # Every managed subprocess is launched with start_new_session=True, so its
    # pid is also its primary process-group id. Keep it even if the wrapper has
    # already exited: descendants may still be alive in that group.
    process_groups = _process_tree_groups(process.pid)
    process_groups.add(process.pid)
    for process_group in process_groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if not process_ended:
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass
    for process_group in process_groups:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            continue
        os.killpg(process_group, signal.SIGKILL)
    if not process_ended:
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import String
    from ur_msgs.msg import HandCommand, ControlStatus

    class ManusNode(Node):
        def __init__(self):
            super().__init__("manus_node")
            self.declare_parameter("sdk_root", os.environ.get("MANUS_SDK_ROOT", ""))
            self.declare_parameter("retarget_root", os.environ.get("MANUS_RETARGET_ROOT", ""))
            self.declare_parameter("python_path", os.environ.get("MANUS_PYTHON_PATH", ""))
            self.declare_parameter("retarget_python", "/usr/bin/python3")
            # SharpaManusClient publishes protobuf MocapKeypoints on 2044;
            # 6667 is reserved for the independent Vive tracker stream.
            self.declare_parameter("mocap_address", "tcp://127.0.0.1:2044")
            self.declare_parameter("hand_action_address", "tcp://127.0.0.1:6668")
            self.declare_parameter("hand_action_bind_address", "tcp://*:6668")
            self.declare_parameter("start_hardware", True)
            self.declare_parameter("client_use_sudo", False)
            self.declare_parameter("retarget_command", "")
            self.declare_parameter("startup_grace_s", setting("manus", "startup_grace_s"))
            self.declare_parameter("raw_timeout_s", setting("manus", "raw_timeout_s"))
            self.declare_parameter("incomplete_hand_timeout_s", setting("manus", "incomplete_hand_timeout_s"))
            self.declare_parameter("action_timeout_s", setting("manus", "action_timeout_s"))
            self.declare_parameter("hil_mode", False)
            self.declare_parameter("workbench_mode", False)
            self._process = None; self._reader = None; self._retarget = None
            self._zmq = None; self._mocap_zmq = None; self._proto = None; self._hardware_error = ""
            self._hardware_started = False; self._last_action_monotonic = 0.0
            self._last_side_action_monotonic = {"left": 0.0, "right": 0.0}
            self._destroying = False
            command = str(self.get_parameter("retarget_command").value).strip()
            self._hardware_requested = bool(
                self.get_parameter("start_hardware").value
            ) and not bool(command)
            self._health = ManusHealthTracker(
                self.get_parameter("startup_grace_s").value,
                self.get_parameter("raw_timeout_s").value,
                self.get_parameter("incomplete_hand_timeout_s").value,
                self.get_parameter("action_timeout_s").value,
            )
            topic = '/hil/hand_candidate' if self.get_parameter('hil_mode').value else '/sharpa/command'
            self.pub = self.create_publisher(HandCommand, topic, 10)
            if self.get_parameter("workbench_mode").value:
                self._direct_pub = self.create_publisher(HandCommand, '/sharpa/command', 10)
                self._candidate_pub = self.create_publisher(HandCommand, '/hil/hand_candidate', 10)
                self.pub = self._candidate_pub
                self.create_subscription(ControlStatus, '/control/status', self._control, 10)
            self.status = self.create_publisher(String, "/teleop/manus_status", 10)
            self.create_subscription(String, "/manus/retargeted_json", self._input, 10)
            if command:
                self._start_process(command)
            elif self._hardware_requested:
                self._start_hardware()
            self._poll_timer = self.create_timer(0.01, self._poll_hardware)
            self._status_timer = self.create_timer(0.5, self._publish_status)

        def _control(self, message):
            self.pub = self._direct_pub if int(message.active_source) == 2 else self._candidate_pub

        def _start_process(self, command):
            self._process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            self._reader = threading.Thread(target=self._read_process, daemon=True); self._reader.start()

        def _start_hardware(self):
            sdk_root = Path(str(self.get_parameter("sdk_root").value)).expanduser()
            retarget_root = Path(str(self.get_parameter("retarget_root").value)).expanduser()
            configured_python_path = str(self.get_parameter("python_path").value).strip()
            python_path = (
                Path(configured_python_path).expanduser()
                if configured_python_path
                else sdk_root.parent / "manus-python"
            )
            client = sdk_root / "SharpaManusClient.out"
            script = retarget_root / "retargeting_manus_demo_multiprocess.py"
            proto = retarget_root / "include" / "proto_hand"
            py = Path(str(self.get_parameter("retarget_python").value)).expanduser()
            missing = [str(p) for p in (client, script, proto, python_path) if not p.exists()]
            if missing:
                self._hardware_error = "missing Manus SDK files: " + ", ".join(missing)
                self.get_logger().error(self._hardware_error); return
            env = os.environ.copy(); env["LD_LIBRARY_PATH"] = str(sdk_root / "ManusSDK" / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
            env["PYTHONPATH"] = str(python_path) + ":" + str(proto) + ":" + str(retarget_root / "include") + ":" + str(retarget_root) + ":" + env.get("PYTHONPATH", "")
            env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
            try:
                if str(python_path) not in sys.path:
                    sys.path.insert(0, str(python_path))
                # SharpaManusClient uses ncurses/termios during startup.  A
                # plain Popen has no controlling TTY and exits immediately.
                # script allocates a pseudo-terminal. The client normally runs
                # as the launch user; sudo remains an explicit opt-in.
                client_command = shlex.join([f"./{client.name}"])
                client_args = ["script", "-qec", client_command, "/dev/null"]
                if bool(self.get_parameter("client_use_sudo").value):
                    client_args = ["sudo", "-n", *client_args]
                self._process = subprocess.Popen(
                    client_args,
                    cwd=str(sdk_root), env=env, stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self._retarget = subprocess.Popen([str(py), str(script), "-mocap_address", str(self.get_parameter("mocap_address").value), "-hand_action_address", str(self.get_parameter("hand_action_bind_address").value), "-filter_alpha", "1.0"], cwd=str(retarget_root), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
                import zmq
                self._zmq = zmq.Context.instance().socket(zmq.SUB); self._zmq.setsockopt(zmq.RCVHWM, 1); self._zmq.connect(str(self.get_parameter("hand_action_address").value)); self._zmq.setsockopt_string(zmq.SUBSCRIBE, "")
                self._mocap_zmq = zmq.Context.instance().socket(zmq.SUB); self._mocap_zmq.setsockopt(zmq.RCVHWM, 1); self._mocap_zmq.connect(str(self.get_parameter("mocap_address").value)); self._mocap_zmq.setsockopt_string(zmq.SUBSCRIBE, "")
                sys.path.insert(0, str(proto)); import sharpa_hand_pb2
                self._proto = sharpa_hand_pb2
                self._hardware_started = True; self._hardware_error = ""
                self._last_action_monotonic = 0.0; self._health.reset()
                self.get_logger().info("Manus client and retarget pipeline started")
            except Exception as exc:
                self._stop_hardware()
                self._hardware_error = str(exc); self.get_logger().error(f"Manus pipeline unavailable: {exc}")

        def _poll_hardware(self):
            if self._proto is None: return
            try:
                import zmq
                if self._mocap_zmq is not None:
                    while True:
                        try: payload = self._mocap_zmq.recv(flags=zmq.NOBLOCK)
                        except zmq.Again: break
                        mocap = self._proto.MocapKeypoints(); mocap.ParseFromString(payload)
                        now = time.monotonic()
                        complete = self._health.observe_raw(
                            len(mocap.left_mocap_pose), len(mocap.right_mocap_pose), now
                        )
                        if not complete:
                            self._hardware_error = (
                                "Manus incomplete mocap frame: "
                                f"left={self._health.side_keypoint_count('left', now)}, "
                                f"right={self._health.side_keypoint_count('right', now)}, "
                                f"expected={self._health.EXPECTED_KEYPOINTS} each"
                            )
                if self._zmq is None:
                    return
                while True:
                    try: payload = self._zmq.recv(flags=zmq.NOBLOCK)
                    except zmq.Again: break
                    action = self._proto.HandAction(); action.ParseFromString(payload)
                    self._health.observe_action()
                    commands = {}
                    side_errors = []
                    for side in ("left", "right"):
                        if not self._health.side_complete(side):
                            continue
                        try:
                            q_side = ManusCore.joint_action_side_to_q(action, side)
                        except Exception as exc:
                            side_errors.append(str(exc))
                            continue
                        commands[side] = q_side
                    if set(commands) == {"left", "right"}:
                        self._publish_q(
                            np.concatenate((commands["left"], commands["right"]))
                        )
                    else:
                        # Preserve one-hand teleoperation when a glove is
                        # unavailable. SharpA deliberately marks these partial
                        # commands invalid for the paired q_state/q_cmd dataset.
                        for side, q_side in commands.items():
                            self._publish_side(side, q_side)
                    incomplete = [
                        side for side in ("left", "right")
                        if not self._health.side_complete(side)
                    ]
                    details = side_errors + [
                        f"{side} Manus mocap incomplete"
                        for side in incomplete
                    ]
                    self._hardware_error = "; ".join(details)
            except Exception as exc:
                self._hardware_error = str(exc)

        def _stop_hardware(self):
            for socket_name in ("_mocap_zmq", "_zmq"):
                socket = getattr(self, socket_name, None)
                if socket is not None:
                    socket.close(linger=0)
                setattr(self, socket_name, None)
            for process_name in ("_retarget", "_process"):
                terminate_process_group(getattr(self, process_name, None))
                setattr(self, process_name, None)
            self._reader = None; self._proto = None
            self._hardware_started = False; self._last_action_monotonic = 0.0
            self._last_side_action_monotonic = {"left": 0.0, "right": 0.0}

        def _publish_status(self):
            now = time.monotonic()
            data_age = None if self._last_action_monotonic <= 0.0 else max(0.0, now - self._last_action_monotonic)
            client_alive = bool(self._process is not None and self._process.poll() is None)
            retarget_alive = bool(self._retarget is not None and self._retarget.poll() is None)
            raw_age = None if self._health.last_raw <= 0.0 else max(0.0, now - self._health.last_raw)
            side_valid = {
                side: bool(
                    now - self._last_side_action_monotonic[side] < 0.5
                    and self._last_side_action_monotonic[side] > 0.0
                    and self._health.side_complete(side, now)
                )
                for side in ("left", "right")
            }
            valid = any(side_valid.values()) if self._hardware_requested else bool(data_age is not None and data_age < 0.5)
            error = self._hardware_error
            if self._hardware_started and not client_alive:
                code = None if self._process is None else self._process.poll()
                error = error or f"Manus client exited with code {code}"
            elif self._hardware_started and not retarget_alive:
                code = None if self._retarget is None else self._retarget.poll()
                error = error or f"Manus retarget exited with code {code}"
            elif self._hardware_started:
                # A live vendor process can stop producing frames without
                # exiting. Surface the stream fault without stopping the
                # pipeline or disabling a healthy single glove.
                error = error or self._health.fault(client_alive, retarget_alive, now)
            out = String()
            out.data = json.dumps({
                "connected": bool(valid),
                "valid": bool(valid),
                "managed_by_ros": True,
                "hardware_started": bool(self._hardware_started),
                "client_use_sudo": bool(self.get_parameter("client_use_sudo").value),
                "client_alive": client_alive,
                "retarget_alive": retarget_alive,
                "data_age_s": data_age,
                "mocap_age_s": raw_age,
                "left_keypoints": self._health.side_keypoint_count("left", now),
                "right_keypoints": self._health.side_keypoint_count("right", now),
                "left_valid": side_valid["left"],
                "right_valid": side_valid["right"],
                "error": error,
            }, separators=(",", ":"))
            self.status.publish(out)

        def _publish_side(self, side, q_side):
            now = time.monotonic()
            self._last_action_monotonic = now
            self._last_side_action_monotonic[side] = now
            command = HandCommand(); command.source = HandCommand.SOURCE_VIVE; command.action_id = ""; command.revision = 0; command.step_index = 0
            q = np.zeros(44, dtype=np.float32)
            q[:22] = q_side if side == "left" else 0.0
            q[22:] = q_side if side == "right" else 0.0
            command.q_cmd = q.tolist()
            command.left_enable = side == "left"
            command.right_enable = side == "right"
            self.pub.publish(command)

        def _publish_q(self, q):
            values = ManusCore.parse(q)
            now = time.monotonic()
            self._last_action_monotonic = now
            self._last_side_action_monotonic["left"] = now
            self._last_side_action_monotonic["right"] = now
            command = HandCommand()
            command.source = HandCommand.SOURCE_VIVE
            command.action_id = ""
            command.revision = 0
            command.step_index = 0
            command.q_cmd = values.tolist()
            command.left_enable = True
            command.right_enable = True
            self.pub.publish(command)

        def _read_process(self):
            if self._process is None or self._process.stdout is None: return
            for line in self._process.stdout:
                line = line.strip()
                if line:
                    self._input(String(data=line))

        def _input(self, message):
            out = String()
            try:
                q = ManusCore.parse(json.loads(message.data)); self._publish_q(q); return
            except Exception as exc:
                out.data = json.dumps({"connected": False, "valid": False, "error": str(exc)})
            self.status.publish(out)

        def destroy_node(self):
            self._destroying = True
            self._stop_hardware()
            return super().destroy_node()

    rclpy.init(args=args); node = ManusNode()

    def handle_sigterm(_signum, _frame):
        # ros2 launch normally performs an orderly shutdown, but a direct
        # SIGTERM to this Python process must not orphan either vendor process.
        node._destroying = True
        node._stop_hardware()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, handle_sigterm)
    try: rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
