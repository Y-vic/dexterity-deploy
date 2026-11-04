"""ROS outlet for AE01 retargeted input, with an isolated native SDK process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .core import AE01Core
from .sdk_worker import PACKET_PREFIX
from .feedback import FEEDBACK_HZ, FeedbackInput, FeedbackWriter
from ur_common.protection_config import setting


def output_topic(hil_mode, workbench_mode, control, now):
    """Only the authoritative, fresh ControlStatus can select the direct outlet."""
    if control is None or now-control[0] > .5:
        return None
    state = control[1]
    if state.active_source == 2:
        if (not hil_mode or workbench_mode) and state.sharpa_state == 1:
            return "/sharpa/command"
        return None
    if state.active_source == 3 and hil_mode:
        return "/hil/hand_candidate"
    return None


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy._rclpy_pybind11 import RCLError
    from std_msgs.msg import Bool, String
    from ur_msgs.msg import HandCommand, ControlStatus, WrenchFrame

    class AE01Node(Node):
        def __init__(self):
            super().__init__("ae01_node")
            root = os.environ.get("AE01_SDK_ROOT", "/opt/avatar-sdk")
            for name, default in (("sdk_root", root), ("sdk_config", str(Path(root)/"share/sdk_config.json")),
                                  ("transport", "wireless"), ("start_hardware", True),
                                  ("data_timeout_s", setting("ae01", "data_timeout_s")),
                                  ("feedback_enabled", True),
                                  ("hil_mode", False), ("workbench_mode", False)):
                self.declare_parameter(name, default)
            self.core = AE01Core(float(self.get_parameter("data_timeout_s").value))
            self._lock = threading.Lock()
            self._control = None
            self._packet = None
            self._worker = None
            self._reader = None
            self._error = ""
            self._started = False
            self._last_published = None
            self._feedback_enabled = bool(self.get_parameter("feedback_enabled").value)
            self._feedback_input = FeedbackInput(setting('tactile', 'max_age_s'),
                setting('observation', 'future_tolerance_s'))
            self._feedback_writer = None
            self._feedback_error = ""
            self._feedback_report = {}
            self._feedback_seen_s = 0.
            self._hardware_stopped = False
            self._pubs = {topic: self.create_publisher(HandCommand, topic, 10)
                          for topic in ("/sharpa/command", "/hil/hand_candidate")}
            self._status = self.create_publisher(String, "/teleop/glove_status", 10)
            self.create_subscription(ControlStatus, "/control/status", self._on_control, 10)
            self.create_subscription(Bool, '/hardware/stop', self._on_stop, 10)
            self.create_subscription(WrenchFrame, '/sharpa/wrench', self._on_wrench, 10)
            if self.get_parameter("start_hardware").value:
                self._start_worker()
            self.create_timer(.01, self._tick)
            self.create_timer(.1, self._publish_status)
            self.create_timer(1/FEEDBACK_HZ, self._publish_feedback)

        def _start_worker(self):
            try:
                command = [sys.executable, "-u", "-m", "ae01_node.sdk_worker", "--parent-pid", str(os.getpid())]
                for flag, parameter in (("--sdk-root", "sdk_root"), ("--config", "sdk_config"), ("--transport", "transport")):
                    command.extend((flag, str(self.get_parameter(parameter).value)))
                if self._feedback_enabled:
                    command.extend(('--feedback', '--feedback-max-age-s', str(self._feedback_input.max_age_s),
                                    '--feedback-future-tolerance-s', str(self._feedback_input.future_tolerance_s)))
                env = os.environ.copy()
                # SDK ELF files carry private RUNPATHs; do not inherit ROS/Pinocchio DSOs.
                env.pop("LD_LIBRARY_PATH", None)
                self._worker = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=None,
                    stdin=subprocess.PIPE if self._feedback_enabled else subprocess.DEVNULL,
                    text=True, bufsize=1, start_new_session=True)
                if self._feedback_enabled:
                    self._feedback_writer = FeedbackWriter(self._worker.stdin)
                self._reader = threading.Thread(target=self._read, daemon=True, name="ae01-input")
                self._reader.start()
                self._started = True
            except Exception as exc:
                self._error = str(exc)
                self.get_logger().error(f"AE01 SDK unavailable: {exc}")

        def _read(self):
            for line in self._worker.stdout:
                if not line.startswith(PACKET_PREFIX):
                    continue
                try:
                    packet = json.loads(line[len(PACKET_PREFIX):])
                    with self._lock:
                        self._packet = packet  # Bounded latest snapshot; never replay a backlog.
                except ValueError as exc:
                    self._error = f"AE01 SDK packet: {exc}"

        def _on_control(self, msg):
            self._control = (time.monotonic(), msg)
            self._publish_feedback()

        def _on_stop(self, msg):
            self._hardware_stopped = bool(msg.data)
            self._publish_feedback()

        def _on_wrench(self, msg):
            try:
                self._feedback_input.observe(msg)
                self._feedback_error = ""
            except (ValueError, TypeError, OverflowError) as exc:
                self._feedback_input.forces = {}
                self._feedback_error = str(exc)

        def _publish_feedback(self):
            if self._feedback_writer is None or not self._alive():
                return
            enabled = {side: self.core.valid(side) and not self._error and not self._hardware_stopped
                       for side in ('left', 'right')}
            packet = self._feedback_input.packet(self._control, enabled)
            if not self._feedback_writer.send(packet):
                self._feedback_error = 'AE01 feedback pipe unavailable; watchdog will clear stale force'

        def _alive(self):
            return self._worker is not None and self._worker.poll() is None

        def _tick(self):
            with self._lock:
                packet, self._packet = self._packet, None
            if packet is not None:
                try:
                    self.core.observe(packet)
                    self._feedback_report = packet.get('feedback', {})
                    self._feedback_seen_s = float(packet['monotonic_s'])
                except (ValueError, TypeError, KeyError) as exc:
                    self.core.sides.clear()
                    self._error = str(exc)
            command = self.core.command(self._alive() and not self._error)
            topic = output_topic(self.get_parameter("hil_mode").value,
                self.get_parameter("workbench_mode").value, self._control, time.monotonic())
            if not command or not topic:
                self._last_published = None
                return
            q, left, right = command
            identity = (topic, left, right, *(self.core.sides.get(side, {}).get("robot_timestamp_ns", 0)
                                            for side in ("left", "right")))
            if identity == self._last_published:
                return
            msg = HandCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "ae01/robot"
            msg.source = HandCommand.SOURCE_VIVE  # Frozen teleop source=2, shared with Vive.
            msg.q_cmd, msg.left_enable, msg.right_enable = q, left, right
            self._pubs[topic].publish(msg)
            self._last_published = identity

        def _publish_status(self):
            alive = self._alive()
            error = self._error or self.core.error
            if self._started and not alive:
                error = error or f"AE01 SDK exited ({self._worker.poll()}); 请重启启动入口"
            payload = self.core.status(alive and not self._error, self._started, error)
            report_fresh = alive and 0 <= time.monotonic()-self._feedback_seen_s < self._feedback_input.max_age_s
            payload['feedback'] = {'rate_hz': FEEDBACK_HZ, **self._feedback_report, 'enabled': self._feedback_enabled,
                'input_fresh': bool(report_fresh and self._feedback_report.get('input_fresh')),
                **{side+'_active': bool(report_fresh and self._feedback_report.get(side+'_active'))
                   for side in ('left', 'right')},
                'error': self._feedback_error or self._feedback_report.get('error', '')}
            self._status.publish(String(data=json.dumps(payload, separators=(",", ":"))))

        def destroy_node(self):
            if self._worker is not None:
                if self._feedback_writer is not None and self._alive():
                    self._feedback_writer.send(self._feedback_input.packet(None, {}))
                if self._worker.stdin is not None:
                    self._worker.stdin.close()  # EOF immediately disables force output in the SDK child.
                if self._worker.poll() is None:
                    self._worker.terminate()
                    try:
                        self._worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._worker.kill()
                        self._worker.wait(timeout=3)
                if self._reader is not None:
                    self._reader.join(timeout=1)
                self._worker.stdout.close()
                self._worker = None
            return super().destroy_node()

    rclpy.init(args=args)
    node = AE01Node()

    def stop(_sig, _frame):
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
