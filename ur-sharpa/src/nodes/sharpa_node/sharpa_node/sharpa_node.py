"""SharpA Wave SDK adapter using the shared UR ROS message contract."""

from __future__ import annotations

import array
import os
import threading
import time
from typing import Any

import numpy as np


class SharpaStateCore:
    """Validate and normalize a driver snapshot before it crosses ROS."""

    @staticmethod
    def normalize(snapshot: Any) -> dict[str, Any]:
        left = np.asarray(snapshot.left_joint, dtype=np.float32).reshape(-1)
        right = np.asarray(snapshot.right_joint, dtype=np.float32).reshape(-1)
        if left.shape != (22,) or right.shape != (22,):
            raise ValueError("SharpA joint state must contain 22 values per hand")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("SharpA joint state contains non-finite values")
        return {"timestamp_ns": int(snapshot.timestamp_ns), "left": left, "right": right,
                "left_valid": bool(snapshot.left_joint_valid), "right_valid": bool(snapshot.right_joint_valid)}


class ManusTeleopCommandCache:
    """Thread-safe per-side cache of received Manus teleop targets.

    This cache describes operator intent for the dataset, independently of
    whether SharpA is running or the hardware arbitration accepts the target.
    Freshness still prevents an old target from being recorded as a new action.
    """

    def __init__(self, freshness_timeout_s: float = 0.25) -> None:
        self.freshness_timeout_ns = int(float(freshness_timeout_s) * 1.0e9)
        if self.freshness_timeout_ns <= 0:
            raise ValueError("hand command freshness timeout must be positive")
        self._lock = threading.Lock()
        self._q_cmd = np.zeros(44, dtype=np.float32)
        self._valid = {"left": False, "right": False}
        self._timestamp_ns = {"left": 0, "right": 0}
        self._source = {"left": 0, "right": 0}

    def invalidate(self, side: str | None = None) -> None:
        with self._lock:
            for current in (("left", "right") if side is None else (side,)):
                self._valid[current] = False

    def record(
        self,
        dispatched: dict[str, Any],
        *,
        source: int,
        timestamp_ns: int | None = None,
    ) -> None:
        stamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        with self._lock:
            for side, offset in (("left", 0), ("right", 22)):
                if side not in dispatched:
                    continue
                values = np.asarray(dispatched[side], dtype=np.float32).reshape(-1)
                if values.shape != (22,) or not np.isfinite(values).all():
                    self._valid[side] = False
                    continue
                self._q_cmd[offset:offset + 22] = values
                self._valid[side] = True
                self._timestamp_ns[side] = stamp
                self._source[side] = int(source)

    def snapshot(
        self, now_ns: int | None = None
    ) -> tuple[np.ndarray, bool, int, int]:
        current = time.time_ns() if now_ns is None else int(now_ns)
        with self._lock:
            ages = {
                side: current - self._timestamp_ns[side]
                for side in ("left", "right")
            }
            valid = bool(
                all(self._valid.values())
                and all(0 <= age <= self.freshness_timeout_ns for age in ages.values())
                and self._source["left"] == self._source["right"]
                and np.isfinite(self._q_cmd).all()
            )
            return (
                self._q_cmd.copy(),
                valid,
                max(self._timestamp_ns.values()),
                self._source["left"] if valid else 0,
            )


# Keep the old import name for downstream code while changing the semantics to
# the received Manus action rather than a successfully dispatched command.
SuccessfulHandCommandCache = ManusTeleopCommandCache


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import Bool
    from ur_msgs.msg import (
        ControlStatus,
        DeformationFrame,
        HandCommand,
        HandStateFrame,
        HardwareCommandResult,
        TauFrame,
        WrenchFrame,
    )
    from ur_common.drivers import SharpADevicePair
    from .arbitration import SharpaState, Source, arbitrate

    class RosSharpaNode(Node):
        def __init__(self):
            super().__init__("sharpa_node")
            self.declare_parameter("sdk_root", os.environ.get("SHARPA_SDK_ROOT", ""))
            self.declare_parameter("skip_tactile", False)
            self.declare_parameter("command_freshness_timeout_s", 0.25)
            self.driver = SharpADevicePair(
                sdk_root=str(self.get_parameter("sdk_root").value),
                enable_control=True,
                skip_tactile=bool(self.get_parameter("skip_tactile").value),
            )
            self._driver_error = ""
            try:
                self.driver.connect()
            except Exception as exc:
                # Hardware absence is a runtime state, not a reason to kill the
                # graph. Keep publishing invalid facts so status/recording can
                # explain why the gate is closed.
                self._driver_error = str(exc)
                self.get_logger().error(f"SharpA hardware unavailable: {exc}")
            self.pub = self.create_publisher(HandStateFrame, "/sharpa/state", 10)
            self.tau_pub = self.create_publisher(TauFrame, "/sharpa/tau", 10)
            self.wrench_pub = self.create_publisher(WrenchFrame, "/sharpa/wrench", 10)
            self.deformation_pub = self.create_publisher(DeformationFrame, "/sharpa/deformation", 1)
            self.result = self.create_publisher(HardwareCommandResult, "/sharpa/command_result", 10)
            self._sharpa_state = SharpaState.STOPPED
            self._active_source = Source.UNKNOWN
            self._stopped = False
            self._command_cache = ManusTeleopCommandCache(
                float(self.get_parameter("command_freshness_timeout_s").value)
            )
            self.create_subscription(HandCommand, "/sharpa/command", self._command, 10)
            self.create_subscription(ControlStatus, "/control/status", self._status, 10)
            self.create_subscription(Bool, "/hardware/stop", self._stop, 10)
            self.timer = self.create_timer(1.0 / 30.0, self._state)

        def _status(self, message):
            try:
                self._sharpa_state = SharpaState(int(message.sharpa_state))
            except ValueError:
                self._sharpa_state = SharpaState.STOPPED
            try:
                self._active_source = Source(int(message.active_source))
            except ValueError:
                self._active_source = Source.UNKNOWN

        def _stop(self, message):
            self._stopped = bool(message.data)
            if self._stopped:
                self.driver.hold()

        def _state(self):
            msg = HandStateFrame(); msg.timestamp_ns = time.time_ns()
            if self._driver_error:
                msg.left_valid = False; msg.right_valid = False; self.pub.publish(msg); return
            try:
                raw = self.driver.read()
                snapshot = SharpaStateCore.normalize(raw)
            except Exception as exc:
                self.get_logger().error(str(exc)); msg.left_valid = False; msg.right_valid = False; self.pub.publish(msg); return
            stamp = snapshot["timestamp_ns"]
            msg.timestamp_ns = stamp
            msg.q = np.concatenate((snapshot["left"], snapshot["right"])).tolist()
            msg.left_valid = snapshot["left_valid"]; msg.right_valid = snapshot["right_valid"]
            msg.tau = np.concatenate((raw.left_tau, raw.right_tau)).tolist()
            msg.tau_valid = bool(raw.left_tau_valid and raw.right_tau_valid)
            msg.dq = [0.0] * 44; msg.dq_valid = False
            q_cmd, q_cmd_valid, q_cmd_timestamp_ns, q_cmd_source = (
                self._command_cache.snapshot()
            )
            msg.q_cmd = q_cmd.tolist()
            msg.q_cmd_valid = q_cmd_valid
            msg.q_cmd_timestamp_ns = q_cmd_timestamp_ns
            msg.q_cmd_source = q_cmd_source
            self.pub.publish(msg)

            tau = TauFrame(); tau.timestamp_ns = stamp
            tau.left = np.asarray(raw.left_tau, dtype=np.float32).tolist(); tau.right = np.asarray(raw.right_tau, dtype=np.float32).tolist()
            tau.left_valid = bool(raw.left_tau_valid); tau.right_valid = bool(raw.right_tau_valid); self.tau_pub.publish(tau)

            wrench = WrenchFrame(); wrench.timestamp_ns = stamp
            wrench.left = np.asarray(raw.left_wrench, dtype=np.float32).reshape(-1).tolist(); wrench.right = np.asarray(raw.right_wrench, dtype=np.float32).reshape(-1).tolist()
            wrench.left_valid = np.asarray(raw.left_wrench_valid, dtype=bool).tolist(); wrench.right_valid = np.asarray(raw.right_wrench_valid, dtype=bool).tolist(); self.wrench_pub.publish(wrench)

            deformation = DeformationFrame(); deformation.timestamp_ns = stamp
            deformation.left = array.array("B", np.asarray(raw.left_deformation, dtype=np.uint8).reshape(-1).tobytes()); deformation.right = array.array("B", np.asarray(raw.right_deformation, dtype=np.uint8).reshape(-1).tobytes())
            deformation.left_valid = np.asarray(raw.left_deformation_valid, dtype=bool).tolist(); deformation.right_valid = np.asarray(raw.right_deformation_valid, dtype=bool).tolist(); self.deformation_pub.publish(deformation)

        def _command(self, command):
            result = HardwareCommandResult(); result.source = command.source; result.action_id = command.action_id; result.revision = command.revision; result.step_index = command.step_index; result.device = "sharpa"; result.timestamp_ns = time.time_ns()
            try:
                try:
                    command_source = Source(int(command.source))
                except ValueError:
                    command_source = Source.UNKNOWN
                values = np.asarray(command.q_cmd, dtype=np.float64)
                if values.shape != (44,) or not np.isfinite(values).all():
                    if command_source == Source.VIVE:
                        if bool(command.left_enable):
                            self._command_cache.invalidate("left")
                        if bool(command.right_enable):
                            self._command_cache.invalidate("right")
                    raise ValueError("q_cmd must have 44 finite values")

                # SOURCE_VIVE is the shared source id used by Manus hand
                # teleop. Capture each enabled side before hardware gating so
                # recording works while SharpA is paused or unavailable.
                if command_source == Source.VIVE:
                    received = {}
                    if bool(command.left_enable):
                        received["left"] = values[:22]
                    if bool(command.right_enable):
                        received["right"] = values[22:]
                    self._command_cache.record(
                        received,
                        source=int(command.source),
                        timestamp_ns=time.time_ns(),
                    )

                if self._stopped: raise RuntimeError("hardware stop asserted")
                decision = arbitrate(self._sharpa_state, self._active_source, command_source)
                if not decision:
                    result.success = False; result.failure_code = "arbitration_rejected"; result.failure_message = decision.reason; self.result.publish(result); return
                if self._driver_error: raise RuntimeError(f"hardware unavailable: {self._driver_error}")
                self.driver.command(
                    values[:22], values[22:],
                    left_enable=bool(command.left_enable),
                    right_enable=bool(command.right_enable),
                )
                result.success = True
            except Exception as exc:
                result.success = False; result.failure_code = "sharpa_command_failed"; result.failure_message = str(exc)
            self.result.publish(result)

        def destroy_node(self):
            self.driver.close(); return super().destroy_node()

    rclpy.init(args=args); node = RosSharpaNode()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
