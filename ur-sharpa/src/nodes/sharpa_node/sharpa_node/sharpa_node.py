"""SharpA Wave SDK adapter using the shared UR ROS message contract."""

from __future__ import annotations

import array
import os
import threading
import time
from typing import Any

import numpy as np


ACTOR_SOURCE = 3


def validate_hand_command(command: Any) -> tuple[np.ndarray, int, bool, bool, bool]:
    """Validate a command before either dry-run ACK or SDK dispatch."""
    source = int(command.source)
    values = np.asarray(command.q_cmd, dtype=np.float64)
    if values.shape != (44,) or not np.isfinite(values).all():
        raise ValueError("q_cmd must have 44 finite values")
    left_enable = bool(command.left_enable)
    right_enable = bool(command.right_enable)
    if not left_enable and not right_enable:
        raise ValueError("at least one SharpA hand must be enabled")
    dry_run = bool(command.dry_run)
    if dry_run and source != ACTOR_SOURCE:
        raise ValueError("only Actor commands may use dry_run")
    return values, source, left_enable, right_enable, dry_run


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


class SuccessfulHandCommandCache:
    """Last complete 44D target successfully handed to the SharpA SDK.

    This mirrors pnd-sharpa's ``last_successful_q_cmd`` contract. A partial
    command is never combined with an older command from the other hand. The
    value remains a zero-order hold until a new complete command succeeds or
    the hardware execution state invalidates it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._q_cmd = np.full(44, np.nan, dtype=np.float32)
        self._valid = False
        self._timestamp_ns = 0
        self._source = 0

    def invalidate(self) -> None:
        with self._lock:
            self._q_cmd.fill(np.nan)
            self._valid = False
            self._timestamp_ns = 0
            self._source = 0

    def record(
        self,
        dispatched: dict[str, Any],
        *,
        source: int,
        timestamp_ns: int | None = None,
    ) -> None:
        stamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        if set(dispatched) != {"left", "right"}:
            self.invalidate()
            return
        left = np.asarray(dispatched["left"], dtype=np.float32).reshape(-1)
        right = np.asarray(dispatched["right"], dtype=np.float32).reshape(-1)
        if (
            left.shape != (22,)
            or right.shape != (22,)
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or stamp <= 0
            or int(source) <= 0
        ):
            self.invalidate()
            return
        with self._lock:
            self._q_cmd[:22] = left
            self._q_cmd[22:] = right
            self._valid = True
            self._timestamp_ns = stamp
            self._source = int(source)

    def snapshot(self) -> tuple[np.ndarray, bool, int, int]:
        with self._lock:
            return (
                self._q_cmd.copy(),
                bool(self._valid),
                int(self._timestamp_ns),
                int(self._source),
            )


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import Bool, String
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
    from ur_common.execution_log import execution_event_json
    from .arbitration import SharpaState, Source, arbitrate

    class RosSharpaNode(Node):
        def __init__(self):
            super().__init__("sharpa_node")
            self.declare_parameter("sdk_root", os.environ.get("SHARPA_SDK_ROOT", ""))
            self.declare_parameter("skip_tactile", False)
            self._sdk_lock = threading.RLock()
            self.driver = SharpADevicePair(
                sdk_root=str(self.get_parameter("sdk_root").value),
                enable_control=True,
                skip_tactile=bool(self.get_parameter("skip_tactile").value),
            )
            self._driver_error = ""
            try:
                self.driver.connect()
                self.driver.set_floating()
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
            self._execution_event_pub = self.create_publisher(
                String, "/deploy/execution_event", 100
            )
            self._sharpa_state = SharpaState.STOPPED
            self._active_source = Source.UNKNOWN
            self._stopped = False
            self._floating = True
            self._command_cache = SuccessfulHandCommandCache()
            self.create_subscription(HandCommand, "/sharpa/command", self._command, 10)
            self.create_subscription(ControlStatus, "/control/status", self._status, 10)
            self.create_subscription(Bool, "/hardware/stop", self._stop, 10)
            self.timer = self.create_timer(1.0 / 30.0, self._state)

        def _status(self, message):
            with self._sdk_lock:
                previous_state = self._sharpa_state
                previous_source = self._active_source
                try:
                    self._sharpa_state = SharpaState(int(message.sharpa_state))
                except ValueError:
                    self._sharpa_state = SharpaState.STOPPED
                try:
                    self._active_source = Source(int(message.active_source))
                except ValueError:
                    self._active_source = Source.UNKNOWN

                if self._active_source != previous_source:
                    self._command_cache.invalidate()
                if (
                    self._sharpa_state == SharpaState.STOPPED
                    and previous_state != SharpaState.STOPPED
                ):
                    self._set_floating()
                elif (
                    self._sharpa_state == SharpaState.RUNNING
                    and previous_state != SharpaState.RUNNING
                    and not self._stopped
                ):
                    self._set_position_control()

        def _set_floating(self):
            with self._sdk_lock:
                self._command_cache.invalidate()
                if self._floating:
                    return
                try:
                    self.driver.set_floating()
                    self._floating = True
                except Exception as exc:
                    self.get_logger().error(f"failed to set SharpA floating mode: {exc}")

        def _set_position_control(self):
            with self._sdk_lock:
                if not self._floating:
                    return
                self._command_cache.invalidate()
                try:
                    self.driver.set_position_control()
                    self._floating = False
                except Exception as exc:
                    self.get_logger().error(
                        f"failed to restore SharpA position mode: {exc}"
                    )

        def _stop(self, message):
            with self._sdk_lock:
                self._stopped = bool(message.data)
                if self._stopped:
                    self._set_floating()
                elif self._sharpa_state == SharpaState.RUNNING:
                    self._set_position_control()

        def _state(self):
            msg = HandStateFrame(); msg.timestamp_ns = time.time_ns()
            if self._driver_error:
                msg.left_valid = False; msg.right_valid = False; self.pub.publish(msg); return
            try:
                # Match pnd-sharpa: measured q and the most recent successfully
                # dispatched q_cmd are captured under one SDK-side lock and
                # published in the same feedback message.
                with self._sdk_lock:
                    raw = self.driver.read()
                    snapshot = SharpaStateCore.normalize(raw)
                    command_snapshot = self._command_cache.snapshot()
            except Exception as exc:
                self.get_logger().error(str(exc)); msg.left_valid = False; msg.right_valid = False; self.pub.publish(msg); return
            stamp = snapshot["timestamp_ns"]
            msg.timestamp_ns = stamp
            msg.q = np.concatenate((snapshot["left"], snapshot["right"])).tolist()
            msg.left_valid = snapshot["left_valid"]; msg.right_valid = snapshot["right_valid"]
            msg.tau = np.concatenate((raw.left_tau, raw.right_tau)).tolist()
            msg.tau_valid = bool(raw.left_tau_valid and raw.right_tau_valid)
            msg.dq = [0.0] * 44; msg.dq_valid = False
            q_cmd, q_cmd_valid, q_cmd_timestamp_ns, q_cmd_source = command_snapshot
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
            result.dry_run = bool(command.dry_run)
            self._execution_event(
                "hardware_command_received", command, device="sharpa"
            )
            try:
                values, source, left_enable, right_enable, dry_run = (
                    validate_hand_command(command)
                )
                try:
                    command_source = Source(source)
                except ValueError:
                    command_source = Source.UNKNOWN
                if dry_run:
                    result.success = True
                    self.result.publish(result)
                    self._result_event(command, result)
                    return
                with self._sdk_lock:
                    if self._stopped:
                        raise RuntimeError("hardware stop asserted")
                    if command_source == Source.BIAS and self._floating:
                        # Reset/homing is an active machine operation and therefore
                        # uses the same POSITION mode as normal running.
                        self._set_position_control()
                    decision = arbitrate(
                        self._sharpa_state, self._active_source, command_source
                    )
                    if not decision:
                        result.success = False
                        result.failure_code = "arbitration_rejected"
                        result.failure_message = decision.reason
                        self.result.publish(result)
                        self._result_event(command, result)
                        return
                    if self._driver_error:
                        raise RuntimeError(
                            f"hardware unavailable: {self._driver_error}"
                        )
                    if left_enable or right_enable:
                        # Invalidate before the SDK call. A partial command or
                        # any SDK failure must not leave an older full command
                        # looking like the command paired with future state.
                        self._command_cache.invalidate()
                    sdk_started_ns = time.monotonic_ns()
                    self._execution_event(
                        "hardware_sdk_command_started", command, device="sharpa"
                    )
                    dispatched = self.driver.command(
                        values[:22], values[22:],
                        left_enable=left_enable,
                        right_enable=right_enable,
                        event_callback=lambda event, **fields: self._execution_event(
                            event, command, device="sharpa", **fields
                        ),
                    )
                    self._execution_event(
                        "hardware_sdk_command_completed",
                        command,
                        device="sharpa",
                        duration_ms=(time.monotonic_ns() - sdk_started_ns)
                        / 1_000_000.0,
                        success=True,
                    )
                    self._command_cache.record(
                        dispatched,
                        source=int(command.source),
                        timestamp_ns=time.time_ns(),
                    )
                result.success = True
            except Exception as exc:
                result.success = False; result.failure_code = "sharpa_command_failed"; result.failure_message = str(exc)
                self._execution_event(
                    "hardware_sdk_command_failed",
                    command,
                    device="sharpa",
                    success=False,
                    error=str(exc),
                )
            self.result.publish(result)
            self._result_event(command, result)

        def _execution_event(self, event, command, **fields):
            try:
                message = String()
                message.data = execution_event_json(
                    "sharpa_node",
                    event,
                    action_id=str(command.action_id),
                    revision=int(command.revision),
                    step_index=int(command.step_index),
                    source=int(command.source),
                    dry_run=bool(command.dry_run),
                    **fields,
                )
                self._execution_event_pub.publish(message)
            except Exception as exc:
                self.get_logger().warning(f"failed to publish execution event: {exc}")

        def _result_event(self, command, result):
            self._execution_event(
                "hardware_command_result_published",
                command,
                device="sharpa",
                success=bool(result.success),
                failure_code=str(result.failure_code),
                failure_message=str(result.failure_message),
            )

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
