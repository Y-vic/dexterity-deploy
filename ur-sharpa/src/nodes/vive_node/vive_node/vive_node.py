from __future__ import annotations

import json
import time

import numpy as np

from .tracker_source import (
    AdaptivePoseFilter,
    DEFAULT_ZMQ_ADDRESS,
    ZmqTrackerReceiver,
    apply_relative_pose_delta,
    interpolate_pose,
    limit_ur_tcp_target,
    retarget_vive_wrist,
)


VIVE_PUBLISHER_HINT = (
    r"Windows: cd D:\project\DexCap\STEP1_collect_data_202408updates "
    r"&& python .\vive_process_send.py"
)
IK_BACKTRACK_FRACTIONS = (0.75, 0.5, 0.25, 0.125, 0.0625)


def validate_ik_joint_step(
    current, target, max_delta, side, stage="JOINT_STEP_LIMIT"
):
    current_array = np.asarray(current, dtype=np.float64).reshape(-1)
    target_array = np.asarray(target, dtype=np.float64).reshape(-1)
    limit = float(max_delta)
    if current_array.shape != (6,) or target_array.shape != (6,):
        raise ValueError("IK joint vectors must have shape (6,)")
    if not np.isfinite(current_array).all() or not np.isfinite(target_array).all():
        raise ValueError("IK joint vectors must be finite")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_delta must be finite and positive")
    delta = (target_array - current_array + np.pi) % (2.0 * np.pi) - np.pi
    rejected_axes = np.where(np.abs(delta) > limit + 1.0e-9)[0]
    if rejected_axes.size:
        raise RuntimeError(
            f"{side} IK joint step exceeds {limit:.3f} rad; "
            f"stage={stage} axes={rejected_axes.tolist()} "
            f"delta={np.round(delta, 4).tolist()} "
            f"current={np.round(current_array, 4).tolist()} "
            f"target={np.round(target_array, 4).tolist()}"
        )
    return delta


def solve_ik_with_cartesian_backtracking(
    kinematics,
    side,
    measured_joint,
    reference_joint,
    target_tcp,
    max_joint_step,
    fractions=IK_BACKTRACK_FRACTIONS,
):
    """Keep the current IK branch by shortening a rejected Cartesian step.

    Near a wrist singularity, the full Vive target can temporarily expose only
    the opposite analytic IK branch.  Moving part-way from the measured TCP
    normally leaves the current branch available.  Every fallback candidate
    still goes through FastIK's wrist-flip gate and the measured-joint step
    gate; this never turns a large branch jump into a clipped robot command.

    Returns ``(solution, active_rejection, used_fraction, direct_rejection)``.
    A fraction below 1 means a safe fallback was found.  Zero means every
    fallback was rejected and the returned solution is the solver's hold.
    """
    measured = np.asarray(measured_joint, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_joint, dtype=np.float64).reshape(-1)
    target = np.asarray(target_tcp, dtype=np.float64)
    if measured.shape != (6,) or reference.shape != (6,):
        raise ValueError("IK joint vectors must have shape (6,)")

    solver = kinematics.solver(side)
    saved_target = (
        None
        if solver.last_target_joint is None
        else np.asarray(solver.last_target_joint, dtype=np.float64).copy()
    )
    solution = np.asarray(
        kinematics.arm_ik(
            target,
            reference,
            side=side,
            ref_joint=reference,
        ),
        dtype=np.float64,
    )
    direct_rejection = solver.last_rejection
    if (
        direct_rejection is None
        or direct_rejection.get("stage") != "WRIST_FLIP_HOLD"
    ):
        return solution, direct_rejection, 1.0, direct_rejection

    measured_tcp = kinematics.arm_fk(measured)
    for fraction in fractions:
        amount = float(fraction)
        if not np.isfinite(amount) or amount <= 0.0 or amount >= 1.0:
            raise ValueError("IK backtrack fractions must be within (0, 1)")
        fallback_target = interpolate_pose(measured_tcp, target, amount)
        candidate = np.asarray(
            kinematics.arm_ik(
                fallback_target,
                reference,
                side=side,
                ref_joint=reference,
            ),
            dtype=np.float64,
        )
        if solver.last_rejection is not None:
            continue
        try:
            validate_ik_joint_step(
                measured,
                candidate,
                max_joint_step,
                side,
                stage="CARTESIAN_BACKTRACK_STEP_LIMIT",
            )
        except RuntimeError:
            # solve() records an accepted candidate immediately. Do not let a
            # candidate rejected by the second gate become the next IK branch
            # reference.
            solver.last_target_joint = (
                None if saved_target is None else saved_target.copy()
            )
            continue
        return candidate, None, amount, direct_rejection

    solver.last_target_joint = None if saved_target is None else saved_target.copy()
    solver.last_rejection = direct_rejection
    return solution, direct_rejection, 0.0, direct_rejection


def compact_ik_diagnostic(diagnostic):
    """Round numeric diagnostics so one rejection fits on one ROS log line."""

    def compact(value):
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, list):
            return [compact(item) for item in value]
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        return value

    return compact(dict(diagnostic))


def vive_side_validity(frame, stale_topics):
    """Return independent wrist validity; chest remains the shared reference."""
    stale = set(stale_topics)
    if frame is None or "chest" in stale:
        return {"left": False, "right": False}
    return {
        side: bool(
            f"{side}_elbow" not in stale and getattr(frame, side) is not None
        )
        for side in ("left", "right")
    }


def main(args=None):
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import String
    from ur_msgs.msg import ArmCommand, ControlStatus, UrStateFrame
    from ur_common.kinematics import UrCommonKinematics

    class ViveNode(Node):
        def __init__(self):
            super().__init__("vive_node")
            self.declare_parameter("zmq_address", DEFAULT_ZMQ_ADDRESS)
            self.declare_parameter("stale_timeout_s", 0.25)
            self.declare_parameter("frequency", 30.0)
            self.declare_parameter("base_offset", [0.05, 0.0, 0.40])
            self.declare_parameter("translation_scale", 1.0)
            self.declare_parameter("pose_filter_enabled", True)
            self.declare_parameter("position_deadband_m", 0.0015)
            self.declare_parameter("rotation_deadband_rad", 0.008726646259971648)
            self.declare_parameter("position_min_cutoff_hz", 1.5)
            self.declare_parameter("position_beta", 5.0)
            self.declare_parameter("rotation_min_cutoff_hz", 2.0)
            self.declare_parameter("rotation_beta", 1.5)
            self.declare_parameter("derivative_cutoff_hz", 1.0)
            self.declare_parameter("max_translation_delta_m", 0.03)
            self.declare_parameter("max_rotation_delta_rad", 0.12)
            self.declare_parameter("max_joint_step_rad", 0.35)
            self.declare_parameter("initial_joint_tolerance_rad", 0.05)
            self.address = str(self.get_parameter("zmq_address").value)
            self.stale_timeout_s = float(
                self.get_parameter("stale_timeout_s").value
            )
            self.frequency = float(self.get_parameter("frequency").value)
            self.base_offset = np.asarray(
                self.get_parameter("base_offset").value, dtype=np.float64
            ).reshape(3)
            self.translation_scale = float(
                self.get_parameter("translation_scale").value
            )
            self.pose_filter_enabled = bool(
                self.get_parameter("pose_filter_enabled").value
            )
            pose_filter_config = {
                "frequency_hz": self.frequency,
                "position_deadband_m": float(
                    self.get_parameter("position_deadband_m").value
                ),
                "rotation_deadband_rad": float(
                    self.get_parameter("rotation_deadband_rad").value
                ),
                "position_min_cutoff_hz": float(
                    self.get_parameter("position_min_cutoff_hz").value
                ),
                "position_beta": float(
                    self.get_parameter("position_beta").value
                ),
                "rotation_min_cutoff_hz": float(
                    self.get_parameter("rotation_min_cutoff_hz").value
                ),
                "rotation_beta": float(
                    self.get_parameter("rotation_beta").value
                ),
                "derivative_cutoff_hz": float(
                    self.get_parameter("derivative_cutoff_hz").value
                ),
            }
            self.max_translation_delta_m = float(
                self.get_parameter("max_translation_delta_m").value
            )
            self.max_rotation_delta_rad = float(
                self.get_parameter("max_rotation_delta_rad").value
            )
            self.max_joint_step_rad = float(
                self.get_parameter("max_joint_step_rad").value
            )
            self.initial_joint_tolerance_rad = float(
                self.get_parameter("initial_joint_tolerance_rad").value
            )
            if (
                not np.isfinite(self.max_translation_delta_m)
                or self.max_translation_delta_m <= 0.0
            ):
                raise ValueError("max_translation_delta_m must be finite and positive")
            if (
                not np.isfinite(self.max_rotation_delta_rad)
                or self.max_rotation_delta_rad <= 0.0
            ):
                raise ValueError("max_rotation_delta_rad must be finite and positive")
            if self.initial_joint_tolerance_rad <= 0.0:
                raise ValueError("initial_joint_tolerance_rad must be positive")
            if not np.isfinite(self.max_joint_step_rad) or self.max_joint_step_rad <= 0.0:
                raise ValueError("max_joint_step_rad must be finite and positive")
            self.pose_filters = {
                side: AdaptivePoseFilter(**pose_filter_config)
                for side in ("left", "right")
            }
            self.receiver = ZmqTrackerReceiver(self.address)
            self.receiver.start()
            from ur_common.safety import SafetyConfig

            self.kinematics = UrCommonKinematics(
                config=SafetyConfig(
                    enable_jump_gate=False,
                    enable_flip_hold=True,
                    enable_joint_clip=False,
                    frequency=self.frequency,
                )
            )
            self.seeds = {
                "left": np.zeros(6, dtype=np.float64),
                "right": np.zeros(6, dtype=np.float64),
            }
            self.measured_joint = {
                "left": np.zeros(6, dtype=np.float64),
                "right": np.zeros(6, dtype=np.float64),
            }
            self.measured_capture_tcp = {}
            self.ur_state_valid = False
            self._ur_state_error = ""
            self.control_state = ControlStatus.UR_STOPPED
            self._head_origin = None
            self._tracker_origin = {}
            self._robot_origin = {}
            self._calibrated = {"left": False, "right": False}
            self._first_command_pending = {"left": True, "right": True}
            self._motion_ready = {"left": False, "right": False}
            self._last_wrist_sequence = {"left": 0, "right": 0}
            self._last_command_monotonic = 0.0
            self._last_command_monotonic_by_side = {"left": 0.0, "right": 0.0}
            self._last_ik_error = {"left": "", "right": ""}
            self._ik_rejection_count = {"left": 0, "right": 0}
            self._ik_rejection_stage = {"left": "", "right": ""}
            self._last_ik_rejection_log_at = {"left": 0.0, "right": 0.0}
            self._last_ik_backtrack_log_at = {"left": 0.0, "right": 0.0}
            self.command_pub = self.create_publisher(ArmCommand, "/ur/command", 10)
            self.status_pub = self.create_publisher(
                String, "/teleop/vive_status", 10
            )
            self.create_subscription(UrStateFrame, "/ur/state", self._on_ur, 10)
            self.create_subscription(
                ControlStatus, "/control/status", self._on_control, 10
            )
            self.timer = self.create_timer(1.0 / self.frequency, self._tick)

        def _on_ur(self, message):
            if not bool(message.valid) or not bool(message.normal_mode):
                if self.ur_state_valid:
                    self._reset_alignment()
                self.ur_state_valid = False
                self._ur_state_error = message.status_message or "invalid UR state"
                return
            if not self.ur_state_valid:
                self._reset_alignment()
            try:
                measured_joint = {
                    "left": np.asarray(message.left_joint, dtype=np.float64),
                    "right": np.asarray(message.right_joint, dtype=np.float64),
                }
                measured_capture_tcp = {}
                for side in ("left", "right"):
                    if (
                        measured_joint[side].shape != (6,)
                        or not np.isfinite(measured_joint[side]).all()
                    ):
                        raise ValueError(f"invalid {side} UR joint state")
                    ur_base_tcp = self.kinematics.arm_fk(measured_joint[side])
                    measured_capture_tcp[side] = (
                        self.kinematics.geometry.ur_base_tcp_to_capture_tcp(
                            ur_base_tcp, side
                        )
                    )
            except Exception as exc:
                self.ur_state_valid = False
                self._ur_state_error = f"UR FK failed: {exc}"
                return
            self.measured_joint = measured_joint
            if self.control_state != ControlStatus.UR_RUNNING:
                self.seeds = {
                    side: measured_joint[side].copy() for side in ("left", "right")
                }
            self.measured_capture_tcp = measured_capture_tcp
            self.ur_state_valid = True
            self._ur_state_error = ""

        def _reset_alignment(self, side=None):
            sides = ("left", "right") if side is None else (side,)
            for current_side in sides:
                self._calibrated[current_side] = False
                self._first_command_pending[current_side] = True
                self._motion_ready[current_side] = False
                self._tracker_origin.pop(current_side, None)
                self._robot_origin.pop(current_side, None)
                self._last_wrist_sequence[current_side] = 0
                self.pose_filters[current_side].reset()
                self.kinematics.reset(side=current_side)
                self._last_ik_error[current_side] = ""
            if side is None:
                self._head_origin = None

        def _on_control(self, message):
            previous = self.control_state
            self.control_state = int(message.ur_state)
            if self.control_state != ControlStatus.UR_RUNNING:
                self._reset_alignment()
            elif previous != ControlStatus.UR_RUNNING:
                self._reset_alignment()

        def _align_side(self, target, side):
            if not self._calibrated[side]:
                if not self.ur_state_valid or side not in self.measured_capture_tcp:
                    raise RuntimeError(
                        f"waiting for valid UR state before {side} Vive alignment"
                    )
                self._tracker_origin[side] = target.copy()
                self._robot_origin[side] = self.measured_capture_tcp[side].copy()
                self.seeds[side] = self.measured_joint[side].copy()
                self.pose_filters[side].reset(
                    self._robot_origin[side], time.monotonic()
                )
                self._calibrated[side] = True
                return self._robot_origin[side].copy()
            return apply_relative_pose_delta(
                target,
                self._tracker_origin[side],
                self._robot_origin[side],
                self.translation_scale,
            )

        def _report_ik_rejection(self, side, diagnostic):
            payload = compact_ik_diagnostic(diagnostic)
            stage = str(payload.get("stage", "IK_REJECTED"))
            self._ik_rejection_count[side] += 1
            now = time.monotonic()
            if (
                stage != self._ik_rejection_stage[side]
                or now - self._last_ik_rejection_log_at[side] >= 0.5
            ):
                payload["side"] = side
                payload["consecutive"] = self._ik_rejection_count[side]
                self.get_logger().warning(
                    "IK_REJECT "
                    + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                )
                self._last_ik_rejection_log_at[side] = now
            self._ik_rejection_stage[side] = stage

        def _clear_ik_rejection(self, side):
            count = self._ik_rejection_count[side]
            if count:
                self.get_logger().info(
                    f"IK_RECOVER side={side} after={count} rejected frame(s) "
                    f"last_stage={self._ik_rejection_stage[side]}"
                )
            self._ik_rejection_count[side] = 0
            self._ik_rejection_stage[side] = ""

        def _report_ik_backtrack(self, side, fraction, direct_rejection):
            now = time.monotonic()
            if now - self._last_ik_backtrack_log_at[side] < 0.5:
                return
            diagnostic = compact_ik_diagnostic(direct_rejection)
            self.get_logger().info(
                f"IK_BACKTRACK side={side} fraction={fraction:.4f} "
                f"direct_min_delta_rad="
                f"{diagnostic.get('minimum_max_delta_rad', 'unknown')}"
            )
            self._last_ik_backtrack_log_at[side] = now

        def _tick(self):
            frame = self.receiver.latch.peek()
            stale_topics = (
                ()
                if frame is None
                else frame.stale_topics(self.stale_timeout_s)
            )
            side_tracking_valid = vive_side_validity(frame, stale_topics)
            tracking_valid = any(side_tracking_valid.values())
            if frame is None or "chest" in stale_topics:
                self._reset_alignment()
            elif frame is not None:
                for side in ("left", "right"):
                    if f"{side}_elbow" in stale_topics or getattr(frame, side) is None:
                        self._reset_alignment(side)
            if tracking_valid and self.control_state == ControlStatus.UR_RUNNING and self.ur_state_valid:
                for side in ("left", "right"):
                    if not side_tracking_valid[side]:
                        continue
                    try:
                        sequence = frame.wrist_sequence(side)
                        if sequence <= self._last_wrist_sequence[side]:
                            continue
                        was_calibrated = self._calibrated[side]
                        target, self._head_origin = retarget_vive_wrist(
                            frame, side, self._head_origin, self.base_offset
                        )
                        target = self.kinematics.gate_wrist(target, side)[0]
                        target = self._align_side(target, side)
                        self._last_wrist_sequence[side] = sequence
                        if not was_calibrated:
                            # This sample establishes this arm's reference only.
                            # The next valid sample is the first one allowed to
                            # produce a delta and an IK target.
                            continue
                        if self.pose_filter_enabled:
                            target = self.pose_filters[side].update(
                                target, time.monotonic()
                            )
                        ur_base_target = limit_ur_tcp_target(
                            self.kinematics,
                            side,
                            self.measured_joint[side],
                            target,
                            self.max_translation_delta_m,
                            self.max_rotation_delta_rad,
                        )
                        joint_limit = (
                            self.initial_joint_tolerance_rad
                            if self._first_command_pending[side]
                            else self.max_joint_step_rad
                        )
                        joint_stage = (
                            "INITIAL_STEP_LIMIT"
                            if self._first_command_pending[side]
                            else "JOINT_STEP_LIMIT"
                        )
                        (
                            solution,
                            solver_rejection,
                            backtrack_fraction,
                            direct_rejection,
                        ) = solve_ik_with_cartesian_backtracking(
                            self.kinematics,
                            side,
                            self.measured_joint[side],
                            self.seeds[side],
                            ur_base_target,
                            joint_limit,
                        )
                        solution = np.asarray(solution, dtype=np.float32)
                        if solver_rejection is not None:
                            self._report_ik_rejection(side, solver_rejection)
                        elif backtrack_fraction < 1.0:
                            self._report_ik_backtrack(
                                side, backtrack_fraction, direct_rejection
                            )
                        try:
                            validate_ik_joint_step(
                                self.measured_joint[side],
                                solution,
                                joint_limit,
                                side,
                                stage=joint_stage,
                            )
                        except RuntimeError as exc:
                            self._report_ik_rejection(
                                side,
                                {
                                    "stage": joint_stage,
                                    "limit_rad": joint_limit,
                                    "detail": str(exc),
                                },
                            )
                            raise
                        if solver_rejection is None:
                            self._clear_ik_rejection(side)
                        self._first_command_pending[side] = False
                        self.seeds[side] = solution
                        self._motion_ready[side] = True
                        command = ArmCommand()
                        command.source = ArmCommand.SOURCE_VIVE
                        command.period_s = 1.0 / self.frequency
                        command.left_joint = solution.tolist() if side == "left" else [0.0] * 6
                        command.right_joint = solution.tolist() if side == "right" else [0.0] * 6
                        command.left_enable = side == "left"
                        command.right_enable = side == "right"
                        self.command_pub.publish(command)
                        now = time.monotonic()
                        self._last_command_monotonic = now
                        self._last_command_monotonic_by_side[side] = now
                        self._last_ik_error[side] = ""
                    except Exception as exc:
                        error = str(exc)
                        if "stage=" not in error:
                            self._report_ik_rejection(
                                side,
                                {
                                    "stage": "VIVE_IK_PIPELINE_ERROR",
                                    "detail": error,
                                },
                            )
                        self._reset_alignment(side)
                        self._last_ik_error[side] = error
            self._publish_status(frame, stale_topics, tracking_valid, side_tracking_valid)

        def _publish_status(self, frame, stale_topics, tracking_valid, side_tracking_valid):
            stats = self.receiver.latch.stats()
            error = self.receiver.start_error or stats.get("last_error", "")
            if not error and frame is None:
                missing = ", ".join(stats.get("missing_topics", []))
                error = (
                    f"未收到 Vive 数据（缺少 {missing}）。{VIVE_PUBLISHER_HINT}"
                )
            elif not error and stale_topics:
                error = "Vive topic 过期: " + ", ".join(stale_topics)
            elif not error and self._ur_state_error:
                error = self._ur_state_error
            elif not error and any(self._last_ik_error.values()):
                error = "Vive IK: " + "; ".join(
                    f"{side}={value}"
                    for side, value in self._last_ik_error.items() if value
                )
            ready_arms = {
                side: bool(
                    side_tracking_valid[side]
                    and self.ur_state_valid
                    and self.control_state == ControlStatus.UR_RUNNING
                    and self._motion_ready[side]
                    and not self._last_ik_error[side]
                )
                for side in ("left", "right")
            }
            data = {
                "connected": bool(tracking_valid),
                "valid": bool(tracking_valid),
                "command_ready": any(ready_arms.values()),
                "valid_arms": dict(side_tracking_valid),
                "command_ready_arms": ready_arms,
                "receiver_running": self.receiver.running,
                "socket_ready": self.receiver.socket_ready,
                "address": self.address,
                "operator_hint": VIVE_PUBLISHER_HINT,
                "control_running": self.control_state == ControlStatus.UR_RUNNING,
                "calibrated": all(self._calibrated.values()),
                "calibrated_arms": dict(self._calibrated),
                "last_command_age_s_by_arm": {
                    side: (
                        None if self._last_command_monotonic_by_side[side] <= 0.0
                        else round(time.monotonic() - self._last_command_monotonic_by_side[side], 3)
                    )
                    for side in ("left", "right")
                },
                "last_command_age_s": (
                    None
                    if self._last_command_monotonic <= 0.0
                    else round(
                        time.monotonic() - self._last_command_monotonic, 3
                    )
                ),
                "error": error,
                **stats,
            }
            status = String()
            status.data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            self.status_pub.publish(status)

        def destroy_node(self):
            self.receiver.stop()
            return super().destroy_node()

    rclpy.init(args=args)
    node = ViveNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
