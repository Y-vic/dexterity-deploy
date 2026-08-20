"""Quest relative-pose wrist retargeting for Adam Pro."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String

from quest_node.command_gate import (
    ADAM_COMMAND_JOINTS_19,
    positions_from_joint_arrays,
)
from quest_node.joy_indices import QuestJoyButton
from quest_node.adam_bimanual_ik import AdamBimanualIkSolver, Pose3


DEFAULT_MODEL = (
    "/opt/pnd/pnd_teleop/install/adam_description/share/adam_description/"
    "urdf/adam_pro/adam_pro.urdf"
)
EXECUTION_FRAMES = {
    "Head": "QuestExecutionHead",
    "Left": "QuestExecutionLeftHand",
    "Right": "QuestExecutionRightHand",
}


@dataclass(frozen=True)
class RetargetCalibration:
    alignment_rotation: np.ndarray
    tracker_initial: Mapping[str, Pose3]
    robot_initial: Mapping[str, Pose3]


def compute_alignment_rotation(
    left_position: np.ndarray, right_position: np.ndarray
) -> np.ndarray:
    """Return aligned-frame axes expressed in the Quest ROS frame."""

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = np.asarray(left_position, dtype=np.float64) - np.asarray(
        right_position,
        dtype=np.float64,
    )
    lateral -= np.dot(lateral, up) * up
    norm = float(np.linalg.norm(lateral))
    if norm < 0.05:
        raise ValueError("Quest hands are too close to define the calibration frame")
    lateral /= norm
    forward = np.cross(lateral, up)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1.0e-6:
        raise ValueError("Quest hand alignment is degenerate")
    forward /= forward_norm
    lateral = np.cross(up, forward)
    lateral /= np.linalg.norm(lateral)
    return np.column_stack((forward, lateral, up))


def relative_target_pose(
    current: Pose3,
    tracker_initial: Pose3,
    robot_initial: Pose3,
    alignment_rotation: np.ndarray,
    *,
    position_scale: float = 1.0,
) -> Pose3:
    """Map a tracked pose change onto the robot's calibration pose."""

    alignment = np.asarray(alignment_rotation, dtype=np.float64)
    position = robot_initial.position + position_scale * alignment.T @ (
        current.position - tracker_initial.position
    )
    rotation = (
        alignment.T
        @ current.rotation
        @ tracker_initial.rotation.T
        @ alignment
        @ robot_initial.rotation
    )
    return Pose3(position=position, rotation=rotation)


def targets_from_calibration(
    current: Mapping[str, Pose3],
    calibration: RetargetCalibration,
    *,
    position_scale: float = 1.0,
) -> dict[str, Pose3]:
    return {
        name: relative_target_pose(
            current[name],
            calibration.tracker_initial[name],
            calibration.robot_initial[name],
            calibration.alignment_rotation,
            position_scale=position_scale,
        )
        for name in ("Head", "Left", "Right")
    }


def pose3_status(pose: Pose3) -> dict[str, list[float]]:
    return {
        "position": [float(value) for value in pose.position],
        "quaternion_xyzw": [
            float(value) for value in Rotation.from_matrix(pose.rotation).as_quat()
        ],
    }


class QuestRetargetNode(Node):
    def __init__(self) -> None:
        super().__init__("quest_retarget")
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("model_path", DEFAULT_MODEL)
        self.declare_parameter("output_topic", "/joint_states")
        self.declare_parameter(
            "bias_joint_states_topic", "/adam_bias_command_joint_states"
        )
        self.declare_parameter("bias_state_timeout", 0.5)
        self.declare_parameter("bias_settle_time", 0.25)
        self.declare_parameter("tracking_status_topic", "/quest/webvr_status")
        self.declare_parameter("status_topic", "/quest/retarget_status")
        self.declare_parameter("joy_topic", "/_quest/joy")
        self.declare_parameter("control_loop_rate", 50.0)
        self.declare_parameter("tracking_timeout", 0.2)
        self.declare_parameter("enable_neck", True)
        self.declare_parameter("retarget_method", "nonlinear_ik")
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("iterations", 5)
        self.declare_parameter("solve_dt", 0.05)
        self.declare_parameter("solver", "daqp")
        self.declare_parameter("damping", 0.0)
        self.declare_parameter("lm_damping", 1.0)
        self.declare_parameter("line_search_steps", 10)
        self.declare_parameter("wrist_position_cost", 50.0)
        self.declare_parameter("wrist_orientation_cost", 1.0)
        self.declare_parameter("elbow_position_cost", 10.0)
        self.declare_parameter("smoothness_cost", 0.2)
        self.declare_parameter("posture_cost", 0.05)
        self.declare_parameter("shoulder_prior_wrist_position_cost", 20.0)
        self.declare_parameter("shoulder_prior_wrist_orientation_cost", 18.0)
        self.declare_parameter("shoulder_prior_orientation_cost", 2.0)
        self.declare_parameter("nonlinear_translation_cost", 50.0)
        self.declare_parameter("nonlinear_rotation_cost", 1.0)
        self.declare_parameter("nonlinear_posture_cost", 0.02)
        self.declare_parameter("nonlinear_smoothness_cost", 0.1)
        self.declare_parameter("nonlinear_filter_enabled", True)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.rate = float(self.get_parameter("control_loop_rate").value)
        self.tracking_timeout = float(self.get_parameter("tracking_timeout").value)
        self.bias_state_timeout = float(self.get_parameter("bias_state_timeout").value)
        self.bias_settle_time = float(self.get_parameter("bias_settle_time").value)
        self.enable_neck = bool(self.get_parameter("enable_neck").value)
        self.retarget_method = str(
            self.get_parameter("retarget_method").value
        ).lower()
        self.position_scale = float(self.get_parameter("position_scale").value)
        self.iterations = int(self.get_parameter("iterations").value)
        self.solve_dt = float(self.get_parameter("solve_dt").value)
        if (
            self.rate <= 0.0
            or self.tracking_timeout <= 0.0
            or self.bias_state_timeout <= 0.0
            or self.bias_settle_time < 0.0
            or self.position_scale <= 0.0
            or self.iterations <= 0
            or self.solve_dt <= 0.0
        ):
            raise ValueError(
                "rate, timeouts, scale, iterations and solve_dt must be positive"
            )
        self.solver = AdamBimanualIkSolver(
            str(self.get_parameter("model_path").value),
            retarget_method=self.retarget_method,
            solver=str(self.get_parameter("solver").value),
            damping=float(self.get_parameter("damping").value),
            lm_damping=float(self.get_parameter("lm_damping").value),
            line_search_steps=int(self.get_parameter("line_search_steps").value),
            wrist_position_cost=float(self.get_parameter("wrist_position_cost").value),
            wrist_orientation_cost=float(
                self.get_parameter("wrist_orientation_cost").value
            ),
            elbow_position_cost=float(
                self.get_parameter("elbow_position_cost").value
            ),
            smoothness_cost=float(self.get_parameter("smoothness_cost").value),
            posture_cost=float(self.get_parameter("posture_cost").value),
            shoulder_prior_wrist_position_cost=float(
                self.get_parameter("shoulder_prior_wrist_position_cost").value
            ),
            shoulder_prior_wrist_orientation_cost=float(
                self.get_parameter("shoulder_prior_wrist_orientation_cost").value
            ),
            shoulder_prior_orientation_cost=float(
                self.get_parameter("shoulder_prior_orientation_cost").value
            ),
            nonlinear_translation_cost=float(
                self.get_parameter("nonlinear_translation_cost").value
            ),
            nonlinear_rotation_cost=float(
                self.get_parameter("nonlinear_rotation_cost").value
            ),
            nonlinear_posture_cost=float(
                self.get_parameter("nonlinear_posture_cost").value
            ),
            nonlinear_smoothness_cost=float(
                self.get_parameter("nonlinear_smoothness_cost").value
            ),
            nonlinear_filter_enabled=bool(
                self.get_parameter("nonlinear_filter_enabled").value
            ),
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("output_topic").value),
            10,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("bias_joint_states_topic").value),
            self._bias_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("tracking_status_topic").value),
            self._status_callback,
            10,
        )
        self.create_subscription(
            Joy,
            str(self.get_parameter("joy_topic").value),
            self._joy_callback,
            10,
        )
        self.bias_positions: dict[str, float] = {}
        self.last_bias_time: float | None = None
        self.last_bias_change_time: float | None = None
        self.tracking_ready = False
        self.last_tracking_time: float | None = None
        self.last_source_sequence: int | None = None
        self.calibration: RetargetCalibration | None = None
        self.calibration_pending = False
        self.calibrate_button_pressed = False
        self.published = 0
        self.dropped = 0
        self.calibrations = 0
        self.last_error = ""
        self.solve_times_ms: deque[float] = deque(maxlen=256)
        self.last_wrist_errors = {"Left": math.inf, "Right": math.inf}
        self.last_targets: dict[str, Pose3] = {}
        self.last_solved_wrists: dict[str, Pose3] = {}
        self.last_elbow_outer_distances = {"Left": math.nan, "Right": math.nan}
        self.create_timer(1.0 / self.rate, self._tick)
        self.create_timer(2.0, self._log_status)
        self.get_logger().info(
            "Quest retarget ready: A maps gated Quest wrist deltas 1:1 "
            "from the current bias pose; Pink solves arms only, "
            f"method={self.retarget_method}, neck_tracking={self.enable_neck}, "
            "and waist remains at bias"
        )

    def _bias_callback(self, msg: JointState) -> None:
        if msg.header.frame_id and not msg.header.frame_id.startswith(
            "adam_bias_command:bias:"
        ):
            return
        try:
            positions = positions_from_joint_arrays(msg.name, msg.position)
        except ValueError as exc:
            self.last_error = f"bias:{exc}"
            return
        if all(name in positions for name in ADAM_COMMAND_JOINTS_19):
            if not self.bias_positions or any(
                abs(positions[name] - self.bias_positions[name]) > 1.0e-6
                for name in ADAM_COMMAND_JOINTS_19
            ):
                self.last_bias_change_time = time.monotonic()
            self.bias_positions = positions
            self.last_bias_time = time.monotonic()

    def _status_callback(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        ready = (
            status.get("connected") is True
            and status.get("calibrated") is True
            and status.get("tracking_fresh") is True
        )
        if ready and status.get("event") == "frame":
            self.last_tracking_time = time.monotonic()
            source_sequence = status.get("source_sequence")
            self.last_source_sequence = (
                int(source_sequence) if source_sequence is not None else None
            )
        calibration_valid = (
            status.get("connected") is True
            and status.get("calibrated") is True
        )
        if self.calibration is not None and not calibration_valid:
            self.calibration = None
            self.calibration_pending = False
        self.tracking_ready = ready

    def _joy_callback(self, msg: Joy) -> None:
        index = int(QuestJoyButton.R_A)
        pressed = len(msg.buttons) > index and msg.buttons[index] == 1
        if pressed and not self.calibrate_button_pressed:
            self.calibration_pending = True
        self.calibrate_button_pressed = pressed

    def _read_execution_poses(self) -> dict[str, Pose3] | None:
        poses = {}
        try:
            for name, frame in EXECUTION_FRAMES.items():
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    frame,
                    Time(),
                ).transform
                poses[name] = Pose3(
                    position=np.array(
                        [
                            transform.translation.x,
                            transform.translation.y,
                            transform.translation.z,
                        ],
                        dtype=np.float64,
                    ),
                    rotation=Rotation.from_quat(
                        [
                            transform.rotation.x,
                            transform.rotation.y,
                            transform.rotation.z,
                            transform.rotation.w,
                        ]
                    ).as_matrix(),
                )
        except Exception as exc:
            self.last_error = f"raw_tf:{type(exc).__name__}:{exc}"
            return None
        return poses

    def _try_calibrate(self, execution_poses: Mapping[str, Pose3]) -> bool:
        if not self.bias_positions:
            self.last_error = "calibration requires a final bias JointState"
            return False
        try:
            if (
                self.last_bias_time is None
                or time.monotonic() - self.last_bias_time > self.bias_state_timeout
            ):
                raise ValueError("final bias JointState is stale")
            if (
                self.last_bias_change_time is None
                or time.monotonic() - self.last_bias_change_time < self.bias_settle_time
            ):
                raise ValueError("final bias pose is still settling")
            robot_initial = self.solver.set_reference(self.bias_positions)
            alignment = compute_alignment_rotation(
                execution_poses["Left"].position,
                execution_poses["Right"].position,
            )
        except Exception as exc:
            self.last_error = f"calibration:{type(exc).__name__}:{exc}"
            return False
        self.calibration = RetargetCalibration(
            alignment_rotation=alignment,
            tracker_initial={
                name: execution_poses[name] for name in EXECUTION_FRAMES
            },
            robot_initial=robot_initial,
        )
        self.calibration_pending = False
        self.calibrations += 1
        self.last_error = ""
        self.get_logger().info(
            "Quest calibration accepted: human pose is zero, robot pose is "
            f"current bias, position_scale={self.position_scale:.3f}"
        )
        return True

    def _tick(self) -> None:
        now = time.monotonic()
        if (
            not self.tracking_ready
            or self.last_tracking_time is None
            or now - self.last_tracking_time > self.tracking_timeout
        ):
            self.dropped += 1
            return
        execution_poses = self._read_execution_poses()
        if execution_poses is None:
            self.dropped += 1
            return
        if self.calibration_pending and not self._try_calibrate(execution_poses):
            self.dropped += 1
            return
        if self.calibration is None:
            self.dropped += 1
            return
        targets = targets_from_calibration(
            execution_poses,
            self.calibration,
            position_scale=self.position_scale,
        )
        started = time.perf_counter()
        try:
            self.solver.set_targets(targets, track_neck=self.enable_neck)
            self.solver.solve(
                iterations=self.iterations,
                solve_dt=self.solve_dt,
            )
            values = self.solver.positions_19()
            self.last_wrist_errors = self.solver.wrist_errors(targets)
            self.last_solved_wrists = self.solver.wrist_poses()
            self.last_elbow_outer_distances = self.solver.elbow_outer_distances()
        except Exception as exc:
            self.dropped += 1
            self.last_error = f"solve:{type(exc).__name__}:{exc}"
            return
        self.solve_times_ms.append((time.perf_counter() - started) * 1000.0)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "quest_retarget"
        msg.name = list(ADAM_COMMAND_JOINTS_19)
        msg.position = values
        msg.velocity = [0.0] * len(values)
        msg.effort = [0.0] * len(values)
        self.publisher.publish(msg)
        self.published += 1
        self.last_targets = dict(targets)
        self.last_error = ""
        self._publish_retarget_status("frame")

    def _log_status(self) -> None:
        average_ms = (
            statistics.fmean(self.solve_times_ms) if self.solve_times_ms else None
        )
        bias_age_ms = (
            None
            if self.last_bias_time is None
            else (time.monotonic() - self.last_bias_time) * 1000.0
        )
        self.get_logger().info(
            "Quest retarget status: "
            f"tracking_ready={self.tracking_ready}, calibrated={self.calibration is not None}, "
            f"calibrations={self.calibrations}, bias_age_ms={bias_age_ms}, "
            f"method={self.retarget_method}, neck_tracking={self.enable_neck}, "
            f"published={self.published}, dropped={self.dropped}, "
            f"solve_ms={average_ms or 'n/a'}, wrist_position_mm={self.last_wrist_errors}, "
            f"elbow_outer_m={self.last_elbow_outer_distances}, "
            f"last_error={self.last_error!r}"
        )
        self._publish_retarget_status("status")

    def _publish_retarget_status(self, event: str) -> None:
        average_ms = (
            statistics.fmean(self.solve_times_ms) if self.solve_times_ms else None
        )
        message = String()
        message.data = json.dumps(
            {
                "protocol_version": 1,
                "event": event,
                "tracking_ready": self.tracking_ready,
                "calibrated": self.calibration is not None,
                "source_sequence": self.last_source_sequence,
                "retarget_sequence": self.published,
                "published": self.published,
                "dropped": self.dropped,
                "solve_ms": average_ms,
                "target_poses": {
                    side: pose3_status(pose)
                    for side, pose in self.last_targets.items()
                    if side in {"Left", "Right"}
                },
                "solved_wrist_poses": {
                    side: pose3_status(pose)
                    for side, pose in self.last_solved_wrists.items()
                },
                "wrist_position_residual_mm": {
                    side: value if math.isfinite(value) else None
                    for side, value in self.last_wrist_errors.items()
                },
                "last_error": self.last_error,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        self.status_publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = QuestRetargetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
