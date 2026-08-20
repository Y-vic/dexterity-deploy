#!/usr/bin/env python3
"""GMR-backed Noitom retarget node with vendor-compatible ROS IO."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import statistics
import sys
import time
import types
from collections import deque
from typing import Iterable


DEFAULT_GMR_PYTHON = "/home/pnd-humanoid/Deploy/.venv-gmr/bin/python"


def _maybe_reexec_into_gmr_python() -> None:
    requested = os.environ.get("NOITOM_GMR_PYTHON") or DEFAULT_GMR_PYTHON
    python_path = pathlib.Path(requested)
    if not python_path.is_file():
        return
    venv_prefix = python_path.parent.parent.resolve()
    if pathlib.Path(sys.prefix).resolve() == venv_prefix:
        return

    env = os.environ.copy()
    env["NOITOM_GMR_REEXEC"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    os.execve(str(python_path), [str(python_path), *sys.argv], env)


_maybe_reexec_into_gmr_python()

import mujoco as mj  # noqa: E402
import mink  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import tf2_ros  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.time import Time  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from mink.limits.limit import Constraint, Limit  # noqa: E402


DEFAULT_GMR_REPO = "/home/pnd-humanoid/Deploy/GMR-master"

NOITOM_BODY31_BONES = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "Spine2",
    "Head",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
)
UPPER_BODY_BONES = (
    "Hips",
    "Spine2",
    "Head",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
)
LOWER_BODY_BONES = {
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
}

ADAM_JOINT_NAME_MAP = (
    ("dof_pos/waistRoll", "waistRoll"),
    ("dof_pos/waistPitch", "waistPitch"),
    ("dof_pos/waistYaw", "waistYaw"),
    ("dof_pos/neckYaw", "neckYaw"),
    ("dof_pos/neckPitch", "neckPitch"),
    ("dof_pos/shoulderPitch_Left", "shoulderPitch_Left"),
    ("dof_pos/shoulderRoll_Left", "shoulderRoll_Left"),
    ("dof_pos/shoulderYaw_Left", "shoulderYaw_Left"),
    ("dof_pos/elbow_Left", "elbow_Left"),
    ("dof_pos/wristYaw_Left", "wristYaw_Left"),
    ("dof_pos/wristPitch_Left", "wristPitch_Left"),
    ("dof_pos/wristRoll_Left", "wristRoll_Left"),
    ("dof_pos/shoulderPitch_Right", "shoulderPitch_Right"),
    ("dof_pos/shoulderRoll_Right", "shoulderRoll_Right"),
    ("dof_pos/shoulderYaw_Right", "shoulderYaw_Right"),
    ("dof_pos/elbow_Right", "elbow_Right"),
    ("dof_pos/wristYaw_Right", "wristYaw_Right"),
    ("dof_pos/wristPitch_Right", "wristPitch_Right"),
    ("dof_pos/wristRoll_Right", "wristRoll_Right"),
)
ADAM_COMMAND_JOINTS_19 = [ros_name for ros_name, _ in ADAM_JOINT_NAME_MAP]


def _load_module(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_gmr_class(gmr_repo_path: str):
    repo = pathlib.Path(gmr_repo_path).expanduser().resolve()
    package_path = repo / "general_motion_retargeting"
    if not (package_path / "motion_retarget.py").exists():
        raise FileNotFoundError(
            f"GMR motion_retarget.py not found under {package_path}"
        )

    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = [str(package_path)]
    sys.modules["general_motion_retargeting"] = package
    _load_module(
        "general_motion_retargeting.params",
        package_path / "params.py",
    )
    motion_module = _load_module(
        "general_motion_retargeting.motion_retarget",
        package_path / "motion_retarget.py",
    )
    return motion_module.GeneralMotionRetargeting


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def xyzw_to_wxyz(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array([w, x, y, z], dtype=np.float64)


def percentile(values: deque[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(len(sorted_values) * pct)))
    return sorted_values[index]


class RootLockLimit(Limit):
    """Lock the MuJoCo floating-base tangent coordinates.

    The ROS retarget output only contains Adam joint angles, not the floating
    root pose, so allowing IK to solve through root motion produces joint angles
    that are not directly executable by the downstream command path.
    """

    def __init__(self, nv: int) -> None:
        self.nv = nv
        self.indices = tuple(range(6))

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        del configuration, dt
        rows = len(self.indices)
        g = np.zeros((2 * rows, self.nv), dtype=np.float64)
        h = np.zeros(2 * rows, dtype=np.float64)
        for row, index in enumerate(self.indices):
            g[row, index] = 1.0
            g[row + rows, index] = -1.0
        return Constraint(G=g, h=h)


class NoitomGMRRetarget(Node):
    def __init__(self) -> None:
        super().__init__("_noitom_retarget")
        self.declare_parameter("base_frame", "world_zup")
        self.declare_parameter("control_loop_rate", 100.0)
        self.declare_parameter("gmr_repo_path", os.environ.get("PND_GMR_REPO", DEFAULT_GMR_REPO))
        self.declare_parameter("robot", "pnd_adam_pro_body31")
        self.declare_parameter("solver", "daqp")
        self.declare_parameter("damping", 0.3)
        self.declare_parameter("use_velocity_limit", False)
        self.declare_parameter("upper_body_only", True)
        self.declare_parameter("lock_root", False)
        self.declare_parameter("reset_each_frame", False)
        self.declare_parameter("posture_cost", 0.05)
        self.declare_parameter("apply_pnd_coordinate_transform", False)
        self.declare_parameter("offset_to_ground", False)
        self.declare_parameter("status_period", 2.0)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.control_loop_rate = float(self.get_parameter("control_loop_rate").value)
        self.gmr_repo_path = str(self.get_parameter("gmr_repo_path").value)
        self.robot = str(self.get_parameter("robot").value)
        self.solver = str(self.get_parameter("solver").value)
        self.damping = float(self.get_parameter("damping").value)
        self.use_velocity_limit = bool(self.get_parameter("use_velocity_limit").value)
        self.upper_body_only = bool(self.get_parameter("upper_body_only").value)
        self.lock_root = bool(self.get_parameter("lock_root").value)
        self.reset_each_frame = bool(self.get_parameter("reset_each_frame").value)
        self.posture_cost = float(self.get_parameter("posture_cost").value)
        self.apply_pnd_coordinate_transform = bool(
            self.get_parameter("apply_pnd_coordinate_transform").value
        )
        self.offset_to_ground = bool(self.get_parameter("offset_to_ground").value)
        self.status_period = float(self.get_parameter("status_period").value)

        if self.control_loop_rate <= 0.0:
            raise ValueError("control_loop_rate must be positive")
        if self.status_period <= 0.0:
            raise ValueError("status_period must be positive")

        gmr_class = load_gmr_class(self.gmr_repo_path)
        self.gmr = gmr_class(
            src_human="noitom",
            tgt_robot=self.robot,
            solver=self.solver,
            damping=self.damping,
            verbose=False,
            use_velocity_limit=self.use_velocity_limit,
        )
        if self.upper_body_only:
            self._remove_lower_body_tasks()
        self._patch_gmr_retarget()
        if self.lock_root:
            self.gmr.ik_limits.append(RootLockLimit(self.gmr.model.nv))
        self.reference_qpos = self.gmr.configuration.data.qpos.copy()
        if self.posture_cost > 0.0:
            self._add_posture_regularization()
        self.qpos_index_by_joint = self._make_qpos_index_by_joint()
        self._validate_output_joints()

        self.bone_names = UPPER_BODY_BONES if self.upper_body_only else NOITOM_BODY31_BONES
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)

        self.rotation_matrix = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            dtype=np.float64,
        )
        self.rotation_quat = xyzw_to_wxyz(
            R.from_matrix(self.rotation_matrix).as_quat()
        )

        self.ticks = 0
        self.published = 0
        self.dropped_tf = 0
        self.dropped_retarget = 0
        self.last_error = ""
        self.retarget_ms = deque(maxlen=512)
        self.publish_intervals_ms = deque(maxlen=512)
        self.last_publish_time: float | None = None
        self.last_log_time = time.monotonic()
        self.last_log_published = 0

        self.create_timer(1.0 / self.control_loop_rate, self._tick)
        self.create_timer(self.status_period, self._log_status)

        self.get_logger().info(
            "Noitom GMR retarget: "
            f"base_frame={self.base_frame}, control_loop_rate={self.control_loop_rate}, "
            f"repo={self.gmr_repo_path}, robot={self.robot}, solver={self.solver}, "
            f"damping={self.damping}, upper_body_only={self.upper_body_only}, "
            f"lock_root={self.lock_root}, "
            f"reset_each_frame={self.reset_each_frame}, "
            f"posture_cost={self.posture_cost}, "
            f"apply_pnd_coordinate_transform={self.apply_pnd_coordinate_transform}"
        )

    def _patch_gmr_retarget(self) -> None:
        def retarget(gmr, human_data, offset_to_ground=False):
            gmr.update_targets(human_data, offset_to_ground)

            if gmr.use_ik_match_table1:
                curr_error = gmr.error1()
                dt = gmr.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    gmr.configuration,
                    gmr.tasks1,
                    dt,
                    gmr.solver,
                    gmr.damping,
                    limits=gmr.ik_limits,
                )
                gmr.configuration.integrate_inplace(vel1, dt)
                next_error = gmr.error1()
                num_iter = 0
                while curr_error - next_error > 0.001 and num_iter < gmr.max_iter:
                    curr_error = next_error
                    dt = gmr.configuration.model.opt.timestep
                    vel1 = mink.solve_ik(
                        gmr.configuration,
                        gmr.tasks1,
                        dt,
                        gmr.solver,
                        gmr.damping,
                        limits=gmr.ik_limits,
                    )
                    gmr.configuration.integrate_inplace(vel1, dt)
                    next_error = gmr.error1()
                    num_iter += 1

            if gmr.use_ik_match_table2:
                curr_error = gmr.error2()
                dt = gmr.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    gmr.configuration,
                    gmr.tasks2,
                    dt,
                    gmr.solver,
                    gmr.damping,
                    limits=gmr.ik_limits,
                )
                gmr.configuration.integrate_inplace(vel2, dt)
                next_error = gmr.error2()
                num_iter = 0
                while curr_error - next_error > 0.001 and num_iter < gmr.max_iter:
                    curr_error = next_error
                    dt = gmr.configuration.model.opt.timestep
                    vel2 = mink.solve_ik(
                        gmr.configuration,
                        gmr.tasks2,
                        dt,
                        gmr.solver,
                        gmr.damping,
                        limits=gmr.ik_limits,
                    )
                    gmr.configuration.integrate_inplace(vel2, dt)
                    next_error = gmr.error2()
                    num_iter += 1

            return gmr.configuration.data.qpos.copy()

        self.gmr.retarget = types.MethodType(retarget, self.gmr)

    def _remove_lower_body_tasks(self) -> None:
        for body_name in LOWER_BODY_BONES:
            task = self.gmr.human_body_to_task1.pop(body_name, None)
            if task is not None:
                self.gmr.task_errors1.pop(task, None)
                self.gmr.tasks1 = [
                    existing for existing in self.gmr.tasks1 if existing is not task
                ]
            task = self.gmr.human_body_to_task2.pop(body_name, None)
            if task is not None:
                self.gmr.task_errors2.pop(task, None)
                self.gmr.tasks2 = [
                    existing for existing in self.gmr.tasks2 if existing is not task
                ]

    def _make_qpos_index_by_joint(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for joint_id in range(self.gmr.model.njnt):
            name = mj.mj_id2name(self.gmr.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                continue
            mapping[name] = int(self.gmr.model.jnt_qposadr[joint_id])
        return mapping

    def _validate_output_joints(self) -> None:
        missing = [
            robot_joint
            for _, robot_joint in ADAM_JOINT_NAME_MAP
            if robot_joint not in self.qpos_index_by_joint
        ]
        if missing:
            raise ValueError(f"GMR robot model is missing joints: {missing}")

    def _add_posture_regularization(self) -> None:
        posture_task = mink.PostureTask(self.gmr.model, cost=self.posture_cost)
        posture_task.set_target(self.reference_qpos.copy())
        if self.gmr.use_ik_match_table1:
            self.gmr.tasks1.append(posture_task)
        if self.gmr.use_ik_match_table2:
            self.gmr.tasks2.append(posture_task)

    def _read_human_frame(self) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
        frame = {}
        try:
            for bone in self.bone_names:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    bone,
                    Time(),
                )
                pos = np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ],
                    dtype=np.float64,
                )
                quat = normalize_quat_wxyz(
                    np.array(
                        [
                            transform.transform.rotation.w,
                            transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                        ],
                        dtype=np.float64,
                    )
                )
                if self.apply_pnd_coordinate_transform:
                    pos = self.rotation_matrix @ pos
                    quat = normalize_quat_wxyz(quat_mul_wxyz(self.rotation_quat, quat))
                frame[bone] = (pos, quat)
        except Exception as exc:  # noqa: BLE001 - TF errors are expected at startup.
            self.last_error = f"tf:{type(exc).__name__}:{exc}"
            return None
        return frame

    def _tick(self) -> None:
        self.ticks += 1
        human_frame = self._read_human_frame()
        if human_frame is None:
            self.dropped_tf += 1
            return

        start = time.perf_counter()
        try:
            if self.reset_each_frame:
                self.gmr.configuration.update(self.reference_qpos.copy())
            qpos = self.gmr.retarget(
                human_frame,
                offset_to_ground=self.offset_to_ground,
            )
            positions = self._extract_positions(qpos)
        except Exception as exc:  # noqa: BLE001 - keep node alive for transient faults.
            self.dropped_retarget += 1
            self.last_error = f"retarget:{type(exc).__name__}:{exc}"
            return

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.retarget_ms.append(elapsed_ms)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "noitom_gmr_retarget"
        msg.name = list(ADAM_COMMAND_JOINTS_19)
        msg.position = positions
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        self.publisher.publish(msg)

        now = time.monotonic()
        if self.last_publish_time is not None:
            self.publish_intervals_ms.append((now - self.last_publish_time) * 1000.0)
        self.last_publish_time = now
        self.published += 1
        self.last_error = ""

    def _extract_positions(self, qpos: np.ndarray) -> list[float]:
        positions = [
            float(qpos[self.qpos_index_by_joint[robot_joint]])
            for _, robot_joint in ADAM_JOINT_NAME_MAP
        ]
        if not all(np.isfinite(positions)):
            raise ValueError("non-finite GMR qpos output")
        return positions

    def _log_status(self) -> None:
        now = time.monotonic()
        elapsed = max(1e-6, now - self.last_log_time)
        output_rate = (self.published - self.last_log_published) / elapsed
        self.last_log_time = now
        self.last_log_published = self.published

        avg_ms = statistics.fmean(self.retarget_ms) if self.retarget_ms else None
        p95_ms = percentile(self.retarget_ms, 0.95)
        interval_avg_ms = (
            statistics.fmean(self.publish_intervals_ms)
            if self.publish_intervals_ms
            else None
        )
        self.get_logger().info(
            "Noitom GMR status: "
            f"ticks={self.ticks}, published={self.published}, "
            f"output_rate_hz={output_rate:.1f}, dropped_tf={self.dropped_tf}, "
            f"dropped_retarget={self.dropped_retarget}, "
            f"retarget_avg_ms={avg_ms if avg_ms is not None else 'n/a'}, "
            f"retarget_p95_ms={p95_ms if p95_ms is not None else 'n/a'}, "
            f"publish_interval_avg_ms={interval_avg_ms if interval_avg_ms is not None else 'n/a'}, "
            f"last_error={self.last_error!r}"
        )


def main() -> None:
    rclpy.init()
    node = NoitomGMRRetarget()
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
