"""Stable bimanual inverse kinematics for the Adam Pro model."""

from __future__ import annotations

import math
import pathlib
from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from pink.tasks.task import Task
from scipy.spatial.transform import Rotation

from quest_node.command_gate import (
    ADAM_COMMAND_JOINTS_19,
    ARM_JOINTS,
    LEFT_ARM_JOINTS,
    NECK_JOINTS,
    RIGHT_ARM_JOINTS,
    WAIST_JOINTS,
)


ARM_JOINT_NAMES = tuple(name.removeprefix("dof_pos/") for name in ARM_JOINTS)
EE_PARENT_FRAMES = {"Left": "wristRollLeft", "Right": "wristRollRight"}
EE_FRAMES = {"Left": "L_ee", "Right": "R_ee"}
SHOULDER_FRAMES = {"Left": "shoulderPitchLeft", "Right": "shoulderPitchRight"}
ELBOW_FRAMES = {"Left": "elbowLeft", "Right": "elbowRight"}
SHOULDER_PRIOR_FRAMES = {
    "Left": "shoulderYawLeft",
    "Right": "shoulderYawRight",
}
RETARGET_METHODS = (
    "local_qp",
    "shoulder_prior",
    "nonlinear_ik",
    "elbow_pole",
)
NATURAL_ARM_POSTURE_WEIGHTS = np.asarray(
    [1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25] * 2,
    dtype=np.float64,
)


@dataclass(frozen=True)
class Pose3:
    position: np.ndarray
    rotation: np.ndarray


class ElbowOuterTask(Task):
    """Penalize an elbow only while it is inside its shoulder."""

    def __init__(
        self,
        frame: str,
        shoulder_position: np.ndarray,
        outward_direction: np.ndarray,
        *,
        margin: float,
        cost: float,
        gain: float = 0.2,
    ) -> None:
        super().__init__(cost=cost, gain=gain, lm_damping=0.0)
        self.frame = frame
        self.shoulder_position = np.asarray(
            shoulder_position, dtype=np.float64
        ).copy()
        direction = np.asarray(outward_direction, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1.0e-9:
            raise ValueError("elbow outward direction must be non-zero")
        self.outward_direction = direction / direction_norm
        self.margin = float(margin)

    def _signed_distance(self, configuration: pink.Configuration) -> float:
        elbow_position = configuration.get_transform_frame_to_world(
            self.frame
        ).translation
        return float(
            self.outward_direction @ (elbow_position - self.shoulder_position)
        )

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        violation = self.margin - self._signed_distance(configuration)
        return np.asarray([max(0.0, violation)], dtype=np.float64)

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        if self._signed_distance(configuration) >= self.margin:
            return np.zeros((1, configuration.model.nv), dtype=np.float64)
        elbow_transform = configuration.get_transform_frame_to_world(self.frame)
        elbow_jacobian_local = configuration.get_frame_jacobian(self.frame)[:3]
        elbow_jacobian_world = elbow_transform.rotation @ elbow_jacobian_local
        return -self.outward_direction[None, :] @ elbow_jacobian_world


def two_bone_elbow_target(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    pole: np.ndarray,
    upper_arm_length: float,
    forearm_length: float,
) -> np.ndarray:
    shoulder_to_wrist = np.asarray(wrist, dtype=np.float64) - np.asarray(
        shoulder, dtype=np.float64
    )
    raw_distance = float(np.linalg.norm(shoulder_to_wrist))
    if raw_distance < 1.0e-6:
        raise ValueError("wrist target is too close to the shoulder")
    direction = shoulder_to_wrist / raw_distance
    minimum_distance = abs(upper_arm_length - forearm_length) + 1.0e-6
    maximum_distance = upper_arm_length + forearm_length - 1.0e-6
    distance = float(np.clip(raw_distance, minimum_distance, maximum_distance))
    along = (
        upper_arm_length**2
        - forearm_length**2
        + distance**2
    ) / (2.0 * distance)
    height = math.sqrt(max(0.0, upper_arm_length**2 - along**2))
    pole_direction = np.asarray(pole, dtype=np.float64)
    pole_direction -= direction * float(np.dot(pole_direction, direction))
    pole_norm = float(np.linalg.norm(pole_direction))
    if pole_norm < 1.0e-6:
        raise ValueError("elbow pole is parallel to the shoulder-wrist line")
    pole_direction /= pole_norm
    return np.asarray(shoulder, dtype=np.float64) + (
        along * direction + height * pole_direction
    )


class AdamBimanualIkSolver:
    """Solve Adam Pro wrist targets with damped nonlinear IK."""

    def __init__(
        self,
        model_path: str,
        *,
        retarget_method: str = "nonlinear_ik",
        solver: str = "daqp",
        damping: float = 0.0,
        lm_damping: float = 1.0,
        line_search_steps: int = 10,
        wrist_position_cost: float = 50.0,
        wrist_orientation_cost: float = 1.0,
        elbow_position_cost: float = 10.0,
        elbow_outer_cost: float = 50.0,
        elbow_outer_margin: float = 0.02,
        smoothness_cost: float = 0.2,
        posture_cost: float = 0.05,
        shoulder_prior_wrist_position_cost: float = 20.0,
        shoulder_prior_wrist_orientation_cost: float = 18.0,
        shoulder_prior_orientation_cost: float = 2.0,
        nonlinear_translation_cost: float = 50.0,
        nonlinear_rotation_cost: float = 1.0,
        nonlinear_posture_cost: float = 0.02,
        nonlinear_smoothness_cost: float = 0.1,
        nonlinear_filter_enabled: bool = True,
    ) -> None:
        path = pathlib.Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Adam Pro URDF not found: {path}")
        self.full_model = pin.buildModelFromUrdf(str(path))
        self._add_end_effector_frames()
        missing = [
            name for name in ARM_JOINT_NAMES if not self.full_model.existJointName(name)
        ]
        if missing:
            raise ValueError(f"Adam Pro model is missing arm joints: {missing}")

        self.retarget_method = str(retarget_method).lower()
        if self.retarget_method not in RETARGET_METHODS:
            raise ValueError(
                f"unknown retarget method {retarget_method!r}; "
                f"expected one of {RETARGET_METHODS}"
            )
        self.solver = solver
        self.damping = damping
        self.lm_damping = lm_damping
        self.line_search_steps = line_search_steps
        if self.damping < 0.0 or self.lm_damping < 0.0:
            raise ValueError("IK damping values must be non-negative")
        if self.line_search_steps <= 0:
            raise ValueError("IK line_search_steps must be positive")
        self.wrist_position_cost = wrist_position_cost
        self.wrist_orientation_cost = wrist_orientation_cost
        self.elbow_position_cost = float(elbow_position_cost)
        self.elbow_outer_cost = float(elbow_outer_cost)
        self.elbow_outer_margin = float(elbow_outer_margin)
        if self.elbow_outer_cost < 0.0 or self.elbow_outer_margin < 0.0:
            raise ValueError("elbow outer cost and margin must be non-negative")
        self.smoothness_cost = smoothness_cost
        self.posture_cost = posture_cost
        self.shoulder_prior_wrist_position_cost = float(
            shoulder_prior_wrist_position_cost
        )
        self.shoulder_prior_wrist_orientation_cost = float(
            shoulder_prior_wrist_orientation_cost
        )
        self.shoulder_prior_orientation_cost = float(
            shoulder_prior_orientation_cost
        )
        self.nonlinear_translation_cost = float(nonlinear_translation_cost)
        self.nonlinear_rotation_cost = float(nonlinear_rotation_cost)
        self.nonlinear_posture_cost = float(nonlinear_posture_cost)
        self.nonlinear_smoothness_cost = float(nonlinear_smoothness_cost)
        self.nonlinear_filter_enabled = bool(nonlinear_filter_enabled)
        method_costs = (
            self.shoulder_prior_wrist_position_cost,
            self.shoulder_prior_wrist_orientation_cost,
            self.shoulder_prior_orientation_cost,
            self.nonlinear_translation_cost,
            self.nonlinear_rotation_cost,
            self.nonlinear_posture_cost,
            self.nonlinear_smoothness_cost,
        )
        if any(cost < 0.0 for cost in method_costs):
            raise ValueError("retarget method costs must be non-negative")
        self.model: pin.Model | None = None
        self.data: pin.Data | None = None
        self.arm_indices: np.ndarray | None = None
        self.arm_side_indices: dict[str, np.ndarray] = {}
        self.default_qpos: np.ndarray | None = None
        self.last_output_qpos: np.ndarray | None = None
        self.waist_positions: dict[str, float] = {}
        self.neck_positions: dict[str, float] = {}
        self.torso_rotation: np.ndarray | None = None
        self.shoulder_positions: dict[str, np.ndarray] = {}
        self.arm_lengths: dict[str, tuple[float, float]] = {}
        self.elbow_poles: dict[str, np.ndarray] = {}
        self.elbow_outward_directions: dict[str, np.ndarray] = {}
        self.shoulder_prior_orientations: dict[str, np.ndarray] = {}
        self.nonlinear_problem = None
        self.nonlinear_history: deque[np.ndarray] = deque(maxlen=4)
        self.targets = None

    @staticmethod
    def _objective(tasks, configuration: pink.Configuration) -> float:
        objective = 0.0
        for task in tasks:
            error = np.asarray(task.compute_error(configuration), dtype=np.float64)
            weighted_error = (
                error
                if task.cost is None
                else np.asarray(task.cost, dtype=np.float64) * error
            )
            objective += float(weighted_error @ weighted_error)
        return objective

    def _add_end_effector_frames(self) -> None:
        for side in ("Left", "Right"):
            ee_name = EE_FRAMES[side]
            if self.full_model.existFrame(ee_name):
                continue
            parent_frame_id = self.full_model.getFrameId(EE_PARENT_FRAMES[side])
            parent_frame = self.full_model.frames[parent_frame_id]
            angle = -math.pi / 2.0 if side == "Left" else math.pi / 2.0
            offset = pin.SE3(
                Rotation.from_euler("z", angle).as_matrix(),
                np.zeros(3, dtype=np.float64),
            )
            self.full_model.addFrame(
                pin.Frame(
                    ee_name,
                    parent_frame.parentJoint,
                    parent_frame_id,
                    parent_frame.placement * offset,
                    pin.FrameType.OP_FRAME,
                ),
                False,
            )

    @staticmethod
    def _frame_pose(data: pin.Data, model: pin.Model, frame_name: str) -> Pose3:
        transform = data.oMf[model.getFrameId(frame_name)]
        return Pose3(
            position=transform.translation.copy(),
            rotation=transform.rotation.copy(),
        )

    def _reduce_at_reference(self, full_reference: np.ndarray) -> None:
        unlocked = set(ARM_JOINT_NAMES)
        locked_joint_ids = [
            self.full_model.getJointId(name)
            for name in self.full_model.names[1:]
            if name not in unlocked
        ]
        self.model = pin.buildReducedModel(
            self.full_model,
            locked_joint_ids,
            full_reference,
        )
        self.data = self.model.createData()
        if self.model.nq != 14 or self.model.nv != 14:
            raise ValueError(
                f"Adam reduced bimanual model must be 14-DOF, got "
                f"nq={self.model.nq}, nv={self.model.nv}"
            )
        self.arm_indices = np.asarray(
            [
                self.model.joints[self.model.getJointId(name)].idx_q
                for name in ARM_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self.arm_side_indices = {
            side: np.asarray(
                [
                    self.model.joints[
                        self.model.getJointId(name.removeprefix("dof_pos/"))
                    ].idx_q
                    for name in joint_names
                ],
                dtype=np.int64,
            )
            for side, joint_names in (
                ("Left", LEFT_ARM_JOINTS),
                ("Right", RIGHT_ARM_JOINTS),
            )
        }

    def _assemble_arm_qpos(self, positions: Mapping[str, float]) -> np.ndarray:
        if self.model is None or self.arm_indices is None:
            raise RuntimeError("solver has not been calibrated to a bias pose")
        qpos = pin.neutral(self.model)
        qpos[self.arm_indices] = [float(positions[name]) for name in ARM_JOINTS]
        return qpos

    def set_reference(self, positions: Mapping[str, float]):
        missing = [name for name in ADAM_COMMAND_JOINTS_19 if name not in positions]
        if missing:
            raise ValueError(f"bias pose is missing joints: {missing}")
        reference = pin.neutral(self.full_model)
        for ros_name in ADAM_COMMAND_JOINTS_19:
            joint_name = ros_name.removeprefix("dof_pos/")
            joint = self.full_model.joints[self.full_model.getJointId(joint_name)]
            reference[joint.idx_q] = float(positions[ros_name])
        full_data = self.full_model.createData()
        pin.forwardKinematics(self.full_model, full_data, reference)
        pin.updateFramePlacements(self.full_model, full_data)

        self._reduce_at_reference(reference)
        self.default_qpos = self._assemble_arm_qpos(positions)
        self.last_output_qpos = self.default_qpos.copy()
        self.waist_positions = {name: float(positions[name]) for name in WAIST_JOINTS}
        self.neck_positions = {name: float(positions[name]) for name in NECK_JOINTS}
        self.torso_rotation = self._frame_pose(
            full_data, self.full_model, "torso"
        ).rotation
        torso_left = self.torso_rotation[:, 1]
        for side in ("Left", "Right"):
            shoulder = self._frame_pose(
                full_data, self.full_model, SHOULDER_FRAMES[side]
            ).position
            elbow = self._frame_pose(
                full_data, self.full_model, ELBOW_FRAMES[side]
            ).position
            wrist = self._frame_pose(
                full_data, self.full_model, EE_FRAMES[side]
            ).position
            shoulder_to_wrist = wrist - shoulder
            direction = shoulder_to_wrist / np.linalg.norm(shoulder_to_wrist)
            pole = elbow - shoulder
            pole -= direction * float(np.dot(pole, direction))
            pole /= np.linalg.norm(pole)
            self.shoulder_positions[side] = shoulder
            self.arm_lengths[side] = (
                float(np.linalg.norm(elbow - shoulder)),
                float(np.linalg.norm(wrist - elbow)),
            )
            self.elbow_poles[side] = pole
            self.elbow_outward_directions[side] = (
                torso_left.copy() if side == "Left" else -torso_left.copy()
            )
            self.shoulder_prior_orientations[side] = self._frame_pose(
                full_data,
                self.full_model,
                SHOULDER_PRIOR_FRAMES[side],
            ).rotation
        self.nonlinear_problem = None
        self.nonlinear_history.clear()
        reference_poses = {
            "Head": self._frame_pose(full_data, self.full_model, "neckPitch_link"),
            "Left": self._frame_pose(full_data, self.full_model, EE_FRAMES["Left"]),
            "Right": self._frame_pose(full_data, self.full_model, EE_FRAMES["Right"]),
        }
        if self.retarget_method == "nonlinear_ik":
            self._initialize_nonlinear_problem()
            self.targets = reference_poses
            self._solve_nonlinear()
            self.last_output_qpos = self.default_qpos.copy()
            self.nonlinear_history.clear()
        self.targets = None
        return reference_poses

    def set_targets(
        self,
        targets: Mapping[str, Pose3],
        *,
        track_neck: bool = True,
    ) -> None:
        if self.torso_rotation is None:
            raise RuntimeError("solver has not been calibrated to a bias pose")
        self.targets = targets
        if not track_neck:
            return
        local_head = self.torso_rotation.T @ targets["Head"].rotation
        yaw = math.atan2(float(local_head[1, 0]), float(local_head[0, 0]))
        pitch = math.atan2(
            float(-local_head[2, 0]),
            math.hypot(float(local_head[0, 0]), float(local_head[1, 0])),
        )
        for ros_name, value in (
            ("dof_pos/neckYaw", yaw),
            ("dof_pos/neckPitch", pitch),
        ):
            joint_name = ros_name.removeprefix("dof_pos/")
            joint = self.full_model.joints[self.full_model.getJointId(joint_name)]
            lower = float(self.full_model.lowerPositionLimit[joint.idx_q])
            upper = float(self.full_model.upperPositionLimit[joint.idx_q])
            self.neck_positions[ros_name] = float(np.clip(value, lower, upper))

    def _pink_tasks(self) -> list[Task]:
        if (
            self.model is None
            or self.default_qpos is None
            or self.last_output_qpos is None
            or self.targets is None
        ):
            raise RuntimeError("solver has no reference or targets")

        if self.retarget_method == "shoulder_prior":
            wrist_position_cost = self.shoulder_prior_wrist_position_cost
            wrist_orientation_cost = self.shoulder_prior_wrist_orientation_cost
            wrist_gain = 1.0
        else:
            wrist_position_cost = self.wrist_position_cost
            wrist_orientation_cost = self.wrist_orientation_cost
            wrist_gain = 0.2

        left_ee_task = FrameTask(
            frame=EE_FRAMES["Left"],
            position_cost=wrist_position_cost,
            orientation_cost=wrist_orientation_cost,
            lm_damping=self.lm_damping,
            gain=wrist_gain,
        )
        right_ee_task = FrameTask(
            frame=EE_FRAMES["Right"],
            position_cost=wrist_position_cost,
            orientation_cost=wrist_orientation_cost,
            lm_damping=self.lm_damping,
            gain=wrist_gain,
        )
        left_ee_task.set_target(
            pin.SE3(self.targets["Left"].rotation, self.targets["Left"].position)
        )
        right_ee_task.set_target(
            pin.SE3(self.targets["Right"].rotation, self.targets["Right"].position)
        )
        tasks = [left_ee_task, right_ee_task]
        if self.retarget_method == "shoulder_prior":
            for side in ("Left", "Right"):
                shoulder_task = FrameTask(
                    frame=SHOULDER_PRIOR_FRAMES[side],
                    position_cost=0.0,
                    orientation_cost=self.shoulder_prior_orientation_cost,
                    lm_damping=self.lm_damping,
                    gain=1.0,
                )
                shoulder_task.set_target(
                    pin.SE3(
                        self.shoulder_prior_orientations[side],
                        np.zeros(3, dtype=np.float64),
                    )
                )
                tasks.append(shoulder_task)
            return tasks

        if self.retarget_method == "elbow_pole" and self.elbow_position_cost > 0.0:
            for side in ("Left", "Right"):
                elbow_task = FrameTask(
                    frame=ELBOW_FRAMES[side],
                    position_cost=self.elbow_position_cost,
                    orientation_cost=0.0,
                    lm_damping=self.lm_damping,
                    gain=0.2,
                )
                elbow_target = two_bone_elbow_target(
                    self.shoulder_positions[side],
                    self.targets[side].position,
                    self.elbow_poles[side],
                    *self.arm_lengths[side],
                )
                elbow_task.set_target(pin.SE3(np.eye(3), elbow_target))
                tasks.append(elbow_task)
        if self.retarget_method == "elbow_pole" and self.elbow_outer_cost > 0.0:
            for side in ("Left", "Right"):
                tasks.append(
                    ElbowOuterTask(
                        frame=ELBOW_FRAMES[side],
                        shoulder_position=self.shoulder_positions[side],
                        outward_direction=self.elbow_outward_directions[side],
                        margin=self.elbow_outer_margin,
                        cost=self.elbow_outer_cost,
                    )
                )
        smoothness_task = PostureTask(
            cost=self.smoothness_cost * NATURAL_ARM_POSTURE_WEIGHTS,
            lm_damping=0.0,
            gain=0.2,
        )
        posture_task = PostureTask(
            cost=self.posture_cost * NATURAL_ARM_POSTURE_WEIGHTS,
            lm_damping=0.0,
            gain=0.2,
        )
        smoothness_task.set_target(self.last_output_qpos)
        posture_task.set_target(self.default_qpos)
        tasks.extend((smoothness_task, posture_task))
        return tasks

    def _solve_pink(self, *, iterations: int, solve_dt: float) -> None:
        if self.model is None or self.data is None or self.last_output_qpos is None:
            raise RuntimeError("solver has not been calibrated")
        tasks = self._pink_tasks()
        configuration = pink.Configuration(
            model=self.model,
            data=self.data,
            q=self.last_output_qpos,
            copy_data=True,
            forward_kinematics=True,
        )
        for _ in range(iterations):
            try:
                velocity = pink.solve_ik(
                    configuration=configuration,
                    tasks=tasks,
                    dt=solve_dt,
                    solver=self.solver,
                    damping=self.damping,
                )
                if not np.all(np.isfinite(velocity)):
                    break
            except Exception:
                break
            delta = velocity * solve_dt
            if float(np.linalg.norm(delta)) < 1.0e-10:
                break
            objective = self._objective(tasks, configuration)
            step_scale = 1.0
            accepted = False
            for _ in range(self.line_search_steps):
                candidate = pin.integrate(
                    self.model,
                    configuration.q,
                    step_scale * delta,
                )
                candidate = np.clip(
                    candidate,
                    self.model.lowerPositionLimit,
                    self.model.upperPositionLimit,
                )
                trial = pink.Configuration(
                    model=self.model,
                    data=self.data,
                    q=candidate,
                    copy_data=True,
                    forward_kinematics=True,
                )
                if self._objective(tasks, trial) <= objective:
                    configuration.update(candidate)
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                break
        self.last_output_qpos = configuration.q.copy()

    def _initialize_nonlinear_problem(self) -> None:
        if self.model is None or self.default_qpos is None:
            raise RuntimeError("solver has not been calibrated")
        try:
            import casadi
        except ImportError as exc:
            raise RuntimeError(
                "retarget_method=nonlinear_ik requires the optional casadi package"
            ) from exc

        symbolic_q = casadi.SX.sym("q", self.model.nq)
        symbolic_left = casadi.SX.sym("left_target", 4, 4)
        symbolic_right = casadi.SX.sym("right_target", 4, 4)
        forward_kinematics = self._build_nonlinear_forward_kinematics(
            casadi,
            symbolic_q,
        )
        (
            left_current,
            right_current,
            left_elbow,
            right_elbow,
        ) = forward_kinematics(symbolic_q)

        translation_error = casadi.Function(
            "adam_nonlinear_translation_error",
            [symbolic_q, symbolic_left, symbolic_right],
            [
                casadi.vertcat(
                    left_current[:3, 3] - symbolic_left[:3, 3],
                    right_current[:3, 3] - symbolic_right[:3, 3],
                )
            ],
        )
        rotation_error = casadi.Function(
            "adam_nonlinear_rotation_error",
            [symbolic_q, symbolic_left, symbolic_right],
            [
                casadi.vertcat(
                    self._casadi_rotation_residual(
                        casadi,
                        left_current[:3, :3] @ symbolic_left[:3, :3].T,
                    ),
                    self._casadi_rotation_residual(
                        casadi,
                        right_current[:3, :3] @ symbolic_right[:3, :3].T,
                    ),
                )
            ],
        )
        elbow_violations = []
        for side, elbow_position in (
            ("Left", left_elbow),
            ("Right", right_elbow),
        ):
            outward = casadi.DM(self.elbow_outward_directions[side]).reshape((3, 1))
            shoulder = casadi.DM(self.shoulder_positions[side]).reshape((3, 1))
            signed_distance = casadi.dot(outward, elbow_position - shoulder)
            elbow_violations.append(
                casadi.fmax(0.0, self.elbow_outer_margin - signed_distance)
            )
        elbow_violation = casadi.Function(
            "adam_nonlinear_elbow_outer_violation",
            [symbolic_q],
            [casadi.vertcat(*elbow_violations)],
        )

        problem = casadi.Opti()
        q = problem.variable(self.model.nq)
        q_last = problem.parameter(self.model.nq)
        q_bias = problem.parameter(self.model.nq)
        left_target = problem.parameter(4, 4)
        right_target = problem.parameter(4, 4)
        problem.subject_to(
            problem.bounded(
                self.model.lowerPositionLimit,
                q,
                self.model.upperPositionLimit,
            )
        )
        problem.minimize(
            self.nonlinear_translation_cost
            * casadi.sumsqr(translation_error(q, left_target, right_target))
            + self.nonlinear_rotation_cost
            * casadi.sumsqr(rotation_error(q, left_target, right_target))
            + self.nonlinear_posture_cost * casadi.sumsqr(q - q_bias)
            + self.nonlinear_smoothness_cost * casadi.sumsqr(q - q_last)
            + self.elbow_outer_cost * casadi.sumsqr(elbow_violation(q))
        )
        problem.solver(
            "ipopt",
            {
                "expand": True,
                "detect_simple_bounds": True,
                "calc_lam_p": False,
                "print_time": False,
                "ipopt.sb": "yes",
                "ipopt.print_level": 0,
                "ipopt.max_iter": 30,
                "ipopt.tol": 1.0e-4,
                "ipopt.acceptable_tol": 5.0e-4,
                "ipopt.acceptable_iter": 5,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.jacobian_approximation": "exact",
            },
        )
        self.nonlinear_problem = {
            "opti": problem,
            "q": q,
            "q_last": q_last,
            "q_bias": q_bias,
            "left_target": left_target,
            "right_target": right_target,
            "forward_kinematics": forward_kinematics,
        }

    @staticmethod
    def _casadi_axis_rotation(casadi, joint_type: str, angle):
        cosine = casadi.cos(angle)
        sine = casadi.sin(angle)
        if joint_type == "JointModelRX":
            return casadi.vertcat(
                casadi.horzcat(1.0, 0.0, 0.0),
                casadi.horzcat(0.0, cosine, -sine),
                casadi.horzcat(0.0, sine, cosine),
            )
        if joint_type == "JointModelRY":
            return casadi.vertcat(
                casadi.horzcat(cosine, 0.0, sine),
                casadi.horzcat(0.0, 1.0, 0.0),
                casadi.horzcat(-sine, 0.0, cosine),
            )
        if joint_type == "JointModelRZ":
            return casadi.vertcat(
                casadi.horzcat(cosine, -sine, 0.0),
                casadi.horzcat(sine, cosine, 0.0),
                casadi.horzcat(0.0, 0.0, 1.0),
            )
        raise ValueError(f"unsupported nonlinear IK joint type: {joint_type}")

    def _build_nonlinear_forward_kinematics(self, casadi, symbolic_q):
        if self.model is None:
            raise RuntimeError("solver has not been calibrated")
        world_rotations = [casadi.SX.eye(3)]
        world_translations = [casadi.SX.zeros(3, 1)]
        for joint_id in range(1, self.model.njoints):
            joint = self.model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    "nonlinear IK only supports scalar revolute arm joints; "
                    f"{self.model.names[joint_id]} is {joint.shortname()}"
                )
            parent_id = int(self.model.parents[joint_id])
            placement = self.model.jointPlacements[joint_id]
            parent_rotation = world_rotations[parent_id]
            placed_rotation = parent_rotation @ casadi.DM(placement.rotation)
            world_translations.append(
                world_translations[parent_id]
                + parent_rotation @ casadi.DM(placement.translation)
            )
            world_rotations.append(
                placed_rotation
                @ self._casadi_axis_rotation(
                    casadi,
                    joint.shortname(),
                    symbolic_q[joint.idx_q],
                )
            )

        transforms = []
        for side in ("Left", "Right"):
            frame = self.model.frames[self.model.getFrameId(EE_FRAMES[side])]
            parent_rotation = world_rotations[frame.parentJoint]
            frame_rotation = parent_rotation @ casadi.DM(frame.placement.rotation)
            frame_translation = (
                world_translations[frame.parentJoint]
                + parent_rotation @ casadi.DM(frame.placement.translation)
            )
            transforms.append(
                casadi.vertcat(
                    casadi.horzcat(frame_rotation, frame_translation),
                    casadi.DM([[0.0, 0.0, 0.0, 1.0]]),
                )
            )
        elbow_positions = []
        for side in ("Left", "Right"):
            frame = self.model.frames[self.model.getFrameId(ELBOW_FRAMES[side])]
            parent_rotation = world_rotations[frame.parentJoint]
            elbow_positions.append(
                world_translations[frame.parentJoint]
                + parent_rotation @ casadi.DM(frame.placement.translation)
            )
        return casadi.Function(
            "adam_nonlinear_forward_kinematics",
            [symbolic_q],
            transforms + elbow_positions,
        )

    @staticmethod
    def _casadi_rotation_residual(casadi, rotation):
        skew_vector = 0.5 * casadi.vertcat(
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
        sine_norm = casadi.sqrt(casadi.sumsqr(skew_vector) + 1.0e-12)
        cosine = 0.5 * (casadi.trace(rotation) - 1.0)
        angle = casadi.atan2(sine_norm, cosine)
        return casadi.vertcat(angle, 0.0, 0.0)

    @staticmethod
    def _pose_matrix(pose: Pose3) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = pose.rotation
        matrix[:3, 3] = pose.position
        return matrix

    def _solve_nonlinear(self) -> None:
        if (
            self.nonlinear_problem is None
            or self.targets is None
            or self.default_qpos is None
            or self.last_output_qpos is None
        ):
            raise RuntimeError("nonlinear IK has not been calibrated")
        problem = self.nonlinear_problem
        opti = problem["opti"]
        opti.set_initial(problem["q"], self.last_output_qpos)
        opti.set_value(problem["q_last"], self.last_output_qpos)
        opti.set_value(problem["q_bias"], self.default_qpos)
        opti.set_value(problem["left_target"], self._pose_matrix(self.targets["Left"]))
        opti.set_value(
            problem["right_target"], self._pose_matrix(self.targets["Right"])
        )
        try:
            solution = opti.solve()
            qpos = np.asarray(solution.value(problem["q"]), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise RuntimeError(f"nonlinear IK failed: {exc}") from exc
        if not np.all(np.isfinite(qpos)):
            raise RuntimeError("nonlinear IK produced non-finite joint positions")
        previous_qpos = self.last_output_qpos.copy()
        qpos = np.clip(
            qpos,
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )
        qpos = self._protect_outward_elbows(qpos, previous_qpos)
        if self.nonlinear_filter_enabled:
            qpos = self._filter_nonlinear_qpos(qpos)
            qpos = self._protect_outward_elbows(qpos, previous_qpos)
        self.last_output_qpos = qpos

    def _protect_outward_elbows(
        self,
        candidate_qpos: np.ndarray,
        previous_qpos: np.ndarray,
    ) -> np.ndarray:
        """Reject a single arm only when it moves farther inside the torso."""

        candidate = np.asarray(candidate_qpos, dtype=np.float64).copy()
        previous_distances = self.elbow_outer_distances(previous_qpos)
        candidate_distances = self.elbow_outer_distances(candidate)
        for side in ("Left", "Right"):
            candidate_distance = candidate_distances[side]
            previous_distance = previous_distances[side]
            if candidate_distance < 0.0 and candidate_distance < previous_distance:
                candidate[self.arm_side_indices[side]] = previous_qpos[
                    self.arm_side_indices[side]
                ]
        return candidate

    def _filter_nonlinear_qpos(self, qpos: np.ndarray) -> np.ndarray:
        self.nonlinear_history.append(np.asarray(qpos, dtype=np.float64).copy())
        weights = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        history = list(reversed(self.nonlinear_history))
        active_weights = weights[: len(history)]
        active_weights /= np.sum(active_weights)
        return np.sum(
            [weight * value for weight, value in zip(active_weights, history)],
            axis=0,
        )

    def solve(self, *, iterations: int, solve_dt: float) -> None:
        if self.retarget_method == "nonlinear_ik":
            self._solve_nonlinear()
        else:
            self._solve_pink(iterations=iterations, solve_dt=solve_dt)

    def positions_19(self) -> list[float]:
        if self.last_output_qpos is None or self.arm_indices is None:
            raise RuntimeError("solver has not been calibrated")
        arm_positions = {
            name: float(self.last_output_qpos[index])
            for name, index in zip(ARM_JOINTS, self.arm_indices, strict=True)
        }
        positions = {
            **self.waist_positions,
            **self.neck_positions,
            **arm_positions,
        }
        return [positions[name] for name in ADAM_COMMAND_JOINTS_19]

    def wrist_errors(self, targets: Mapping[str, Pose3]) -> dict[str, float]:
        if self.model is None or self.data is None or self.last_output_qpos is None:
            raise RuntimeError("solver has not been calibrated")
        pin.forwardKinematics(self.model, self.data, self.last_output_qpos)
        pin.updateFramePlacements(self.model, self.data)
        return {
            side: float(
                np.linalg.norm(
                    self.data.oMf[self.model.getFrameId(EE_FRAMES[side])].translation
                    - targets[side].position
                )
                * 1000.0
            )
            for side in ("Left", "Right")
        }

    def wrist_poses(self) -> dict[str, Pose3]:
        """Return the solved wrist poses for diagnostics."""

        if self.model is None or self.data is None or self.last_output_qpos is None:
            raise RuntimeError("solver has not been calibrated")
        pin.forwardKinematics(self.model, self.data, self.last_output_qpos)
        pin.updateFramePlacements(self.model, self.data)
        return {
            side: self._frame_pose(self.data, self.model, EE_FRAMES[side])
            for side in ("Left", "Right")
        }

    def elbow_outer_distances(
        self, qpos: np.ndarray | None = None
    ) -> dict[str, float]:
        """Return signed shoulder-to-elbow distances along each outward axis."""

        if self.model is None or self.data is None:
            raise RuntimeError("solver has not been calibrated")
        if qpos is None:
            if self.last_output_qpos is None:
                raise RuntimeError("solver has not been calibrated")
            qpos = self.last_output_qpos
        pin.forwardKinematics(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)
        return {
            side: float(
                self.elbow_outward_directions[side]
                @ (
                    self.data.oMf[
                        self.model.getFrameId(ELBOW_FRAMES[side])
                    ].translation
                    - self.shoulder_positions[side]
                )
            )
            for side in ("Left", "Right")
        }
