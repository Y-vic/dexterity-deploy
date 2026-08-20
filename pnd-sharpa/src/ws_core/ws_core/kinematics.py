"""DreamZero SharpA62 FK/IK helpers for workstation deployment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deploy_common.joints import ADAM_COMMAND_JOINTS_19, SHARPA_JOINT_NAMES


ACTION_HORIZON = 24
ACTION_DIM = 62
DEFAULT_MODEL_XML = Path("adam_pro_sharpa_kinematics.xml")

PND_TO_MUJOCO_JOINT = {
    "dof_pos/waistRoll": "waistRoll",
    "dof_pos/waistPitch": "waistPitch",
    "dof_pos/waistYaw": "waistYaw",
    "dof_pos/neckYaw": "neckYaw",
    "dof_pos/neckPitch": "neckPitch",
    "dof_pos/shoulderPitch_Left": "shoulderPitch_Left",
    "dof_pos/shoulderRoll_Left": "shoulderRoll_Left",
    "dof_pos/shoulderYaw_Left": "shoulderYaw_Left",
    "dof_pos/elbow_Left": "elbow_Left",
    "dof_pos/wristYaw_Left": "wristYaw_Left",
    "dof_pos/wristPitch_Left": "wristPitch_Left",
    "dof_pos/wristRoll_Left": "wristRoll_Left",
    "dof_pos/shoulderPitch_Right": "shoulderPitch_Right",
    "dof_pos/shoulderRoll_Right": "shoulderRoll_Right",
    "dof_pos/shoulderYaw_Right": "shoulderYaw_Right",
    "dof_pos/elbow_Right": "elbow_Right",
    "dof_pos/wristYaw_Right": "wristYaw_Right",
    "dof_pos/wristPitch_Right": "wristPitch_Right",
    "dof_pos/wristRoll_Right": "wristRoll_Right",
}

ARM_PND_JOINTS = [
    "dof_pos/shoulderPitch_Left",
    "dof_pos/shoulderRoll_Left",
    "dof_pos/shoulderYaw_Left",
    "dof_pos/elbow_Left",
    "dof_pos/wristYaw_Left",
    "dof_pos/wristPitch_Left",
    "dof_pos/wristRoll_Left",
    "dof_pos/shoulderPitch_Right",
    "dof_pos/shoulderRoll_Right",
    "dof_pos/shoulderYaw_Right",
    "dof_pos/elbow_Right",
    "dof_pos/wristYaw_Right",
    "dof_pos/wristPitch_Right",
    "dof_pos/wristRoll_Right",
]
WAIST_PND_JOINTS = [
    "dof_pos/waistRoll",
    "dof_pos/waistPitch",
    "dof_pos/waistYaw",
]
NECK_PND_JOINTS = [
    "dof_pos/neckYaw",
    "dof_pos/neckPitch",
]
PRETRAIN_SHARPA_JOINT_NAMES = [
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
]
PND_ROOT_TO_PRETRAIN = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
POSTTRAIN_RAW2HAND_BASE = {
    "left": (
        np.asarray(
            [
                [-0.05063197761774063, -0.9985009431838989, -0.020789900794625282],
                [-0.9941514730453491, 0.04840133339166641, 0.09654103964567184],
                [-0.09539006650447845, 0.025556374341249466, -0.995111882686615],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [0.006119123660027981, -0.004442085511982441, -0.03515041619539261],
            dtype=np.float32,
        ),
    ),
    "right": (
        np.asarray(
            [
                [-0.037729986011981964, 0.9990273714065552, -0.022819984704256058],
                [0.9962800741195679, 0.035836100578308105, -0.07836932688951492],
                [-0.07747532427310944, -0.02569197118282318, -0.9966631531715393],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [0.003884613746777177, 0.002192644402384758, -0.035935886204242706],
            dtype=np.float32,
        ),
    ),
}


class KinematicsError(RuntimeError):
    """Raised when FK/IK input cannot produce a legal command."""


@dataclass
class ConvertedState:
    hip_pose_9d_world: np.ndarray
    left_wrist_9d_hip: np.ndarray
    right_wrist_9d_hip: np.ndarray
    sharpa_q44: np.ndarray
    hand_pose_62d: np.ndarray
    report: dict[str, Any]


@dataclass
class ActionTargets:
    action_abs62: np.ndarray
    selected_abs62: np.ndarray
    adam_q19: list[float]
    adam_valid: bool
    adam_source: str
    sharpa_q44: list[float]
    sharpa_valid: bool
    ik_qpos: np.ndarray | None
    report: dict[str, Any]


def resolve_model_xml(value: str | Path | None) -> Path:
    raw = Path(value) if value else _default_model_xml()
    candidates = [raw]
    if not raw.is_absolute():
        cwd = Path.cwd()
        candidates.append(cwd / raw)
        parents = list(Path(__file__).resolve().parents)
        for parent in parents:
            candidates.append(parent / raw)
            candidates.append(parent / "Deploy" / raw)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()
    raise KinematicsError(f"MuJoCo model XML not found: {raw}")


def _default_model_xml() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory("adam_sharpa_description"))
            / "mujoco"
            / DEFAULT_MODEL_XML
        )
    except Exception:
        return (
            Path(__file__).resolve().parents[2]
            / "adam_sharpa_description"
            / "mujoco"
            / DEFAULT_MODEL_XML
        )


def _require_mujoco():
    try:
        import mujoco  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise KinematicsError("Python package 'mujoco' is required for FK/IK") from exc
    return mujoco


def _require_least_squares():
    try:
        from scipy.optimize import least_squares  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise KinematicsError("Python package 'scipy' is required for IK") from exc
    return least_squares


def finite_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise KinematicsError(f"{name} is not numeric") from exc
    if array.shape != shape:
        raise KinematicsError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise KinematicsError(f"{name} contains NaN or Inf")
    return array


def mat_to_rot6d(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float32).reshape(3, 3)
    return np.concatenate([rot[:, 0], rot[:, 1]]).astype(np.float32)


def rot6d_to_mat(r6d: np.ndarray) -> np.ndarray:
    r6d = np.asarray(r6d, dtype=np.float32).reshape(6)
    a1 = r6d[:3]
    a2 = r6d[3:]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return np.eye(3, dtype=np.float32)
    b1 = a1 / n1
    b2 = a2 - float(np.dot(b1, a2)) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-8:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(seed, b1))) > 0.95:
            seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b2 = seed - float(np.dot(b1, seed)) * b1
        n2 = np.linalg.norm(b2)
    b2 = b2 / max(float(n2), 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1).astype(np.float32)


def matrix_to_rotvec(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    cos_theta = (float(np.trace(rot)) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    vee = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-6:
        return (0.5 * vee).astype(np.float32)
    return (theta / (2.0 * math.sin(theta)) * vee).astype(np.float32)


def pose9_relative_to_absolute(relative_pose9: np.ndarray, reference_pose9: np.ndarray) -> np.ndarray:
    p_ref = np.asarray(reference_pose9[:3], dtype=np.float32)
    r_ref = rot6d_to_mat(reference_pose9[3:9])
    p_rel = np.asarray(relative_pose9[:3], dtype=np.float32)
    r_rel = rot6d_to_mat(relative_pose9[3:9])
    p_abs = p_ref + r_ref @ p_rel
    r_abs = r_ref @ r_rel
    return np.concatenate([p_abs, mat_to_rot6d(r_abs)]).astype(np.float32)


def pose9_absolute_to_relative(absolute_pose9: np.ndarray, reference_pose9: np.ndarray) -> np.ndarray:
    p_ref = np.asarray(reference_pose9[:3], dtype=np.float32)
    r_ref = rot6d_to_mat(reference_pose9[3:9])
    p_abs = np.asarray(absolute_pose9[:3], dtype=np.float32)
    r_abs = rot6d_to_mat(absolute_pose9[3:9])
    p_rel = r_ref.T @ (p_abs - p_ref)
    r_rel = r_ref.T @ r_abs
    return np.concatenate([p_rel, mat_to_rot6d(r_rel)]).astype(np.float32)


def action62_relative_to_absolute(action_62d: np.ndarray, anchor_state_62d: np.ndarray) -> np.ndarray:
    action = np.asarray(action_62d, dtype=np.float32)
    if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] != ACTION_DIM:
        raise KinematicsError(
            f"action_hand_pose_62d has shape {action.shape}, expected (T, {ACTION_DIM})"
        )
    if not np.all(np.isfinite(action)):
        raise KinematicsError("action_hand_pose_62d contains NaN or Inf")
    action = action.copy()
    anchor = finite_array("anchor_state_62d", anchor_state_62d, (ACTION_DIM,))
    for row_idx in range(action.shape[0]):
        for offset in (0, 9):
            action[row_idx, offset : offset + 9] = pose9_relative_to_absolute(
                action[row_idx, offset : offset + 9],
                anchor[offset : offset + 9],
            )
    return action.astype(np.float32)


def action62_absolute_to_relative(action_62d: np.ndarray, anchor_state_62d: np.ndarray) -> np.ndarray:
    action = np.asarray(action_62d, dtype=np.float32)
    if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] != ACTION_DIM:
        raise KinematicsError(
            f"action_hand_pose_62d has shape {action.shape}, expected (T, {ACTION_DIM})"
        )
    if not np.all(np.isfinite(action)):
        raise KinematicsError("action_hand_pose_62d contains NaN or Inf")
    action = action.copy()
    anchor = finite_array("anchor_state_62d", anchor_state_62d, (ACTION_DIM,))
    for row_idx in range(action.shape[0]):
        for offset in (0, 9):
            action[row_idx, offset : offset + 9] = pose9_absolute_to_relative(
                action[row_idx, offset : offset + 9],
                anchor[offset : offset + 9],
            )
    return action.astype(np.float32)


def _reorder_joint_matrix(
    values: np.ndarray,
    source_names: list[str],
    target_names: list[str],
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(source_names):
        raise KinematicsError(
            f"joint matrix has shape {matrix.shape}, expected (T, {len(source_names)})"
        )
    index = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in index]
    if missing:
        raise KinematicsError(f"joint matrix is missing names: {missing}")
    return np.ascontiguousarray(matrix[:, [index[name] for name in target_names]])


def physical62_to_posttrain_absolute(action_62d: np.ndarray) -> np.ndarray:
    action = np.asarray(action_62d, dtype=np.float32)
    if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] != ACTION_DIM:
        raise KinematicsError(f"physical action has shape {action.shape}, expected (T, 62)")
    out = np.empty_like(action)
    for row_idx in range(action.shape[0]):
        for side, offset in (("left", 0), ("right", 9)):
            physical_pose = action[row_idx, offset : offset + 9]
            physical_pos = physical_pose[:3]
            physical_rot = rot6d_to_mat(physical_pose[3:9])
            base_rot, base_trans = POSTTRAIN_RAW2HAND_BASE[side]
            baked_pos = physical_pos + physical_rot @ base_trans
            baked_rot = physical_rot @ base_rot
            out[row_idx, offset : offset + 9] = np.concatenate(
                [
                    PND_ROOT_TO_PRETRAIN @ baked_pos,
                    mat_to_rot6d(PND_ROOT_TO_PRETRAIN @ baked_rot),
                ]
            )
    out[:, 18:62] = _reorder_joint_matrix(
        action[:, 18:62],
        list(SHARPA_JOINT_NAMES),
        PRETRAIN_SHARPA_JOINT_NAMES,
    )
    return out.astype(np.float32)


def posttrain_absolute_to_physical62(action_62d: np.ndarray) -> np.ndarray:
    action = np.asarray(action_62d, dtype=np.float32)
    if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] != ACTION_DIM:
        raise KinematicsError(f"posttrain action has shape {action.shape}, expected (T, 62)")
    out = np.empty_like(action)
    for row_idx in range(action.shape[0]):
        for side, offset in (("left", 0), ("right", 9)):
            baked_pose = action[row_idx, offset : offset + 9]
            baked_pos = PND_ROOT_TO_PRETRAIN.T @ baked_pose[:3]
            baked_rot = PND_ROOT_TO_PRETRAIN.T @ rot6d_to_mat(baked_pose[3:9])
            base_rot, base_trans = POSTTRAIN_RAW2HAND_BASE[side]
            physical_rot = baked_rot @ base_rot.T
            physical_pos = baked_pos - physical_rot @ base_trans
            out[row_idx, offset : offset + 9] = np.concatenate(
                [physical_pos, mat_to_rot6d(physical_rot)]
            )
    out[:, 18:62] = _reorder_joint_matrix(
        action[:, 18:62],
        PRETRAIN_SHARPA_JOINT_NAMES,
        list(SHARPA_JOINT_NAMES),
    )
    return out.astype(np.float32)


def _joint_values_by_name(section: dict[str, Any], expected_names: list[str], label: str) -> np.ndarray:
    names = section.get("name")
    q = section.get("q")
    if not isinstance(names, list) or not isinstance(q, list):
        raise KinematicsError(f"{label} state is missing name/q")
    by_name = {str(name): value for name, value in zip(names, q, strict=False)}
    values = []
    missing = []
    for name in expected_names:
        if name not in by_name:
            missing.append(name)
            values.append(0.0)
        else:
            values.append(by_name[name])
    if missing:
        raise KinematicsError(f"{label} state missing joints: {missing[:6]}")
    return finite_array(f"{label}.q", values, (len(expected_names),))


def robot_sections(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    adam = state.get("adam")
    sharpa = state.get("sharpa")
    if not isinstance(adam, dict):
        raise KinematicsError("robot_state missing adam section")
    if not isinstance(sharpa, dict):
        raise KinematicsError("robot_state missing sharpa section")
    return adam, sharpa


class PndKinematics:
    def __init__(
        self,
        model_xml: str | Path | None = None,
        *,
        include_waist: bool = False,
        include_neck: bool = False,
        ik_max_nfev: int = 80,
        ik_pos_weight: float = 45.0,
        ik_rot_weight: float = 3.5,
        ik_reg_weight: float = 0.08,
        ik_smooth_weight: float = 0.04,
        ik_diff_step: float = 1e-4,
        fallback_unlimited_low: float = -math.pi,
        fallback_unlimited_high: float = math.pi,
        fallback_sharpa_low: float = -1.5,
        fallback_sharpa_high: float = 2.1,
    ) -> None:
        self.model_xml = resolve_model_xml(model_xml)
        self.mujoco = _require_mujoco()
        self.least_squares = _require_least_squares()
        self.model = self.mujoco.MjModel.from_xml_path(str(self.model_xml))
        self.data = self.mujoco.MjData(self.model)
        self.ik_max_nfev = int(ik_max_nfev)
        self.ik_pos_weight = float(ik_pos_weight)
        self.ik_rot_weight = float(ik_rot_weight)
        self.ik_reg_weight = float(ik_reg_weight)
        self.ik_smooth_weight = float(ik_smooth_weight)
        self.ik_diff_step = float(ik_diff_step)
        if not math.isfinite(self.ik_diff_step) or self.ik_diff_step <= 0.0:
            raise KinematicsError("ik_diff_step must be positive")
        self.fallback_unlimited_low = float(fallback_unlimited_low)
        self.fallback_unlimited_high = float(fallback_unlimited_high)
        self.fallback_sharpa_low = float(fallback_sharpa_low)
        self.fallback_sharpa_high = float(fallback_sharpa_high)
        self.joint_qpos_addr: dict[str, int] = {}
        self.joint_ranges: dict[str, tuple[float, float]] = {}
        for joint_id in range(self.model.njnt):
            name = self.mujoco.mj_id2name(
                self.model,
                self.mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if not name:
                continue
            self.joint_qpos_addr[name] = int(self.model.jnt_qposadr[joint_id])
            if bool(self.model.jnt_limited[joint_id]):
                low, high = self.model.jnt_range[joint_id]
            else:
                low, high = self.fallback_unlimited_low, self.fallback_unlimited_high
            self.joint_ranges[name] = (float(low), float(high))

        self.body_ids = {
            "hip": self._body_id("pelvis"),
            "left": self._body_id("wristRollLeft"),
            "right": self._body_id("wristRollRight"),
        }
        self.variable_pnd_joints = (WAIST_PND_JOINTS if include_waist else []) + ARM_PND_JOINTS
        self.command_pnd_joints = (
            (WAIST_PND_JOINTS if include_waist else [])
            + (NECK_PND_JOINTS if include_neck else [])
            + ARM_PND_JOINTS
        )
        self.variable_mujoco_joints = [PND_TO_MUJOCO_JOINT[name] for name in self.variable_pnd_joints]
        self.variable_addrs = [self.joint_qpos_addr[name] for name in self.variable_mujoco_joints]

    def _body_id(self, name: str) -> int:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KinematicsError(f"MuJoCo body not found: {name}")
        return int(body_id)

    def convert_state(self, state: dict[str, Any]) -> ConvertedState:
        adam_section, sharpa_section = robot_sections(state)
        adam_names = [str(name) for name in adam_section.get("name", [])]
        adam_q = finite_array("adam.q", adam_section.get("q"), (len(adam_names),))
        sharpa_q44 = _joint_values_by_name(sharpa_section, list(SHARPA_JOINT_NAMES), "sharpa")
        qpos = self.qpos_from_pnd(adam_names, adam_q, normalize_neck=True)
        self.set_sharpa_q44(qpos, sharpa_q44)
        self.data.qpos[:] = qpos
        self.mujoco.mj_forward(self.model, self.data)

        hip_pose = self._body_pose_9d_world(self.body_ids["hip"])
        left_wrist = self._body_pose_9d_in_hip(self.body_ids["left"], self.body_ids["hip"])
        right_wrist = self._body_pose_9d_in_hip(self.body_ids["right"], self.body_ids["hip"])
        hand_pose = np.concatenate([left_wrist, right_wrist, sharpa_q44], axis=0).astype(np.float32)
        report = {
            "schema": "ws.fk_report.v1",
            "model_xml": str(self.model_xml),
            "left_wrist_9d_hip": left_wrist.tolist(),
            "right_wrist_9d_hip": right_wrist.tolist(),
            "sharpa_q44_source": "robot_state",
            "finite": True,
        }
        return ConvertedState(
            hip_pose_9d_world=hip_pose,
            left_wrist_9d_hip=left_wrist,
            right_wrist_9d_hip=right_wrist,
            sharpa_q44=sharpa_q44,
            hand_pose_62d=hand_pose,
            report=report,
        )

    def plan_action(
        self,
        *,
        action_rel62: Any,
        anchor_state_62d: Any,
        robot_state: dict[str, Any],
        action_step_index: int,
        enable_adam: bool,
        enable_sharpa: bool,
        action_frame: str = "relative_eef",
        qpos_previous: Any | None = None,
    ) -> ActionTargets:
        action_in = np.asarray(action_rel62, dtype=np.float32)
        if action_in.ndim != 2 or action_in.shape[0] <= 0 or action_in.shape[1] != ACTION_DIM:
            raise KinematicsError(
                f"action_hand_pose_62d has shape {action_in.shape}, expected (T, {ACTION_DIM})"
            )
        if not np.all(np.isfinite(action_in)):
            raise KinematicsError("action_hand_pose_62d contains NaN or Inf")
        anchor = finite_array("anchor_state_62d", anchor_state_62d, (ACTION_DIM,))
        frame = str(action_frame or "relative_eef")
        if frame in {"absolute_current_hip", "absolute_hip", "current_robot_hip"}:
            action_abs = action_in.copy()
        else:
            action_abs = action62_relative_to_absolute(action_in, anchor)
        if action_step_index < 0 or action_step_index >= action_in.shape[0]:
            raise KinematicsError("action_step_index outside action horizon")
        selected = action_abs[action_step_index]

        adam_section, sharpa_section = robot_sections(robot_state)
        adam_names = [str(name) for name in adam_section.get("name", [])]
        adam_q = finite_array("adam.q", adam_section.get("q"), (len(adam_names),))
        current_q19 = self.current_adam_q19(adam_section)
        qpos_ref = self.qpos_from_pnd(adam_names, adam_q, normalize_neck=False)

        ik_info: dict[str, Any] | None = None
        adam_source = "disabled"
        adam_valid = False
        adam_q19 = current_q19.copy()
        adam_limit_report: dict[str, Any] = {"checked": False}
        qpos_out: np.ndarray | None = None
        if enable_adam:
            qpos_start = (
                qpos_ref.copy()
                if qpos_previous is None
                else finite_array(
                    "qpos_previous",
                    qpos_previous,
                    qpos_ref.shape,
                ).copy()
            )
            qpos_out, ik_info = self.solve_ik_step(qpos_ref, qpos_start, selected)
            adam_q19_np = self.qpos_to_pnd_command_positions(qpos_out, adam_section)
            adam_q19_np, adam_limit_report = self.clip_named_values(
                list(ADAM_COMMAND_JOINTS_19),
                adam_q19_np,
                kind="adam",
            )
            adam_q19 = adam_q19_np.astype(float).tolist()
            adam_source = "mujoco_ik_from_action_wrist18"
            adam_valid = True

        sharpa_q44 = [0.0] * len(SHARPA_JOINT_NAMES)
        sharpa_valid = False
        sharpa_limit_report: dict[str, Any] = {"checked": False}
        if enable_sharpa:
            raw_sharpa = finite_array("action.sharpa_q44", selected[18:62], (44,))
            clipped_sharpa, sharpa_limit_report = self.clip_named_values(
                list(SHARPA_JOINT_NAMES),
                raw_sharpa,
                kind="sharpa",
            )
            sharpa_q44 = clipped_sharpa.astype(float).tolist()
            sharpa_valid = True

        self._assert_finite_list("adam_q19", adam_q19)
        self._assert_finite_list("sharpa_q44", sharpa_q44)
        report = {
            "schema": "ws.action_plan.v1",
            "model_xml": str(self.model_xml),
            "action_space": "sharpa_dexretarget_position_62d",
            "input_action_frame": frame,
            "command_action_frame": "absolute_current_hip",
            "selected_action_step": int(action_step_index),
            "finite": True,
            "safety": {
                "rules": ["finite_values", "joint_position_limits"],
                "adam_limits": adam_limit_report,
                "sharpa_limits": sharpa_limit_report,
                "removed_rules": [
                    "max_step_delta",
                    "wrist_workspace",
                    "ik_error_gate",
                    "strict_clip_reject",
                ],
            },
            "ik": ik_info or {"enabled": False},
        }
        return ActionTargets(
            action_abs62=action_abs,
            selected_abs62=selected,
            adam_q19=adam_q19,
            adam_valid=adam_valid,
            adam_source=adam_source,
            sharpa_q44=sharpa_q44,
            sharpa_valid=sharpa_valid,
            ik_qpos=qpos_out,
            report=report,
        )

    def current_adam_q19(self, adam_section: dict[str, Any]) -> list[float]:
        return _joint_values_by_name(adam_section, list(ADAM_COMMAND_JOINTS_19), "adam_command").astype(float).tolist()

    def qpos_from_pnd(
        self,
        pnd_names: list[str],
        pnd_positions: np.ndarray,
        *,
        normalize_neck: bool,
    ) -> np.ndarray:
        values = {name: float(value) for name, value in zip(pnd_names, pnd_positions, strict=False)}
        if normalize_neck and "dof_pos/neckYaw" in values and "dof_pos/neckPitch" in values:
            values["dof_pos/neckYaw"], values["dof_pos/neckPitch"] = (
                values["dof_pos/neckPitch"],
                values["dof_pos/neckYaw"],
            )
        qpos = self.model.qpos0.copy()
        for pnd_name, mujoco_name in PND_TO_MUJOCO_JOINT.items():
            if pnd_name in values and mujoco_name in self.joint_qpos_addr:
                qpos[self.joint_qpos_addr[mujoco_name]] = values[pnd_name]
        return qpos

    def set_sharpa_q44(self, qpos: np.ndarray, q44: np.ndarray) -> None:
        for joint_name, value in zip(SHARPA_JOINT_NAMES, np.asarray(q44, dtype=np.float32), strict=True):
            addr = self.joint_qpos_addr.get(joint_name)
            if addr is not None:
                qpos[addr] = float(value)

    def solve_ik_step(
        self,
        qpos_reference: np.ndarray,
        qpos_previous: np.ndarray,
        action_abs62: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target_left = finite_array("target_left_wrist9", action_abs62[0:9], (9,))
        target_right = finite_array("target_right_wrist9", action_abs62[9:18], (9,))
        qpos_base = qpos_previous.copy()
        self.set_sharpa_q44(qpos_base, finite_array("target_sharpa_q44", action_abs62[18:62], (44,)))
        if not self.variable_addrs:
            return qpos_base, {"enabled": False, "reason": "no_variable_joints"}

        x0 = qpos_base[self.variable_addrs].astype(np.float64)
        x_ref = qpos_reference[self.variable_addrs].astype(np.float64)
        lower = []
        upper = []
        for mujoco_name in self.variable_mujoco_joints:
            low, high = self.joint_ranges.get(
                mujoco_name,
                (self.fallback_unlimited_low, self.fallback_unlimited_high),
            )
            lower.append(low)
            upper.append(high)
        lower_arr = np.asarray(lower, dtype=np.float64)
        upper_arr = np.asarray(upper, dtype=np.float64)
        x0 = np.clip(x0, lower_arr, upper_arr)

        def residual(x: np.ndarray) -> np.ndarray:
            qpos = qpos_base.copy()
            qpos[self.variable_addrs] = x
            cur_left = self.body_pose_hip(qpos, "left")
            cur_right = self.body_pose_hip(qpos, "right")
            pieces = []
            for current, target in ((cur_left, target_left), (cur_right, target_right)):
                pieces.append((current[:3] - target[:3]) * self.ik_pos_weight)
                r_current = rot6d_to_mat(current[3:9])
                r_target = rot6d_to_mat(target[3:9])
                pieces.append(matrix_to_rotvec(r_current.T @ r_target) * self.ik_rot_weight)
            pieces.append((x - x_ref) * self.ik_reg_weight)
            pieces.append((x - x0) * self.ik_smooth_weight)
            return np.concatenate(pieces, axis=0)

        def jacobian(x: np.ndarray) -> np.ndarray:
            base_residual = residual(x).astype(np.float64)
            jac = np.empty((base_residual.size, x.size), dtype=np.float64)
            for column in range(x.size):
                direction = 1.0
                if x[column] + self.ik_diff_step > upper_arr[column]:
                    direction = -1.0
                shifted = x.copy()
                shifted[column] += direction * self.ik_diff_step
                shifted[column] = np.clip(
                    shifted[column], lower_arr[column], upper_arr[column]
                )
                actual_step = shifted[column] - x[column]
                if abs(actual_step) < 1e-12:
                    jac[:, column] = 0.0
                    continue
                jac[:, column] = (
                    residual(shifted).astype(np.float64) - base_residual
                ) / actual_step
            return jac

        result = self.least_squares(
            residual,
            x0,
            jac=jacobian,
            bounds=(lower_arr, upper_arr),
            max_nfev=self.ik_max_nfev,
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-4,
        )
        qpos_out = qpos_base.copy()
        qpos_out[self.variable_addrs] = result.x
        achieved_left = self.body_pose_hip(qpos_out, "left")
        achieved_right = self.body_pose_hip(qpos_out, "right")
        left_rotation_error = matrix_to_rotvec(
            rot6d_to_mat(achieved_left[3:9]).T @ rot6d_to_mat(target_left[3:9])
        )
        right_rotation_error = matrix_to_rotvec(
            rot6d_to_mat(achieved_right[3:9]).T @ rot6d_to_mat(target_right[3:9])
        )
        info = {
            "enabled": True,
            "success": bool(result.success),
            "status": int(result.status),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "diff_step": self.ik_diff_step,
            "left_position_error_m": float(np.linalg.norm(achieved_left[:3] - target_left[:3])),
            "right_position_error_m": float(np.linalg.norm(achieved_right[:3] - target_right[:3])),
            "left_orientation_error_rad": float(np.linalg.norm(left_rotation_error)),
            "right_orientation_error_rad": float(np.linalg.norm(right_rotation_error)),
            "left_orientation_error_deg": float(
                np.degrees(np.linalg.norm(left_rotation_error))
            ),
            "right_orientation_error_deg": float(
                np.degrees(np.linalg.norm(right_rotation_error))
            ),
            "note": "IK error is reported only; it is not a safety gate.",
        }
        return qpos_out, info

    def body_pose_hip(self, qpos: np.ndarray, side: str) -> np.ndarray:
        self.data.qpos[:] = qpos
        self.mujoco.mj_forward(self.model, self.data)
        hip_id = self.body_ids["hip"]
        body_id = self.body_ids[side]
        return self._body_pose_9d_in_hip(body_id, hip_id)

    def qpos_to_pnd_command_positions(
        self,
        qpos: np.ndarray,
        adam_section: dict[str, Any],
    ) -> np.ndarray:
        current = _joint_values_by_name(adam_section, list(ADAM_COMMAND_JOINTS_19), "adam_command")
        out = current.astype(np.float32).copy()
        for idx, pnd_name in enumerate(ADAM_COMMAND_JOINTS_19):
            mujoco_name = PND_TO_MUJOCO_JOINT.get(pnd_name)
            if mujoco_name is None:
                continue
            addr = self.joint_qpos_addr.get(mujoco_name)
            if addr is not None:
                out[idx] = float(qpos[addr])
        return out.astype(np.float32)

    def clip_named_values(
        self,
        names: list[str],
        values: np.ndarray,
        *,
        kind: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        out = finite_array(f"{kind}.target", values, (len(names),)).copy()
        clipped = []
        missing_limits = []
        for idx, name in enumerate(names):
            low, high, source = self._limit_for_name(name, kind)
            if source == "missing":
                missing_limits.append(name)
            before = float(out[idx])
            out[idx] = float(np.clip(before, low, high))
            if float(out[idx]) != before:
                clipped.append(
                    {
                        "name": name,
                        "before": before,
                        "after": float(out[idx]),
                        "low": low,
                        "high": high,
                    }
                )
        return out.astype(np.float32), {
            "checked": True,
            "kind": kind,
            "clipped_count": len(clipped),
            "clipped": clipped,
            "missing_limit_names": missing_limits,
        }

    def _limit_for_name(self, pnd_or_sharpa_name: str, kind: str) -> tuple[float, float, str]:
        if kind == "adam":
            mujoco_name = PND_TO_MUJOCO_JOINT.get(pnd_or_sharpa_name)
            if mujoco_name and mujoco_name in self.joint_ranges:
                low, high = self.joint_ranges[mujoco_name]
                return low, high, "mujoco"
            return self.fallback_unlimited_low, self.fallback_unlimited_high, "missing"
        if pnd_or_sharpa_name in self.joint_ranges:
            low, high = self.joint_ranges[pnd_or_sharpa_name]
            return low, high, "mujoco"
        return self.fallback_sharpa_low, self.fallback_sharpa_high, "fallback_sharpa"

    def _body_pose_9d_world(self, body_id: int) -> np.ndarray:
        pos = self.data.xpos[body_id].astype(np.float32)
        rot = self.data.xmat[body_id].reshape(3, 3).astype(np.float32)
        return np.concatenate([pos, mat_to_rot6d(rot)], axis=0).astype(np.float32)

    def _body_pose_9d_in_hip(self, body_id: int, hip_id: int) -> np.ndarray:
        hip_pos = self.data.xpos[hip_id].astype(np.float32)
        hip_rot = self.data.xmat[hip_id].reshape(3, 3).astype(np.float32)
        body_pos = self.data.xpos[body_id].astype(np.float32)
        body_rot = self.data.xmat[body_id].reshape(3, 3).astype(np.float32)
        rel_pos = hip_rot.T @ (body_pos - hip_pos)
        rel_rot = hip_rot.T @ body_rot
        return np.concatenate([rel_pos, mat_to_rot6d(rel_rot)], axis=0).astype(np.float32)

    @staticmethod
    def _assert_finite_list(name: str, values: list[float]) -> None:
        if not all(math.isfinite(float(value)) for value in values):
            raise KinematicsError(f"{name} contains NaN or Inf")
