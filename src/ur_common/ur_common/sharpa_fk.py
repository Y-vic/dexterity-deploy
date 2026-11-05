"""SharpA hand fingertip FK, ported from dexhand_kinematics/sharpa_fk.py.

Joint sequences follow the URDF declaration order so they line up with the
Pinocchio model. That order is identical to the frozen interface order in
`sharpa_interface.sharpa.JOINT_NAMES_PER_HAND`, just with the `left_`/`right_`
prefix, so the 44D interface vector maps straight through (left 22, then
right 22).

Only the two .urdf files were copied into `assets/`; the STL meshes were not.
The URDFs reference them as `package://` visual/collision geometry, which
`pin.buildModelFromUrdf` does not read -- FK needs the kinematic tree only.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

# pinocchio is imported lazily via .hand_wrapper so that the joint-order
# constants, URDF paths, and the interface split helper stay importable on a
# machine without pinocchio. Only the FK calls actually need it.
if TYPE_CHECKING:  # pragma: no cover
    from .hand_wrapper import RobotWrapper

# URDF paths resolve from this file's location so the package stays relocatable.
_URDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "hand_urdf")
LEFT_URDF_PATH = os.path.join(_URDF_DIR, "left_sharpa_ha4", "left_sharpa_ha4_v2_1.urdf")
RIGHT_URDF_PATH = os.path.join(_URDF_DIR, "right_sharpa_ha4", "right_sharpa_ha4_v2_1.urdf")

NUM_JOINTS_PER_HAND = 22
NUM_JOINTS_TOTAL = NUM_JOINTS_PER_HAND * 2  # 44
NUM_FINGERTIPS_PER_HAND = 5

_JOINT_SUFFIXES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)
_TIP_SUFFIXES = ("thumb_fingertip", "index_fingertip", "middle_fingertip",
                 "ring_fingertip", "pinky_fingertip")

LEFT_JOINT_ORDER = [f"left_{name}" for name in _JOINT_SUFFIXES]
RIGHT_JOINT_ORDER = [f"right_{name}" for name in _JOINT_SUFFIXES]
LEFT_TIP_LINKS = [f"left_{name}" for name in _TIP_SUFFIXES]
RIGHT_TIP_LINKS = [f"right_{name}" for name in _TIP_SUFFIXES]

JOINT_ORDER = {"left": LEFT_JOINT_ORDER, "right": RIGHT_JOINT_ORDER}
TIP_LINKS = {"left": LEFT_TIP_LINKS, "right": RIGHT_TIP_LINKS}
URDF_PATH = {"left": LEFT_URDF_PATH, "right": RIGHT_URDF_PATH}

_KEYPOINT_SUFFIXES = (
    "hand_C_MC",
    "thumb_MC", "thumb_PP", "thumb_DP", "thumb_fingertip",
    "index_PP", "index_MP", "index_DP", "index_fingertip",
    "middle_PP", "middle_MP", "middle_DP", "middle_fingertip",
    "ring_PP", "ring_MP", "ring_DP", "ring_fingertip",
    "pinky_PP", "pinky_MP", "pinky_DP", "pinky_fingertip",
)
HAND_KEYPOINT_LINKS = {
    side: [f"{side}_{suffix}" for suffix in _KEYPOINT_SUFFIXES]
    for side in ("left", "right")
}


def _reorder_for_model(order_names, provided_values, robot: "RobotWrapper") -> np.ndarray:
    if len(provided_values) != len(order_names):
        raise ValueError(
            f"Expected {len(order_names)} joint values, got {len(provided_values)}."
        )
    name_to_value = {name: float(val) for name, val in zip(order_names, provided_values)}
    model_names = [name for name in robot.dof_joint_names if name in name_to_value]
    if len(model_names) != len(order_names):
        missing = sorted(set(order_names) - set(model_names))
        raise ValueError(f"Joint names missing in model: {missing}")
    return np.array([name_to_value[name] for name in model_names], dtype=float)


# Cache RobotWrapper so per-frame FK does not rebuild Pinocchio models from
# URDF. Rebuilding per frame was the original performance problem; do not
# remove this. `hand_model_cache_info()` lets tests assert reuse.
_LEFT_HAND: "RobotWrapper | None" = None
_RIGHT_HAND: "RobotWrapper | None" = None
_CACHE_BUILDS = {"left": 0, "right": 0}


def _get_hand_models() -> tuple["RobotWrapper", "RobotWrapper"]:
    global _LEFT_HAND, _RIGHT_HAND
    if _LEFT_HAND is None or _RIGHT_HAND is None:
        from .hand_wrapper import RobotWrapper  # needs pinocchio

        if _LEFT_HAND is None:
            _LEFT_HAND = RobotWrapper(LEFT_URDF_PATH)
            _CACHE_BUILDS["left"] += 1
        if _RIGHT_HAND is None:
            _RIGHT_HAND = RobotWrapper(RIGHT_URDF_PATH)
            _CACHE_BUILDS["right"] += 1
    return _LEFT_HAND, _RIGHT_HAND


def hand_model_cache_info() -> dict:
    """How many times each hand model was actually built from URDF.

    Stays at 1 per hand no matter how many FK calls are made; that is the whole
    point of the cache.
    """
    return {
        "left_builds": _CACHE_BUILDS["left"],
        "right_builds": _CACHE_BUILDS["right"],
        "left_loaded": _LEFT_HAND is not None,
        "right_loaded": _RIGHT_HAND is not None,
    }


def clear_hand_model_cache() -> None:
    """Drop the cached models. For tests only -- never call this per frame."""
    global _LEFT_HAND, _RIGHT_HAND
    _LEFT_HAND = None
    _RIGHT_HAND = None
    _CACHE_BUILDS["left"] = 0
    _CACHE_BUILDS["right"] = 0


def hand_joints_to_fingertip_poses(left_joints: Any, right_joints: Any) -> list:
    """22+22 joint angles (rad) -> list of 10 fingertip POSITIONS (3,).

    Order: left thumb..pinky, then right thumb..pinky. Signature and return
    shape are unchanged from v1 so existing call sites keep working.
    """
    left_hand, right_hand = _get_hand_models()

    left_q = _reorder_for_model(LEFT_JOINT_ORDER, left_joints, left_hand)
    right_q = _reorder_for_model(RIGHT_JOINT_ORDER, right_joints, right_hand)

    left_hand.compute_forward_kinematics(left_q)
    right_hand.compute_forward_kinematics(right_q)

    poses = []
    for link_name in LEFT_TIP_LINKS:
        pose = left_hand.get_link_pose(left_hand.get_link_index(link_name))
        poses.append(pose[:3, 3])
    for link_name in RIGHT_TIP_LINKS:
        pose = right_hand.get_link_pose(right_hand.get_link_index(link_name))
        poses.append(pose[:3, 3])
    return poses


def hand_joints_to_fingertip_transforms(left_joints: Any, right_joints: Any) -> dict:
    """22+22 joint angles -> {'left': (5,4,4), 'right': (5,4,4)}.

    Full homogeneous fingertip poses. The wrapper already produces 4x4, so this
    is the same FK as above without discarding the orientation.
    """
    left_hand, right_hand = _get_hand_models()
    left_q = _reorder_for_model(LEFT_JOINT_ORDER, left_joints, left_hand)
    right_q = _reorder_for_model(RIGHT_JOINT_ORDER, right_joints, right_hand)
    left_hand.compute_forward_kinematics(left_q)
    right_hand.compute_forward_kinematics(right_q)

    out = {}
    for side, hand, links in (("left", left_hand, LEFT_TIP_LINKS),
                              ("right", right_hand, RIGHT_TIP_LINKS)):
        out[side] = np.stack(
            [np.asarray(hand.get_link_pose(hand.get_link_index(name)), dtype=np.float64)
             for name in links]
        )
    return out


def hand_joints_to_keypoints(left_joints: Any, right_joints: Any) -> dict:
    """22+22 joint angles -> 21 hand landmarks in each wrist frame.

    Landmark order matches the PND dashboard skeleton: wrist first, then four
    points for thumb, index, middle, ring and pinky.  Unlike the old dashboard
    approximation, every point comes from the bundled SharpA URDF FK.
    """
    left_hand, right_hand = _get_hand_models()
    left_q = _reorder_for_model(LEFT_JOINT_ORDER, left_joints, left_hand)
    right_q = _reorder_for_model(RIGHT_JOINT_ORDER, right_joints, right_hand)
    left_hand.compute_forward_kinematics(left_q)
    right_hand.compute_forward_kinematics(right_q)

    return {
        side: np.stack(
            [
                np.asarray(
                    hand.get_link_pose(hand.get_link_index(link_name))[:3, 3],
                    dtype=np.float64,
                )
                for link_name in HAND_KEYPOINT_LINKS[side]
            ]
        )
        for side, hand in (("left", left_hand), ("right", right_hand))
    }


def split_interface_hand_joints(joints: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split the frozen 44D interface hand vector into (left 22, right 22)."""
    q = np.asarray(joints, dtype=np.float64).reshape(-1)
    if q.shape[0] != NUM_JOINTS_TOTAL:
        raise ValueError(f"expected {NUM_JOINTS_TOTAL} hand joints, got {q.shape[0]}")
    return q[:NUM_JOINTS_PER_HAND].copy(), q[NUM_JOINTS_PER_HAND:].copy()
