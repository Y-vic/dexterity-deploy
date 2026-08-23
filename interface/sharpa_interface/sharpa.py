"""SharpA interface data contract.

Frozen definitions that MUST be identical across every embodiment and every
recording. If a value here changes, all training data becomes incompatible with
inference and vice versa. Treat this file as an ABI.

Any change requires:
  1. Interface schema version bump.
  2. Migration note in docs/interface_contract.md.
  3. Sign-off from every embodiment owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

# ---------------------------------------------------------------------------
# Canonical wrist pose contract
# ---------------------------------------------------------------------------
# This is the PND policy geometry shared by every embodiment. It deliberately
# contains no robot-specific mount offset, base calibration, or TCP transform.

CANONICAL_EEF_FRAME: Final[str] = "canonical"
CANONICAL_ROOT_FRAME: Final[str] = "canonical_root"
CANONICAL_ROOT_AXIS_ORDER: Final[tuple[str, str, str]] = ("x", "y", "z")
CANONICAL_ROOT_AXES: Final[tuple[str, str, str]] = (
    "left",
    "up",
    "forward",
)
CANONICAL_ROOT_AXIS_DIRECTIONS: Final[Mapping[str, str]] = MappingProxyType({
    "x": "left",
    "y": "up",
    "z": "forward",
})

# The wrist child frames and side order follow the PND SharpA model. A pose
# rotation maps the wrist-local +X/+Y/+Z basis into the canonical root basis.
CANONICAL_WRIST_SIDE_ORDER: Final[tuple[str, str]] = ("left", "right")
CANONICAL_WRIST_FRAME_BY_SIDE: Final[Mapping[str, str]] = MappingProxyType({
    "left": "left_hand_C_MC",
    "right": "right_hand_C_MC",
})
CANONICAL_WRIST_AXIS_ORDER: Final[tuple[str, str, str]] = ("x", "y", "z")
CANONICAL_WRIST_ROTATION_SEMANTICS: Final[str] = (
    "columns_are_wrist_xyz_axes_in_canonical_root"
)

# pose9 = xyz in canonical root (metres), then the first two columns of the
# canonical-root-from-wrist rotation matrix, matching PND `mat_to_rot6d`.
CANONICAL_EEF_POSE9_ORDER: Final[tuple[str, ...]] = (
    "x",
    "y",
    "z",
    "r00",
    "r10",
    "r20",
    "r01",
    "r11",
    "r21",
)
CANONICAL_ROT6D_CONVENTION: Final[str] = "first_two_matrix_columns"
CANONICAL_EEF_POSITION_UNITS: Final[str] = "meter"
CANONICAL_EEF_DEFINITION: Final[str] = "absolute"

# ---------------------------------------------------------------------------
# Joint order
# ---------------------------------------------------------------------------
# 22 DOF per hand. Total 44 DOF across both hands.
# Concatenation order: LEFT hand first, then RIGHT hand.
# Within each hand: thumb, index, middle, ring, pinky.

FINGERS: Final[tuple[str, ...]] = ("thumb", "index", "middle", "ring", "pinky")
HANDS: Final[tuple[str, ...]] = ("left", "right")

JOINT_NAMES_PER_HAND: Final[tuple[str, ...]] = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)
JOINT_NAMES: Final[tuple[str, ...]] = tuple(
    f"{hand}_{name}" for hand in HANDS for name in JOINT_NAMES_PER_HAND
)

NUM_JOINTS_PER_HAND: Final[int] = 22
NUM_JOINTS_TOTAL: Final[int] = NUM_JOINTS_PER_HAND * 2  # 44

JOINT_UNITS: Final[str] = "radian"

# ---------------------------------------------------------------------------
# Tactile
# ---------------------------------------------------------------------------
# Per-hand deformation images. Wire finger order is pinky -> thumb; this is
# intentionally independent of the hand-joint order above.

TACTILE_FINGERS: Final[tuple[str, ...]] = tuple(reversed(FINGERS))
DEFORMATION_SHAPE_PER_HAND: Final[tuple[int, int, int]] = (5, 240, 240)
DEFORMATION_DTYPE: Final[str] = "uint8"
TACTILE_SHAPE_PER_HAND = DEFORMATION_SHAPE_PER_HAND  # compatibility alias

# ---------------------------------------------------------------------------
# Wrench (fingertip force + torque)
# ---------------------------------------------------------------------------
# Per-hand tensor of shape [num_fingers, 6].
# Column order: Fx, Fy, Fz, Tx, Ty, Tz.
# Frame: fingertip-local. +Z along fingertip normal (out of pad),
# +X along finger long axis (proximal->distal), +Y = Z x X.

WRENCH_SHAPE_PER_HAND: Final[tuple[int, int]] = (5, 6)
WRENCH_FORCE_UNITS: Final[str] = "newton"
WRENCH_TORQUE_UNITS: Final[str] = "newton_meter"


# ---------------------------------------------------------------------------
# Helpers (not policy)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SharpAJointIndex:
    """Lookup: hand + finger + dof -> global index into the 44-dim vector.

    Adapters must use this rather than magic offsets. If the joint order ever
    changes, only this file changes.
    """

    hand: str
    finger: str
    dof: str

    def global_index(self) -> int:
        if self.hand not in HANDS:
            raise ValueError(f"unknown hand: {self.hand!r}")
        target = f"{self.finger}_{self.dof}".lower()
        names = tuple(name.lower() for name in JOINT_NAMES_PER_HAND)
        try:
            local_index = names.index(target)
        except ValueError as exc:
            raise ValueError(f"unknown SharpA joint: {target!r}") from exc
        return HANDS.index(self.hand) * NUM_JOINTS_PER_HAND + local_index
