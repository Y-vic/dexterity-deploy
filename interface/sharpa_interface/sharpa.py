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
from typing import Final

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
