"""UR adapter for absolute interface EEF targets.

The shared interface only transports absolute EEF values. UR base-frame
conversion and inverse kinematics stay in the production UR backend.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sharpa_policy_v3_client.hardware_geometry import UrSharpAWireGeometry


class UrIkBackend(Protocol):
    def inverse_kinematics(
        self, side: str, target_rtde_pose: np.ndarray, near_joint: np.ndarray
    ) -> np.ndarray: ...


def interface_wrists_to_ur_joints(
    wrists: np.ndarray,
    current_joints: np.ndarray,
    backend: UrIkBackend,
    *,
    geometry: UrSharpAWireGeometry | None = None,
) -> np.ndarray:
    """Convert two absolute pose9 EEF targets to a finite UR joint[12]."""

    targets = np.asarray(wrists, dtype=np.float64)
    if targets.shape == (18,):
        targets = targets.reshape(2, 9)
    seeds = np.asarray(current_joints, dtype=np.float64)
    if targets.shape != (2, 9) or seeds.shape != (12,):
        raise ValueError("wrists must be (2, 9) and current_joints must be (12,)")
    if not np.isfinite(targets).all() or not np.isfinite(seeds).all():
        raise ValueError("UR IK inputs must be finite")

    wire_geometry = geometry or UrSharpAWireGeometry()
    solved = []
    for index, side in enumerate(("left", "right")):
        target_rtde = wire_geometry.wire_pose_to_rtde_pose(targets[index], side)
        joint = np.asarray(
            backend.inverse_kinematics(
                side, target_rtde, seeds[index * 6 : (index + 1) * 6]
            ),
            dtype=np.float64,
        )
        if joint.shape != (6,) or not np.isfinite(joint).all():
            raise ValueError(f"{side} UR IK must return finite joint[6]")
        solved.append(joint)
    return np.concatenate(solved)
