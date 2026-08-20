"""UR-side wrist geometry adapter for the shared interface.

The UR base convention and RTDE conversion remain local to this module.
"""

from __future__ import annotations

import numpy as np

from .wire_geometry import (
    UrSharpAWireGeometry,
    column_pose9_to_transform,
    transform_to_column_pose9,
)

# Module-level geometry instance (stateless after construction).
_geo = UrSharpAWireGeometry()


def ur_rtde_to_interface_wrists(
    left_rtde_pose: np.ndarray,   # (6,) [x,y,z,rx,ry,rz] in UR base frame
    right_rtde_pose: np.ndarray,  # (6,)
) -> np.ndarray:
    """Return the UR absolute bimanual wrist representation.

    Layout: [Lpos(3), Lquat_wxyz(4), Rpos(3), Rquat_wxyz(4)].
    The geometry (capture frame, SharpA hand offset, server base correction)
    matches UrSharpAWireGeometry exactly.
    """
    left_wire = _geo.rtde_pose_to_wire_pose(left_rtde_pose, "left")
    right_wire = _geo.rtde_pose_to_wire_pose(right_rtde_pose, "right")
    return _wire_poses_to_interface(left_wire, right_wire)


def interface_wrists_to_ur_rtde(
    interface_wrists: np.ndarray,  # (14,)
) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_rtde_pose, right_rtde_pose), each shape (6,)."""
    left_wire = _interface_to_wire_pose(interface_wrists[:7])
    right_wire = _interface_to_wire_pose(interface_wrists[7:])
    left_rtde = _geo.wire_pose_to_rtde_pose(left_wire, "left")
    right_rtde = _geo.wire_pose_to_rtde_pose(right_wire, "right")
    return left_rtde, right_rtde


# ---------------------------------------------------------------------------
# Internal helpers: wire pose (column pose9) and UR-local bimanual pose.
# ---------------------------------------------------------------------------
def _wire_poses_to_interface(
    left_wire: np.ndarray,   # (9,) column pose9 float32
    right_wire: np.ndarray,  # (9,)
) -> np.ndarray:
    out = np.empty(14, dtype=np.float64)
    _fill_wrist_slot(out, 0, left_wire)
    _fill_wrist_slot(out, 7, right_wire)
    return out


def _fill_wrist_slot(out: np.ndarray, offset: int, wire: np.ndarray) -> None:
    t = column_pose9_to_transform(wire)
    out[offset:offset + 3] = t[:3, 3]
    out[offset + 3:offset + 7] = _rotation_matrix_to_quat_wxyz(t[:3, :3])


def _interface_to_wire_pose(wrist7: np.ndarray) -> np.ndarray:
    pos = wrist7[:3]
    quat_wxyz = wrist7[3:7]
    r = _quat_wxyz_to_rotation_matrix(quat_wxyz)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = r
    t[:3, 3] = pos
    return transform_to_column_pose9(t)


def _rotation_matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → quaternion [w, x, y, z]."""
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        s = 0.5 / float(np.sqrt(trace + 1.0))
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]))
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * float(np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]))
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
