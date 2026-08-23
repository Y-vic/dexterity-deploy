"""The UR home pose, as Cartesian matrices plus the conversion to joints.

`left_home_mat` / `right_home_mat` are copied verbatim from
teleop_mocap_ur5_sharpa_hand_zed.py:1122-1133. They are the poses the monolith
handed to `UR5_dual.set_target_pose` immediately after connecting, i.e. the pose
the arms are driven to before teleoperation starts.

FRAME: these matrices are in the CAPTURE frame, not the UR base frame. In the
monolith they went through `UR5_dual.move_to_pose`, which computes
`left_transform @ target_pose @ left_rot` before IK -- and those two constants
(ur5.py:484-485 and 501-502, transform_type='old') are exactly what
`ur_common.geometry.UrSharpAWireGeometry` was built from
(CAPTURE_ROOT_FROM_UR_BASE_XYZ_EULER and capture_tcp_to_ur_tcp). So the port is
`geometry.capture_tcp_to_ur_base_tcp(home_mat, side)` and then `arm_ik`, which
reproduces the monolith's arithmetic through the shared backend rather than a
second copy of it. Hard constraint #2: one kinematics backend, no parallel copy.

`allow_large=True` on the IK call is deliberate and matches the monolith's
homing mode. Homing starts from an ARBITRARY pose, so the excursion to home is
legitimately large; the wrist-flip filter is a per-frame following filter and
would hold here. Joint limit filtering still applies. Teleop following uses
allow_large=False.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Verbatim from teleop_mocap_ur5_sharpa_hand_zed.py:1122-1127.
LEFT_HOME_MAT = np.asarray([
    [0.99464192, -0.07281377, 0.07338657, 0.16677179],
    [0.07856162, 0.99379449, -0.07874395, 0.16679316],
    [-0.06719752, 0.08408743, 0.99419003, 0.36345815],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# Verbatim from teleop_mocap_ur5_sharpa_hand_zed.py:1128-1133.
RIGHT_HOME_MAT = np.asarray([
    [0.97746886, 0.19248338, 0.08662993, 0.15662127],
    [-0.2043652, 0.96569153, 0.16023355, -0.21037435],
    [-0.05281551, -0.17432745, 0.98327031, 0.38592111],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

HOME_MAT = {"left": LEFT_HOME_MAT, "right": RIGHT_HOME_MAT}

# IK seed used when no measured joint state is available yet. Elbow-up, a
# posture the UR5 can reach both home poses from without a shoulder flip.
DEFAULT_IK_SEED = np.asarray([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
                             dtype=np.float64)


class HomePoseError(RuntimeError):
    """The home pose could not be converted to joint angles."""


def home_joints(
    kinematics: Any,
    side: str,
    seed_joints: Any = None,
    home_mat: Any = None,
    residual_tol_m: float = 5e-3,
) -> np.ndarray:
    """Capture-frame home matrix -> UR5 joint angles, via ur_common only.

    :param kinematics: a `ur_common.kinematics.UrCommonKinematics`
    :param side: "left" or "right"
    :param seed_joints: measured joints if known, else DEFAULT_IK_SEED
    :param home_mat: override the 4x4 capture-frame home pose
    :param residual_tol_m: FK(IK(target)) position residual that still counts as
        solved. `FastIKSolver.solve` HOLDS instead of raising when it cannot
        solve, returning the seed -- so a silent failure looks like a valid
        answer. Round-tripping through FK is the only way to tell the two apart,
        and homing must not report success on a hold.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    mat = HOME_MAT[side] if home_mat is None else np.asarray(home_mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError("home_mat must be a 4x4 transform")
    seed = (DEFAULT_IK_SEED if seed_joints is None
            else np.asarray(seed_joints, dtype=np.float64).reshape(-1))
    if seed.shape[0] != 6:
        raise ValueError("seed_joints must have 6 elements")

    ur_base_target = kinematics.geometry.capture_tcp_to_ur_base_tcp(mat, side)
    joints = np.asarray(
        kinematics.arm_ik(ur_base_target, seed, side=side, allow_large=True),
        dtype=np.float64,
    ).reshape(-1)
    if joints.shape[0] != 6 or not np.all(np.isfinite(joints)):
        raise HomePoseError(f"{side} home IK returned a non-finite solution")

    achieved = np.asarray(kinematics.arm_fk(joints), dtype=np.float64)
    residual = float(np.linalg.norm(achieved[:3, 3] - ur_base_target[:3, 3]))
    if residual > residual_tol_m:
        raise HomePoseError(
            f"{side} home IK did not converge: FK residual {residual * 1000.0:.1f}mm "
            f"> {residual_tol_m * 1000.0:.1f}mm (solver held instead of solving)"
        )
    return joints


def both_home_joints(kinematics: Any, seed: dict | None = None) -> dict:
    """{'left': (6,), 'right': (6,)} home joints. Raises HomePoseError on failure."""
    seed = seed or {}
    return {
        side: home_joints(kinematics, side, seed_joints=seed.get(side))
        for side in ("left", "right")
    }
