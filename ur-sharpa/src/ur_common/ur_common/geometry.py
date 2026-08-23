"""Frame conventions and rotation conversions for the UR5 + SharpA pair.

Consolidated from the original UR adapters (frames, rot6d, pose9) and
`robot_control/coord_transform.py` (pose_mat / pose_vector / pose_vector_quat).

Frame chain, base to canonical wrist:

    ur_base --capture_root_from_ur_base--> capture_root
    ur_tcp  --capture_tcp_to_ur_tcp-----> capture_tcp
    capture_tcp --UR_TCP_FROM_SHARPA_HAND--> SharpA hand_C_MC
    hand --UR_ROOT_TO_PRETRAIN----------> canonical root

Interface boundary uses canonical absolute `xyz + rot6d`, shape (9,)
(docs/interface_contract.md). Rot6D is the PND COLUMN convention: the first two
columns of root-from-wrist R. The row representation and server base
corrections below are legacy adapter details and never enter the ROS contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sharpa_interface.sharpa import (
    CANONICAL_EEF_FRAME,
    CANONICAL_ROT6D_CONVENTION,
)

if CANONICAL_ROT6D_CONVENTION != "first_two_matrix_columns":
    raise RuntimeError(
        "ur_common requires the PND canonical column-Rot6D interface"
    )


SIDES = ("left", "right")

# Rotation from UR root frame to the pre-training capture frame.
UR_ROOT_TO_PRETRAIN = np.asarray(
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    dtype=np.float64,
)

# UR-SPECIFIC fixed transform: UR capture TCP -> SharpA hand_C_MC, per side.
# The 29.5 mm mount offset belongs only to this UR embodiment; it is not part
# of the shared canonical interface.
UR_TCP_FROM_SHARPA_HAND = {
    "left": np.asarray(
        ((0.0, 0.0, 1.0, 0.0295), (-1.0, 0.0, 0.0, 0.0),
         (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), dtype=np.float64),
    "right": np.asarray(
        ((0.0, 0.0, 1.0, 0.0295), (1.0, 0.0, 0.0, 0.0),
         (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), dtype=np.float64),
}

# Server-side deploy->model base corrections (baked into wire poses).
SERVER_DEPLOY_FROM_MODEL_BASE = {
    "left": (
        np.asarray((
            (-0.05063197761774063, -0.9985009431838989, -0.020789900794625282),
            (-0.9941514730453491, 0.04840133339166641, 0.09654103964567184),
            (-0.09539006650447845, 0.025556374341249466, -0.995111882686615),
        ), dtype=np.float64),
        np.asarray((0.006119123660027981, -0.004442085511982441, -0.03515041619539261), dtype=np.float64),
    ),
    "right": (
        np.asarray((
            (-0.037729986011981964, 0.9990273714065552, -0.022819984704256058),
            (0.9962800741195679, 0.035836100578308105, -0.07836932688951492),
            (-0.07747532427310944, -0.02569197118282318, -0.9966631531715393),
        ), dtype=np.float64),
        np.asarray((0.003884613746777177, 0.002192644402384758, -0.035935886204242706), dtype=np.float64),
    ),
}

# Left/right UR base relative to the common capture root, as xyz+euler(xyz).
CAPTURE_ROOT_FROM_UR_BASE_XYZ_EULER = {
    "left": (0.5735652, -0.29790113, 0.30791952, 2.24085581, -0.30165123, -1.94184712),
    "right": (-0.58120603, -0.29800993, 0.30125035, -2.21583212, -0.3416026, -1.16897751),
}


def _side(side: str) -> str:
    if side not in SIDES:
        raise ValueError("side must be 'left' or 'right'")
    return side


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array


def validate_transform(value: Any, label: str = "transform") -> np.ndarray:
    t = _finite(value, (4, 4), label)
    if not np.allclose(t[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-7):
        raise ValueError(f"{label} has an invalid homogeneous row")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(r), 1.0, atol=1.0e-6):
        raise ValueError(f"{label} rotation must have determinant +1")
    return t


# ---------------------------------------------------------------------------
# Rot6D <-> matrix <-> quaternion
# ---------------------------------------------------------------------------
def column_rot6d_to_matrix(value: Any) -> np.ndarray:
    v = _finite(value, (6,), "column Rot6D")
    a = v[:3] / np.linalg.norm(v[:3])
    b = v[3:] - float(a @ v[3:]) * a
    b = b / np.linalg.norm(b)
    return np.stack((a, b, np.cross(a, b)), axis=1)


def matrix_to_column_rot6d(value: Any) -> np.ndarray:
    r = _finite(value, (3, 3), "rotation")
    return np.concatenate((r[:, 0], r[:, 1])).astype(np.float32)


def row_rot6d_to_matrix(value: Any) -> np.ndarray:
    v = _finite(value, (6,), "row Rot6D")
    a = v[:3] / np.linalg.norm(v[:3])
    b = v[3:] - float(a @ v[3:]) * a
    b = b / np.linalg.norm(b)
    return np.stack((a, b, np.cross(a, b)), axis=0)


def matrix_to_row_rot6d(value: Any) -> np.ndarray:
    r = _finite(value, (3, 3), "rotation")
    return r[:2].reshape(6).astype(np.float32)


def matrix_to_quat(value: Any) -> np.ndarray:
    """3x3 rotation -> quaternion (x, y, z, w), matching scipy's ordering."""
    r = _finite(value, (3, 3), "rotation")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(r[i, i] - r[j, j] - r[k, k] + 1.0, 0.0)) * 2.0
        q = np.zeros(3)
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        w = (r[k, j] - r[j, k]) / s
        x, y, z = q
    quat = np.asarray((x, y, z, w), dtype=np.float64)
    # Canonical sign: keep w >= 0 so round-trips are comparable elementwise.
    if quat[3] < 0.0:
        quat = -quat
    return quat / np.linalg.norm(quat)


def quat_to_matrix(value: Any) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation."""
    q = _finite(value, (4,), "quaternion")
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float64)


def row_rot6d_to_quat(value: Any) -> np.ndarray:
    return matrix_to_quat(row_rot6d_to_matrix(value))


def quat_to_row_rot6d(value: Any) -> np.ndarray:
    return matrix_to_row_rot6d(quat_to_matrix(value))


# ---------------------------------------------------------------------------
# pose9 (xyz + rot6d), the interface boundary representation
# ---------------------------------------------------------------------------
def column_pose9_to_transform(value: Any) -> np.ndarray:
    p = _finite(value, (9,), "column pose9")
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = column_rot6d_to_matrix(p[3:])
    t[:3, 3] = p[:3]
    return t


def transform_to_column_pose9(value: Any) -> np.ndarray:
    t = validate_transform(value)
    return np.concatenate((t[:3, 3], matrix_to_column_rot6d(t[:3, :3]))).astype(np.float32)


def row_pose9_to_transform(value: Any) -> np.ndarray:
    p = _finite(value, (9,), "row pose9")
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = row_rot6d_to_matrix(p[3:])
    t[:3, 3] = p[:3]
    return t


def transform_to_row_pose9(value: Any) -> np.ndarray:
    t = validate_transform(value)
    return np.concatenate((t[:3, 3], matrix_to_row_rot6d(t[:3, :3]))).astype(np.float32)


# ---------------------------------------------------------------------------
# Euler / rotvec (RTDE TCP poses use rotvec)
# ---------------------------------------------------------------------------
def euler_xyz_to_matrix(value: Any) -> np.ndarray:
    roll, pitch, yaw = _finite(value, (3,), "XYZ Euler angles")
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    Ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    Rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return Rz @ Ry @ Rx


def xyz_euler_to_transform(value: Any) -> np.ndarray:
    p = _finite(value, (6,), "XYZ+Euler pose")
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = euler_xyz_to_matrix(p[3:])
    t[:3, 3] = p[:3]
    return t


def rotvec_to_matrix(value: Any) -> np.ndarray:
    rv = _finite(value, (3,), "rotation vector")
    angle = float(np.linalg.norm(rv))
    if angle < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = rv / angle
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_rotvec(value: Any) -> np.ndarray:
    """3x3 rotation -> rotation vector (axis * angle).

    Computed via quaternion rather than the arccos/sin formula. The direct form
    divides by sin(angle), so its error grows as eps/sin(angle) and blows up as
    the angle approaches pi.

    This is a FIX relative to the original wire_geometry.py, which guarded the
    near-pi case only at `pi - angle < 1e-6` while the standard branch was
    already badly conditioned well before that.

    Measured, not theoretical. Over 400 random reachable UR5 poses, encoding
    `arm_fk(q)`'s rotation to a rotvec and decoding it back:

        this version (quaternion): worst 5.4e-08 rad
        original (arccos/sin):     worst 3.4e-03 rad  (0.19 degrees)

    It matters because this is on the live deploy path -- transform_to_rtde_pose
    feeds every UR TCP command -- and UR5 FK output lands near angle = pi
    routinely for this cell (observed pi - angle down to 2.6e-3), so it is
    ordinary operation rather than a corner case. It also showed up directly as
    a 3.4e-03 rad failure in the FK/IK round-trip test.

    Note the reference's wire_pose_to_rtde_pose path did NOT show this, because
    there the rotation is reconstructed from pose9 and is better conditioned;
    the defect is specific to feeding a raw FK rotation matrix in.

    matrix_to_quat uses the trace/dominant-diagonal branch selection, which is
    stable across the whole range, and canonicalises w >= 0 so the returned
    angle lands in [0, pi].
    """
    r = _finite(value, (3, 3), "rotation")
    quat = matrix_to_quat(r)
    vec, w = quat[:3], float(quat[3])
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-15:
        return np.zeros(3, dtype=np.float64)
    # 2*atan2(|v|, w) is stable at both ends, unlike arccos of the trace.
    return vec / norm * (2.0 * np.arctan2(norm, w))


def rtde_pose_to_transform(value: Any) -> np.ndarray:
    """[x,y,z, rx,ry,rz(rotvec)] -> 4x4. This is the UR RTDE TCP convention."""
    p = _finite(value, (6,), "RTDE TCP pose")
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotvec_to_matrix(p[3:])
    t[:3, 3] = p[:3]
    return t


def transform_to_rtde_pose(value: Any) -> np.ndarray:
    t = validate_transform(value)
    return np.concatenate((t[:3, 3], matrix_to_rotvec(t[:3, :3])))


# ---- coord_transform.py equivalents (euler/quat pose vectors) ----
def pose_vector(pose_matrix: Any) -> np.ndarray:
    """4x4 -> [x,y,z, euler_xyz] (6,). Port of coord_transform.pose_vector."""
    t = _finite(pose_matrix, (4, 4), "pose matrix")
    out = np.zeros(6, dtype=np.float64)
    out[:3] = t[:3, 3]
    out[3:] = matrix_to_euler_xyz(t[:3, :3])
    return out


def pose_vector_quat(pose_matrix: Any) -> np.ndarray:
    """4x4 -> [x,y,z, qx,qy,qz,qw] (7,). Port of coord_transform.pose_vector_quat."""
    t = _finite(pose_matrix, (4, 4), "pose matrix")
    out = np.zeros(7, dtype=np.float64)
    out[:3] = t[:3, 3]
    out[3:] = matrix_to_quat(t[:3, :3])
    return out


def pose_mat(vector: Any) -> np.ndarray:
    """[x,y,z, euler_xyz] (6,) -> 4x4. Port of coord_transform.pose_mat."""
    return xyz_euler_to_transform(vector)


def matrix_to_euler_xyz(value: Any) -> np.ndarray:
    """3x3 -> intrinsic-free XYZ Euler, inverse of euler_xyz_to_matrix."""
    r = _finite(value, (3, 3), "rotation")
    sy = -r[2, 0]
    if abs(sy) < 1.0 - 1.0e-9:
        pitch = float(np.arcsin(np.clip(sy, -1.0, 1.0)))
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    else:  # gimbal lock: pitch = +-pi/2, roll and yaw are degenerate
        pitch = float(np.pi / 2.0 if sy > 0 else -np.pi / 2.0)
        roll = float(np.arctan2(-r[1, 2], r[1, 1]))
        yaw = 0.0
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def _server_base_transform(side: str) -> np.ndarray:
    rotation, translation = SERVER_DEPLOY_FROM_MODEL_BASE[_side(side)]
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


class UrSharpAWireGeometry:
    """Invertible bridge between the real UR pair and policy-v3 wire poses.

    Every method has an exact inverse; the round-trip tests rely on that.
    """

    def __init__(self) -> None:
        self.capture_root_from_ur_base = {
            side: xyz_euler_to_transform(CAPTURE_ROOT_FROM_UR_BASE_XYZ_EULER[side])
            for side in SIDES
        }
        wr = xyz_euler_to_transform((0.0, 0.0, 0.0, -np.pi / 4.0, 0.0, 0.0))
        wr = wr @ xyz_euler_to_transform((0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0))
        self.capture_tcp_to_ur_tcp = {side: wr.copy() for side in SIDES}

    def ur_base_tcp_to_capture_tcp(self, value: Any, side: str) -> np.ndarray:
        ur_base_from_tcp = validate_transform(value)
        root_from_capture = self.capture_root_from_ur_base[_side(side)]
        ur_tcp_from_capture_tcp = self.capture_tcp_to_ur_tcp[side]
        return (
            np.linalg.inv(root_from_capture)
            @ ur_base_from_tcp
            @ np.linalg.inv(ur_tcp_from_capture_tcp)
        )

    def capture_tcp_to_ur_base_tcp(self, value: Any, side: str) -> np.ndarray:
        capture_from_tcp = validate_transform(value)
        return (
            self.capture_root_from_ur_base[_side(side)]
            @ capture_from_tcp
            @ self.capture_tcp_to_ur_tcp[side]
        )

    def capture_tcp_to_model_pose(self, value: Any, side: str) -> np.ndarray:
        capture_from_tcp = validate_transform(value)
        capture_from_hand = capture_from_tcp @ UR_TCP_FROM_SHARPA_HAND[_side(side)]
        model_from_hand = np.eye(4, dtype=np.float64)
        model_from_hand[:3, :3] = UR_ROOT_TO_PRETRAIN @ capture_from_hand[:3, :3]
        model_from_hand[:3, 3] = UR_ROOT_TO_PRETRAIN @ capture_from_hand[:3, 3]
        return transform_to_row_pose9(model_from_hand)

    def model_pose_to_capture_tcp(self, value: Any, side: str) -> np.ndarray:
        model_from_hand = row_pose9_to_transform(value)
        capture_from_hand = np.eye(4, dtype=np.float64)
        capture_from_hand[:3, :3] = UR_ROOT_TO_PRETRAIN.T @ model_from_hand[:3, :3]
        capture_from_hand[:3, 3] = UR_ROOT_TO_PRETRAIN.T @ model_from_hand[:3, 3]
        return capture_from_hand @ np.linalg.inv(UR_TCP_FROM_SHARPA_HAND[_side(side)])

    def capture_tcp_to_canonical_pose(self, value: Any, side: str) -> np.ndarray:
        """UR capture TCP -> canonical absolute column-pose9.

        This is the ROS/interface representation. The 29.5 mm transform above
        is applied here as UR-specific calibration before encoding the shared
        PND wrist frame.
        """
        capture_from_tcp = validate_transform(value)
        capture_from_wrist = capture_from_tcp @ UR_TCP_FROM_SHARPA_HAND[_side(side)]
        canonical_from_wrist = np.eye(4, dtype=np.float64)
        canonical_from_wrist[:3, :3] = UR_ROOT_TO_PRETRAIN @ capture_from_wrist[:3, :3]
        canonical_from_wrist[:3, 3] = UR_ROOT_TO_PRETRAIN @ capture_from_wrist[:3, 3]
        return transform_to_column_pose9(canonical_from_wrist)

    def canonical_pose_to_capture_tcp(self, value: Any, side: str) -> np.ndarray:
        """Canonical absolute column-pose9 -> UR capture TCP."""
        canonical_from_wrist = column_pose9_to_transform(value)
        capture_from_wrist = np.eye(4, dtype=np.float64)
        capture_from_wrist[:3, :3] = UR_ROOT_TO_PRETRAIN.T @ canonical_from_wrist[:3, :3]
        capture_from_wrist[:3, 3] = UR_ROOT_TO_PRETRAIN.T @ canonical_from_wrist[:3, 3]
        return capture_from_wrist @ np.linalg.inv(UR_TCP_FROM_SHARPA_HAND[_side(side)])

    def canonical_pose_to_legacy_server_pose(
        self, value: Any, side: str
    ) -> np.ndarray:
        """Canonical column-pose9 -> current server deploy-physical pose.

        The remote SharpA62 server still applies its historical physical-to-
        posttrain adapter. This method is only for the local transport boundary;
        ROS messages must never use the returned frame.
        """
        canonical_from_wrist = column_pose9_to_transform(value)
        model_pose = transform_to_row_pose9(canonical_from_wrist)
        return self.model_pose_to_wire_pose(model_pose, side)

    def legacy_server_pose_to_canonical_pose(
        self, value: Any, side: str
    ) -> np.ndarray:
        """Current server deploy-physical pose -> canonical column-pose9."""
        model_pose = self.wire_pose_to_model_pose(value, side)
        return transform_to_column_pose9(row_pose9_to_transform(model_pose))

    def wire_pose_to_model_pose(self, value: Any, side: str) -> np.ndarray:
        deploy_from_wrist = column_pose9_to_transform(value)
        baked = deploy_from_wrist @ _server_base_transform(_side(side))
        model_from_hand = np.eye(4, dtype=np.float64)
        model_from_hand[:3, :3] = UR_ROOT_TO_PRETRAIN @ baked[:3, :3]
        model_from_hand[:3, 3] = UR_ROOT_TO_PRETRAIN @ baked[:3, 3]
        return transform_to_row_pose9(model_from_hand)

    def model_pose_to_wire_pose(self, value: Any, side: str) -> np.ndarray:
        model_from_hand = row_pose9_to_transform(value)
        baked = np.eye(4, dtype=np.float64)
        baked[:3, :3] = UR_ROOT_TO_PRETRAIN.T @ model_from_hand[:3, :3]
        baked[:3, 3] = UR_ROOT_TO_PRETRAIN.T @ model_from_hand[:3, 3]
        deploy_from_wrist = baked @ np.linalg.inv(_server_base_transform(_side(side)))
        return transform_to_column_pose9(deploy_from_wrist)

    def capture_tcp_to_wire_pose(self, value: Any, side: str) -> np.ndarray:
        return self.model_pose_to_wire_pose(
            self.capture_tcp_to_model_pose(value, side), side
        )

    def wire_pose_to_capture_tcp(self, value: Any, side: str) -> np.ndarray:
        return self.model_pose_to_capture_tcp(
            self.wire_pose_to_model_pose(value, side), side
        )

    def rtde_pose_to_wire_pose(self, value: Any, side: str) -> np.ndarray:
        capture_tcp = self.ur_base_tcp_to_capture_tcp(rtde_pose_to_transform(value), side)
        return self.capture_tcp_to_wire_pose(capture_tcp, side)

    def wire_pose_to_rtde_pose(self, value: Any, side: str) -> np.ndarray:
        capture_tcp = self.wire_pose_to_capture_tcp(value, side)
        return transform_to_rtde_pose(self.capture_tcp_to_ur_base_tcp(capture_tcp, side))
