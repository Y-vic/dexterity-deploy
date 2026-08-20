"""Coordinate conversions for the local dual-UR/SharpA workstation.

The UR post-training captures, the policy-v3 wire, and the model do not use
the same wrist child frame or Rot6D convention.  This module keeps every
conversion explicit and invertible.  It has no ROS or robot dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SIDES = ("left", "right")
UR_ROOT_TO_PRETRAIN = np.asarray(
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    dtype=np.float64,
)

UR_TCP_FROM_SHARPA_HAND = {
    "left": np.asarray(
        (
            (0.0, 0.0, 1.0, 0.0295),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    ),
    "right": np.asarray(
        (
            (0.0, 0.0, 1.0, 0.0295),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    ),
}

# These are the current server-side SharpA62PolicyGeometry deploy-to-model
# corrections.  The UR capture converter uses UR_TCP_FROM_SHARPA_HAND above,
# so the workstation must bridge between the two physical conventions.
SERVER_DEPLOY_FROM_MODEL_BASE = {
    "left": (
        np.asarray(
            (
                (-0.05063197761774063, -0.9985009431838989, -0.020789900794625282),
                (-0.9941514730453491, 0.04840133339166641, 0.09654103964567184),
                (-0.09539006650447845, 0.025556374341249466, -0.995111882686615),
            ),
            dtype=np.float64,
        ),
        np.asarray(
            (0.006119123660027981, -0.004442085511982441, -0.03515041619539261),
            dtype=np.float64,
        ),
    ),
    "right": (
        np.asarray(
            (
                (-0.037729986011981964, 0.9990273714065552, -0.022819984704256058),
                (0.9962800741195679, 0.035836100578308105, -0.07836932688951492),
                (-0.07747532427310944, -0.02569197118282318, -0.9966631531715393),
            ),
            dtype=np.float64,
        ),
        np.asarray(
            (0.003884613746777177, 0.002192644402384758, -0.035935886204242706),
            dtype=np.float64,
        ),
    ),
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
    transform = _finite(value, (4, 4), label)
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-7):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError(f"{label} rotation must have determinant +1")
    return transform


def column_rot6d_to_matrix(value: Any) -> np.ndarray:
    vector = _finite(value, (6,), "column Rot6D")
    first = vector[:3]
    second = vector[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1.0e-8:
        raise ValueError("column Rot6D first axis is degenerate")
    first = first / first_norm
    second = second - float(first @ second) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1.0e-8:
        raise ValueError("column Rot6D second axis is degenerate")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=1)


def matrix_to_column_rot6d(value: Any) -> np.ndarray:
    rotation = _finite(value, (3, 3), "rotation")
    return np.concatenate((rotation[:, 0], rotation[:, 1])).astype(np.float32)


def row_rot6d_to_matrix(value: Any) -> np.ndarray:
    vector = _finite(value, (6,), "row Rot6D")
    first = vector[:3]
    second = vector[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1.0e-8:
        raise ValueError("row Rot6D first axis is degenerate")
    first = first / first_norm
    second = second - float(first @ second) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1.0e-8:
        raise ValueError("row Rot6D second axis is degenerate")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=0)


def matrix_to_row_rot6d(value: Any) -> np.ndarray:
    rotation = _finite(value, (3, 3), "rotation")
    return rotation[:2].reshape(6).astype(np.float32)


def column_pose9_to_transform(value: Any) -> np.ndarray:
    pose = _finite(value, (9,), "column pose9")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = column_rot6d_to_matrix(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_column_pose9(value: Any) -> np.ndarray:
    transform = validate_transform(value)
    return np.concatenate(
        (transform[:3, 3], matrix_to_column_rot6d(transform[:3, :3]))
    ).astype(np.float32)


def row_pose9_to_transform(value: Any) -> np.ndarray:
    pose = _finite(value, (9,), "row pose9")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = row_rot6d_to_matrix(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_row_pose9(value: Any) -> np.ndarray:
    transform = validate_transform(value)
    return np.concatenate(
        (transform[:3, 3], matrix_to_row_rot6d(transform[:3, :3]))
    ).astype(np.float32)


def euler_xyz_to_matrix(value: Any) -> np.ndarray:
    roll, pitch, yaw = _finite(value, (3,), "XYZ Euler angles")
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotation_x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rotation_y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rotation_z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rotation_z @ rotation_y @ rotation_x


def xyz_euler_to_transform(value: Any) -> np.ndarray:
    pose = _finite(value, (6,), "XYZ+Euler pose")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_xyz_to_matrix(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def rotvec_to_matrix(value: Any) -> np.ndarray:
    rotvec = _finite(value, (3,), "rotation vector")
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_rotvec(value: Any) -> np.ndarray:
    rotation = _finite(value, (3, 3), "rotation")
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-9:
        return 0.5 * np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        )
    if np.pi - angle < 1.0e-6:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
        dominant = int(np.argmax(axis))
        if axis[dominant] < 1.0e-8:
            raise ValueError("cannot recover rotation vector near pi")
        for index in range(3):
            if index != dominant:
                axis[index] = (
                    rotation[dominant, index] + rotation[index, dominant]
                ) / (4.0 * axis[dominant])
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    axis = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    ) / (2.0 * np.sin(angle))
    return axis * angle


def rtde_pose_to_transform(value: Any) -> np.ndarray:
    pose = _finite(value, (6,), "RTDE TCP pose")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotvec_to_matrix(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_rtde_pose(value: Any) -> np.ndarray:
    transform = validate_transform(value)
    return np.concatenate(
        (transform[:3, 3], matrix_to_rotvec(transform[:3, :3]))
    )


def _server_base_transform(side: str) -> np.ndarray:
    rotation, translation = SERVER_DEPLOY_FROM_MODEL_BASE[_side(side)]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


class UrSharpAWireGeometry:
    """Invertible bridge between the real UR pair and policy-v3 wire poses."""

    def __init__(self) -> None:
        self.capture_root_from_ur_base = {
            "left": xyz_euler_to_transform(
                (0.5735652, -0.29790113, 0.30791952, 2.24085581, -0.30165123, -1.94184712)
            ),
            "right": xyz_euler_to_transform(
                (-0.58120603, -0.29800993, 0.30125035, -2.21583212, -0.3416026, -1.16897751)
            ),
        }
        wrist_rotation = xyz_euler_to_transform((0.0, 0.0, 0.0, -np.pi / 4.0, 0.0, 0.0))
        wrist_rotation = wrist_rotation @ xyz_euler_to_transform(
            (0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0)
        )
        self.capture_tcp_to_ur_tcp = {side: wrist_rotation.copy() for side in SIDES}

    def ur_base_tcp_to_capture_tcp(self, value: Any, side: str) -> np.ndarray:
        ur_base_from_tcp = validate_transform(value, "UR base TCP transform")
        root_from_capture = self.capture_root_from_ur_base[_side(side)]
        ur_tcp_from_capture_tcp = self.capture_tcp_to_ur_tcp[side]
        return (
            np.linalg.inv(root_from_capture)
            @ ur_base_from_tcp
            @ np.linalg.inv(ur_tcp_from_capture_tcp)
        )

    def capture_tcp_to_ur_base_tcp(self, value: Any, side: str) -> np.ndarray:
        capture_from_tcp = validate_transform(value, "capture TCP transform")
        return (
            self.capture_root_from_ur_base[_side(side)]
            @ capture_from_tcp
            @ self.capture_tcp_to_ur_tcp[side]
        )

    def capture_tcp_to_model_pose(self, value: Any, side: str) -> np.ndarray:
        capture_from_tcp = validate_transform(value, "capture TCP transform")
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


def relative_wire_pose_to_absolute(value: Any, reference: Any) -> np.ndarray:
    relative = np.asarray(value)
    if relative.dtype != np.float32 or relative.ndim not in (1, 2) or relative.shape[-1] != 9:
        raise ValueError("relative wire pose must be float32 with shape (9,) or (T,9)")
    reference_pose = np.asarray(reference)
    if reference_pose.dtype != np.float32 or reference_pose.shape != (9,):
        raise ValueError("reference wire pose must be float32 with shape (9,)")
    reference_transform = column_pose9_to_transform(reference_pose)
    rows = relative.reshape(-1, 9)
    output = np.empty_like(rows)
    for index, row in enumerate(rows):
        output[index] = transform_to_column_pose9(
            reference_transform @ column_pose9_to_transform(row)
        )
    return output.reshape(relative.shape)
