"""Measured per-arm transforms and joint order at the DFM deployment boundary.

Model poses are physical ego optical hand-root poses in metres/column Rot6D.
Wire poses retain the existing canonical wrist ABI. Each side uses
model_pose = model_from_canonical @ wire_pose @ canonical_wrist_from_model_hand.
This includes the training rig/base and terminal hand-axis conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _transform(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError(f"{name} must be an SE(3) transform")
    return matrix.copy()


def _pose(values):
    first = values[..., 3:6]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate first column of wrist Rot6D")
    first = first / norm
    second = values[..., 6:9]
    second = second - (first * second).sum(axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate second column of wrist Rot6D")
    second = second / norm
    poses = np.broadcast_to(np.eye(4), (*values.shape[:-1], 4, 4)).copy()
    poses[..., :3, :3] = np.stack((first, second, np.cross(first, second)), axis=-1)
    poses[..., :3, 3] = values[..., :3]
    return poses


def _values(poses):
    return np.concatenate((poses[..., :3, 3], poses[..., :3, 0], poses[..., :3, 1]), axis=-1)


class DFMCalibration:
    def __init__(self, payload):
        if payload.get("schema") != "ur_sharpa.dfm_calibration.v1":
            raise ValueError("Expected ur_sharpa.dfm_calibration.v1")
        self.calibration_id = payload.get("calibration_id")
        if not isinstance(self.calibration_id, str) or not self.calibration_id:
            raise ValueError("Calibration requires a nonempty calibration_id")
        if not isinstance(payload.get("source"), dict) or not payload["source"]:
            raise ValueError("Calibration requires recorded source provenance")
        self.source = payload["source"]
        indices = payload.get("model_joint_to_wire_indices")
        if (
            not isinstance(indices, list)
            or any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(22))
        ):
            raise ValueError("model_joint_to_wire_indices must permute 0..21")
        self.indices = np.asarray(indices)
        sides = payload.get("sides")
        if not isinstance(sides, dict) or set(sides) != {"left", "right"}:
            raise ValueError("Calibration requires both arm transforms")
        self.transforms = {}
        for side, values in sides.items():
            left = _transform(values["model_from_canonical"], f"{side}.model_from_canonical")
            right = _transform(values["canonical_wrist_from_model_hand"], f"{side}.canonical_wrist_from_model_hand")
            self.transforms[side] = left, right, np.linalg.inv(left), np.linalg.inv(right)

    @classmethod
    def load(cls, path):
        try:
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        except OSError as exc:
            raise ValueError(f"Missing DFM calibration: {path}") from exc

    def convert(self, values, *, to_model):
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1:] != (62,) or not np.isfinite(values).all():
            raise ValueError("Expected finite physical SharpA62 values")
        result = values.copy()
        for index, side in enumerate(("left", "right")):
            left, right, left_inverse, right_inverse = self.transforms[side]
            start = index * 9
            poses = _pose(values[..., start : start + 9])
            poses = left @ poses @ right if to_model else left_inverse @ poses @ right_inverse
            result[..., start : start + 9] = _values(poses)
            start = 18 + index * 22
            if to_model:
                result[..., start : start + 22] = values[..., start + self.indices]
            else:
                result[..., start + self.indices] = values[..., start : start + 22]
        return result
