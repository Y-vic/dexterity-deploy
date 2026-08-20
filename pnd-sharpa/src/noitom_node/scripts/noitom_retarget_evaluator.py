#!/usr/bin/env python3
"""Evaluate vendor and GMR Noitom retarget outputs with Adam Pro FK metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


DEFAULT_GMR_PYTHON = "/home/pnd-humanoid/Deploy/.venv-gmr/bin/python"


def _maybe_reexec_into_gmr_python() -> None:
    requested = os.environ.get("NOITOM_GMR_PYTHON") or DEFAULT_GMR_PYTHON
    python_path = pathlib.Path(requested)
    if not python_path.is_file():
        return
    venv_prefix = python_path.parent.parent.resolve()
    if pathlib.Path(sys.prefix).resolve() == venv_prefix:
        return
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["NOITOM_GMR_REEXEC"] = "1"
    os.execve(str(python_path), [str(python_path), *sys.argv], env)


_maybe_reexec_into_gmr_python()

import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import tf2_ros  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.time import Time  # noqa: E402
from rclpy.utilities import remove_ros_args  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402


DEFAULT_GMR_REPO = "/home/pnd-humanoid/Deploy/GMR-master"
DEFAULT_VENDOR_TOPIC = "/_noitom/retargeted_joint_states_raw"
DEFAULT_GMR_TOPIC = "/_probe_gmr_retargeted_joint_states_raw"

NOITOM_BONES = (
    "Hips",
    "Spine2",
    "Head",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
)

ADAM_JOINT_NAME_MAP = (
    ("dof_pos/waistRoll", "waistRoll"),
    ("dof_pos/waistPitch", "waistPitch"),
    ("dof_pos/waistYaw", "waistYaw"),
    ("dof_pos/neckYaw", "neckYaw"),
    ("dof_pos/neckPitch", "neckPitch"),
    ("dof_pos/shoulderPitch_Left", "shoulderPitch_Left"),
    ("dof_pos/shoulderRoll_Left", "shoulderRoll_Left"),
    ("dof_pos/shoulderYaw_Left", "shoulderYaw_Left"),
    ("dof_pos/elbow_Left", "elbow_Left"),
    ("dof_pos/wristYaw_Left", "wristYaw_Left"),
    ("dof_pos/wristPitch_Left", "wristPitch_Left"),
    ("dof_pos/wristRoll_Left", "wristRoll_Left"),
    ("dof_pos/shoulderPitch_Right", "shoulderPitch_Right"),
    ("dof_pos/shoulderRoll_Right", "shoulderRoll_Right"),
    ("dof_pos/shoulderYaw_Right", "shoulderYaw_Right"),
    ("dof_pos/elbow_Right", "elbow_Right"),
    ("dof_pos/wristYaw_Right", "wristYaw_Right"),
    ("dof_pos/wristPitch_Right", "wristPitch_Right"),
    ("dof_pos/wristRoll_Right", "wristRoll_Right"),
)
ADAM_COMMAND_JOINTS_19 = [ros_name for ros_name, _ in ADAM_JOINT_NAME_MAP]

INTEREST_FRAMES = (
    "torso",
    "neckPitch_link",
    "shoulderYawLeft",
    "elbowLeft",
    "wristRollLeft",
    "shoulderYawRight",
    "elbowRight",
    "wristRollRight",
)

SEGMENTS = {
    "torso": ("pelvis", "torso"),
    "head": ("torso", "neckPitch_link"),
    "left_upper_arm": ("shoulderYawLeft", "elbowLeft"),
    "left_forearm": ("elbowLeft", "wristRollLeft"),
    "right_upper_arm": ("shoulderYawRight", "elbowRight"),
    "right_forearm": ("elbowRight", "wristRollRight"),
}

TWIST_SEGMENTS = {
    "left_wrist": ("elbowLeft", "wristRollLeft"),
    "right_wrist": ("elbowRight", "wristRollRight"),
}


@dataclass
class Pose:
    pos: np.ndarray
    rot: np.ndarray


@dataclass
class Sample:
    t: float
    q: np.ndarray
    metrics: dict[str, dict[str, float]]


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_from_matrix(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return quat_normalize(
            np.array(
                [
                    0.25 * s,
                    (rot[2, 1] - rot[1, 2]) / s,
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[1, 0] - rot[0, 1]) / s,
                ],
                dtype=np.float64,
            )
        )

    axis = int(np.argmax(np.diag(rot)))
    if axis == 0:
        s = math.sqrt(max(1e-12, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])) * 2.0
        quat = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                0.25 * s,
                (rot[0, 1] + rot[1, 0]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif axis == 1:
        s = math.sqrt(max(1e-12, 1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])) * 2.0
        quat = np.array(
            [
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[0, 1] + rot[1, 0]) / s,
                0.25 * s,
                (rot[1, 2] + rot[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(max(1e-12, 1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])) * 2.0
        quat = np.array(
            [
                (rot[1, 0] - rot[0, 1]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
                (rot[1, 2] + rot[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return quat_normalize(quat)


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    dot = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def rotation_error_deg(target: np.ndarray, actual: np.ndarray) -> float:
    err = target.T @ actual
    cos_angle = (float(np.trace(err)) - 1.0) * 0.5
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def twist_error_deg(axis: np.ndarray, target_rot: np.ndarray, actual_rot: np.ndarray) -> float | None:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return None
    axis = axis / axis_norm
    best: float | None = None
    for col in range(3):
        target_vec = target_rot[:, col]
        actual_vec = actual_rot[:, col]
        target_proj = target_vec - np.dot(target_vec, axis) * axis
        actual_proj = actual_vec - np.dot(actual_vec, axis) * axis
        if np.linalg.norm(target_proj) < 1e-6 or np.linalg.norm(actual_proj) < 1e-6:
            continue
        angle = angle_between_deg(target_proj, actual_proj)
        if angle is None:
            continue
        if best is None or angle < best:
            best = angle
    return best


def pct(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * quantile)))
    return ordered[idx]


def summary(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": pct(finite, 0.95),
        "max": max(finite),
    }


def canonical_joint_name(name: str) -> str:
    if name in ADAM_COMMAND_JOINTS_19:
        return name
    prefixed = f"dof_pos/{name}"
    if prefixed in ADAM_COMMAND_JOINTS_19:
        return prefixed
    return name


class RetargetEvaluator(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("retarget_evaluator")
        self.args = args
        self.model = mj.MjModel.from_xml_path(str(args.robot_xml))
        self.data = mj.MjData(self.model)
        self.joint_qpos_addr = self._joint_qpos_addr()
        self.joint_ranges = self._joint_ranges()
        self.body_ids = self._body_ids()
        self.ik_config = self._load_ik_config(args.ik_config)
        self.rotation_matrix = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            dtype=np.float64,
        )
        self.rotation_quat = quat_from_matrix(self.rotation_matrix)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.samples: dict[str, list[Sample]] = {"vendor": [], "gmr": []}
        self.counts = {"vendor": 0, "gmr": 0}
        self.dropped = defaultdict(int)

        self.create_subscription(
            JointState,
            args.vendor_topic,
            lambda msg: self._on_joint_state("vendor", msg),
            100,
        )
        self.create_subscription(
            JointState,
            args.gmr_topic,
            lambda msg: self._on_joint_state("gmr", msg),
            100,
        )

    def _joint_qpos_addr(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for joint_id in range(self.model.njnt):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                out[name] = int(self.model.jnt_qposadr[joint_id])
        missing = [joint for _, joint in ADAM_JOINT_NAME_MAP if joint not in out]
        if missing:
            raise RuntimeError(f"robot model is missing joints: {missing}")
        return out

    def _joint_ranges(self) -> dict[str, tuple[float, float]]:
        ranges = {}
        for joint_id in range(self.model.njnt):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                continue
            if int(self.model.jnt_limited[joint_id]):
                low, high = self.model.jnt_range[joint_id]
                ranges[name] = (float(low), float(high))
        return ranges

    def _body_ids(self) -> dict[str, int]:
        names = {"pelvis", *INTEREST_FRAMES}
        ids = {}
        for name in names:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise RuntimeError(f"robot model is missing body: {name}")
            ids[name] = int(body_id)
        return ids

    @staticmethod
    def _load_ik_config(path: pathlib.Path) -> dict[str, Any]:
        with path.open() as handle:
            return json.load(handle)

    def _on_joint_state(self, backend: str, msg: JointState) -> None:
        self.counts[backend] += 1
        joint_positions = self._positions_from_msg(msg)
        if joint_positions is None:
            self.dropped[f"{backend}:bad_joint_state"] += 1
            return

        targets = self._read_target_frames()
        if targets is None:
            self.dropped[f"{backend}:tf"] += 1

        q = np.array([joint_positions[name] for name in ADAM_COMMAND_JOINTS_19])
        qpos = self._make_qpos(joint_positions, targets)
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)
        robot_poses = self._robot_poses()
        metrics = self._metrics(robot_poses, targets)
        self.samples[backend].append(Sample(time.monotonic(), q, metrics))

    @staticmethod
    def _positions_from_msg(msg: JointState) -> dict[str, float] | None:
        out: dict[str, float] = {}
        for index, name in enumerate(msg.name):
            canonical = canonical_joint_name(name)
            if canonical not in ADAM_COMMAND_JOINTS_19:
                continue
            if index >= len(msg.position):
                return None
            value = float(msg.position[index])
            if not math.isfinite(value):
                return None
            out[canonical] = value
        if any(name not in out for name in ADAM_COMMAND_JOINTS_19):
            return None
        return out

    def _read_target_frames(self) -> dict[str, Pose] | None:
        human = {}
        try:
            for bone in NOITOM_BONES:
                transform = self.tf_buffer.lookup_transform(
                    self.args.base_frame,
                    bone,
                    Time(),
                )
                pos = np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ],
                    dtype=np.float64,
                )
                quat = quat_normalize(
                    np.array(
                        [
                            transform.transform.rotation.w,
                            transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                        ],
                        dtype=np.float64,
                    )
                )
                if self.args.apply_pnd_coordinate_transform:
                    pos = self.rotation_matrix @ pos
                    quat = quat_normalize(quat_mul(self.rotation_quat, quat))
                human[bone] = Pose(pos, quat_to_matrix(quat))
        except Exception:  # noqa: BLE001 - TF lookup misses are counted by caller.
            return None

        scaled = self._scale_human(human)
        targets_by_human = self._offset_human(scaled)
        targets_by_frame = {}
        for frame_name, entry in self.ik_config["ik_match_table1"].items():
            human_name = entry[0]
            if frame_name == "pelvis" or frame_name in INTEREST_FRAMES:
                if human_name in targets_by_human:
                    targets_by_frame[frame_name] = targets_by_human[human_name]
        return targets_by_frame

    def _scale_human(self, human: dict[str, Pose]) -> dict[str, Pose]:
        root_name = self.ik_config["human_root_name"]
        root = human[root_name]
        scale_table = self.ik_config["human_scale_table"]
        root_scale = float(scale_table.get(root_name, 1.0))
        scaled_root = Pose(root_scale * root.pos, root.rot)
        scaled = {root_name: scaled_root}
        for name, pose in human.items():
            if name == root_name:
                continue
            scale = float(scale_table.get(name, 1.0))
            pos = (pose.pos - root.pos) * scale + scaled_root.pos
            scaled[name] = Pose(pos, pose.rot)
        return scaled

    def _offset_human(self, human: dict[str, Pose]) -> dict[str, Pose]:
        pos_offsets = {}
        rot_offsets = {}
        for _frame_name, entry in self.ik_config["ik_match_table1"].items():
            human_name = entry[0]
            pos_offsets[human_name] = np.array(entry[3], dtype=np.float64)
            rot_offsets[human_name] = quat_to_matrix(
                quat_normalize(np.array(entry[4], dtype=np.float64))
            )

        out = {}
        for name, pose in human.items():
            if name not in pos_offsets:
                out[name] = pose
                continue
            rot = pose.rot @ rot_offsets[name]
            pos = pose.pos + rot @ pos_offsets[name]
            out[name] = Pose(pos, rot)
        return out

    def _make_qpos(
        self,
        joint_positions: dict[str, float],
        targets: dict[str, Pose] | None,
    ) -> np.ndarray:
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[3] = 1.0
        if targets is not None and "pelvis" in targets:
            qpos[:3] = targets["pelvis"].pos
            qpos[3:7] = quat_from_matrix(targets["pelvis"].rot)
        for ros_name, robot_joint in ADAM_JOINT_NAME_MAP:
            qpos[self.joint_qpos_addr[robot_joint]] = joint_positions[ros_name]
        return qpos

    def _robot_poses(self) -> dict[str, Pose]:
        poses = {}
        for name, body_id in self.body_ids.items():
            poses[name] = Pose(
                self.data.xpos[body_id].copy(),
                self.data.xmat[body_id].reshape(3, 3).copy(),
            )
        return poses

    def _metrics(
        self,
        robot: dict[str, Pose],
        target: dict[str, Pose] | None,
    ) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {
            "position_error_m": {},
            "orientation_error_deg": {},
            "direction_error_deg": {},
            "twist_error_deg": {},
        }
        if target is None:
            return metrics

        for frame_name in INTEREST_FRAMES:
            if frame_name not in target:
                continue
            metrics["position_error_m"][frame_name] = float(
                np.linalg.norm(robot[frame_name].pos - target[frame_name].pos)
            )
            metrics["orientation_error_deg"][frame_name] = rotation_error_deg(
                target[frame_name].rot,
                robot[frame_name].rot,
            )

        for name, (start, end) in SEGMENTS.items():
            if start not in target or end not in target:
                continue
            robot_vec = robot[end].pos - robot[start].pos
            target_vec = target[end].pos - target[start].pos
            angle = angle_between_deg(robot_vec, target_vec)
            if angle is not None:
                metrics["direction_error_deg"][name] = angle

        for name, (start, end) in TWIST_SEGMENTS.items():
            if start not in target or end not in target:
                continue
            axis = target[end].pos - target[start].pos
            twist = twist_error_deg(axis, target[end].rot, robot[end].rot)
            if twist is not None:
                metrics["twist_error_deg"][name] = twist
        return metrics

    def summarize(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "topics": {
                "vendor": self.args.vendor_topic,
                "gmr": self.args.gmr_topic,
            },
            "counts": self.counts,
            "dropped": dict(self.dropped),
            "backends": {},
            "pairwise_joint_diff_rad": {},
        }
        for backend in ("vendor", "gmr"):
            out["backends"][backend] = self._summarize_backend(backend)
        out["pairwise_joint_diff_rad"] = self._summarize_pairwise_joint_diff()
        return out

    def _summarize_backend(self, backend: str) -> dict[str, Any]:
        samples = self.samples[backend]
        result: dict[str, Any] = {
            "sample_count": len(samples),
            "rate_hz": None,
            "task_metrics": {},
            "dynamics": {},
            "joint_limits": {},
        }
        if len(samples) >= 2:
            duration = samples[-1].t - samples[0].t
            if duration > 0.0:
                result["rate_hz"] = (len(samples) - 1) / duration

        for metric_name in (
            "position_error_m",
            "orientation_error_deg",
            "direction_error_deg",
            "twist_error_deg",
        ):
            per_key: dict[str, list[float]] = defaultdict(list)
            all_values: list[float] = []
            for sample in samples:
                for key, value in sample.metrics.get(metric_name, {}).items():
                    per_key[key].append(value)
                    all_values.append(value)
            result["task_metrics"][metric_name] = {
                "overall": summary(all_values),
                "by_key": {key: summary(values) for key, values in per_key.items()},
            }

        result["dynamics"] = self._summarize_dynamics(samples)
        result["joint_limits"] = self._summarize_limits(samples)
        return result

    def _summarize_dynamics(self, samples: list[Sample]) -> dict[str, Any]:
        if len(samples) < 2:
            return {}
        q = np.vstack([sample.q for sample in samples])
        t = np.array([sample.t for sample in samples], dtype=np.float64)
        dt = np.diff(t)
        valid = dt > 1e-6
        dq = np.diff(q, axis=0)[valid]
        dt = dt[valid]
        if len(dt) == 0:
            return {}
        vel = np.abs(dq / dt[:, None])
        max_delta = np.abs(dq)
        result = {
            "max_delta_deg": summary(np.degrees(max_delta).ravel().tolist()),
            "velocity_deg_s": summary(np.degrees(vel).ravel().tolist()),
        }
        if len(vel) >= 2:
            dt_acc = dt[1:]
            acc = np.abs(np.diff(vel, axis=0) / dt_acc[:, None])
            result["acceleration_deg_s2"] = summary(np.degrees(acc).ravel().tolist())
            if len(acc) >= 2:
                jerk = np.abs(np.diff(acc, axis=0) / dt_acc[1:, None])
                result["jerk_deg_s3"] = summary(np.degrees(jerk).ravel().tolist())
        return result

    def _summarize_limits(self, samples: list[Sample]) -> dict[str, Any]:
        threshold = math.radians(self.args.limit_margin_deg)
        margins = []
        saturated = 0
        violations = 0
        per_joint_hits = defaultdict(int)
        per_joint_count = defaultdict(int)
        for sample in samples:
            for index, (_ros_name, robot_joint) in enumerate(ADAM_JOINT_NAME_MAP):
                if robot_joint not in self.joint_ranges:
                    continue
                low, high = self.joint_ranges[robot_joint]
                value = float(sample.q[index])
                margin = min(value - low, high - value)
                margins.append(margin)
                per_joint_count[robot_joint] += 1
                if value < low or value > high:
                    violations += 1
                if margin < threshold:
                    saturated += 1
                    per_joint_hits[robot_joint] += 1
        total = len(margins)
        rates = {
            joint: per_joint_hits[joint] / count
            for joint, count in per_joint_count.items()
            if count > 0
        }
        top = sorted(rates.items(), key=lambda item: item[1], reverse=True)[:8]
        return {
            "total_joint_samples": total,
            "saturation_rate": saturated / total if total else None,
            "violation_rate": violations / total if total else None,
            "min_margin_deg": math.degrees(min(margins)) if margins else None,
            "top_saturated_joints": top,
        }

    def _summarize_pairwise_joint_diff(self) -> dict[str, Any]:
        vendor = self.samples["vendor"]
        gmr = self.samples["gmr"]
        if not vendor or not gmr:
            return {"paired_samples": 0}

        diffs = []
        per_joint = defaultdict(list)
        j = 0
        for sample in vendor:
            while j + 1 < len(gmr) and abs(gmr[j + 1].t - sample.t) <= abs(gmr[j].t - sample.t):
                j += 1
            if abs(gmr[j].t - sample.t) > self.args.pair_tolerance_s:
                continue
            diff = np.abs(sample.q - gmr[j].q)
            diffs.extend(diff.tolist())
            for index, value in enumerate(diff):
                per_joint[ADAM_COMMAND_JOINTS_19[index]].append(float(value))
        top = sorted(
            (
                (statistics.fmean(values), pct(values, 0.95), max(values), joint)
                for joint, values in per_joint.items()
                if values
            ),
            reverse=True,
        )[:8]
        return {
            "paired_samples": len(diffs) // len(ADAM_COMMAND_JOINTS_19),
            "overall": summary(diffs),
            "top_joints": [
                {"joint": joint, "mean": mean, "p95": p95_value, "max": max_value}
                for mean, p95_value, max_value, joint in top
            ],
        }


def default_paths(gmr_repo_path: str) -> tuple[pathlib.Path, pathlib.Path]:
    root = pathlib.Path(gmr_repo_path).expanduser().resolve()
    robot_xml = root / "assets" / "pnd_adam_pro_body31" / "adam_pro_sharpa_body.xml"
    ik_config = (
        root
        / "general_motion_retargeting"
        / "ik_configs"
        / "noitom_to_adam_pro_body31.json"
    )
    return robot_xml, ik_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--vendor-topic", default=DEFAULT_VENDOR_TOPIC)
    parser.add_argument("--gmr-topic", default=DEFAULT_GMR_TOPIC)
    parser.add_argument("--base-frame", default="world_zup")
    parser.add_argument(
        "--gmr-repo-path",
        default=os.environ.get("PND_GMR_REPO", DEFAULT_GMR_REPO),
    )
    parser.add_argument("--robot-xml", type=pathlib.Path, default=None)
    parser.add_argument("--ik-config", type=pathlib.Path, default=None)
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument("--pair-tolerance-s", type=float, default=0.04)
    parser.add_argument("--limit-margin-deg", type=float, default=5.0)
    parser.add_argument(
        "--apply-pnd-coordinate-transform",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(remove_ros_args()[1:])
    default_robot_xml, default_ik_config = default_paths(args.gmr_repo_path)
    args.robot_xml = args.robot_xml or default_robot_xml
    args.ik_config = args.ik_config or default_ik_config
    return args


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_report(report: dict[str, Any]) -> None:
    print("Retarget evaluator report")
    print(f"  vendor_topic: {report['topics']['vendor']}")
    print(f"  gmr_topic:    {report['topics']['gmr']}")
    print(f"  dropped:      {report['dropped']}")
    for backend in ("vendor", "gmr"):
        data = report["backends"][backend]
        print(f"\n[{backend}] samples={data['sample_count']} rate_hz={fmt(data['rate_hz'], 2)}")
        for metric_name, label in (
            ("position_error_m", "position_error_m"),
            ("orientation_error_deg", "orientation_error_deg"),
            ("direction_error_deg", "direction_error_deg"),
            ("twist_error_deg", "twist_error_deg"),
        ):
            overall = data["task_metrics"][metric_name]["overall"]
            print(
                f"  {label}: mean={fmt(overall['mean'])} "
                f"median={fmt(overall['median'])} p95={fmt(overall['p95'])} "
                f"max={fmt(overall['max'])} count={overall['count']}"
            )
            by_key = data["task_metrics"][metric_name]["by_key"]
            ranked = sorted(
                by_key.items(),
                key=lambda item: -1.0
                if item[1]["p95"] is None
                else float(item[1]["p95"]),
                reverse=True,
            )[:5]
            for key, values in ranked:
                print(
                    f"    {key}: mean={fmt(values['mean'])} "
                    f"p95={fmt(values['p95'])} max={fmt(values['max'])}"
                )

        dynamics = data["dynamics"]
        if dynamics:
            for key in ("max_delta_deg", "velocity_deg_s", "acceleration_deg_s2", "jerk_deg_s3"):
                if key in dynamics:
                    values = dynamics[key]
                    print(
                        f"  {key}: p95={fmt(values['p95'])} "
                        f"max={fmt(values['max'])}"
                    )

        limits = data["joint_limits"]
        print(
            "  limits: "
            f"saturation_rate={fmt(limits.get('saturation_rate'), 4)} "
            f"violation_rate={fmt(limits.get('violation_rate'), 4)} "
            f"min_margin_deg={fmt(limits.get('min_margin_deg'))}"
        )
        for joint, rate in limits.get("top_saturated_joints", []):
            if rate > 0.0:
                print(f"    saturated {joint}: {rate:.3f}")

    pairwise = report["pairwise_joint_diff_rad"]
    print(f"\n[pairwise joint abs diff] paired_samples={pairwise.get('paired_samples', 0)}")
    if pairwise.get("overall"):
        overall = pairwise["overall"]
        print(
            f"  overall_rad: mean={fmt(overall['mean'])} "
            f"median={fmt(overall['median'])} p95={fmt(overall['p95'])} "
            f"max={fmt(overall['max'])}"
        )
        for item in pairwise.get("top_joints", []):
            print(
                f"    {item['joint']}: mean={fmt(item['mean'])} "
                f"p95={fmt(item['p95'])} max={fmt(item['max'])}"
            )


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")
    if not args.robot_xml.exists():
        raise SystemExit(f"robot XML not found: {args.robot_xml}")
    if not args.ik_config.exists():
        raise SystemExit(f"IK config not found: {args.ik_config}")

    rclpy.init()
    node = RetargetEvaluator(args)
    end = time.monotonic() + args.duration
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        report = node.summarize()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print_report(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
