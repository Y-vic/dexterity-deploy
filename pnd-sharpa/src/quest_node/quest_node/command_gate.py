"""Pure helpers for producing Noitom-compatible Adam commands."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


WAIST_JOINTS = (
    "dof_pos/waistRoll",
    "dof_pos/waistPitch",
    "dof_pos/waistYaw",
)
NECK_JOINTS = (
    "dof_pos/neckYaw",
    "dof_pos/neckPitch",
)
LEFT_ARM_JOINTS = (
    "dof_pos/shoulderPitch_Left",
    "dof_pos/shoulderRoll_Left",
    "dof_pos/shoulderYaw_Left",
    "dof_pos/elbow_Left",
    "dof_pos/wristYaw_Left",
    "dof_pos/wristPitch_Left",
    "dof_pos/wristRoll_Left",
)
RIGHT_ARM_JOINTS = (
    "dof_pos/shoulderPitch_Right",
    "dof_pos/shoulderRoll_Right",
    "dof_pos/shoulderYaw_Right",
    "dof_pos/elbow_Right",
    "dof_pos/wristYaw_Right",
    "dof_pos/wristPitch_Right",
    "dof_pos/wristRoll_Right",
)
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
NECK_WAIST_JOINTS = WAIST_JOINTS + NECK_JOINTS
ADAM_COMMAND_JOINTS_19 = NECK_WAIST_JOINTS + ARM_JOINTS


def canonical_joint_name(name: str) -> str:
    if name in ADAM_COMMAND_JOINTS_19:
        return name
    prefixed = f"dof_pos/{name}"
    if prefixed in ADAM_COMMAND_JOINTS_19:
        return prefixed
    return name


def positions_from_joint_arrays(
    names: Sequence[str],
    positions: Sequence[float],
    *,
    allowed: set[str] | frozenset[str] = frozenset(ADAM_COMMAND_JOINTS_19),
) -> dict[str, float]:
    """Extract finite, recognized positions from a JointState-like pair."""

    result: dict[str, float] = {}
    for index, name in enumerate(names):
        if not name:
            continue
        canonical = canonical_joint_name(str(name))
        if canonical not in allowed:
            continue
        if index >= len(positions):
            raise ValueError(f"JointState position is missing for {canonical}")
        try:
            value = float(positions[index])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-finite joint value for {canonical}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite joint value for {canonical}")
        result[canonical] = value
    return result


def make_command_positions(
    raw_positions: Mapping[str, float],
    *,
    fix_neck_waist: bool,
    bias_positions: Mapping[str, float] | None = None,
    bias_fresh: bool = False,
) -> tuple[dict[str, float], str]:
    """Build the exact 19D contract emitted by the current Noitom gate."""

    missing_arms = [name for name in ARM_JOINTS if name not in raw_positions]
    if missing_arms:
        raise ValueError(f"missing_quest_arm_joints:{missing_arms}")

    if fix_neck_waist:
        bias = bias_positions or {}
        if bias_fresh and all(name in bias for name in NECK_WAIST_JOINTS):
            neck_waist = {name: float(bias[name]) for name in NECK_WAIST_JOINTS}
            source = "bias_command"
        else:
            neck_waist = {name: 0.0 for name in NECK_WAIST_JOINTS}
            source = "zero_fallback"
        positions = {
            **neck_waist,
            **{name: float(raw_positions[name]) for name in ARM_JOINTS},
        }
        return positions, source

    missing = [name for name in ADAM_COMMAND_JOINTS_19 if name not in raw_positions]
    if missing:
        raise ValueError(f"missing_quest_command_joints:{missing}")
    return (
        {name: float(raw_positions[name]) for name in ADAM_COMMAND_JOINTS_19},
        "retarget",
    )


@dataclass(frozen=True)
class TrackingStatus:
    event: str
    sequence: int
    connected: bool
    calibrated: bool
    tracking_fresh: bool


def parse_tracking_status(payload: str) -> TrackingStatus:
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("tracking status must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("tracking status must be a JSON object")
    try:
        sequence = int(data.get("sequence", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("tracking status sequence must be an integer") from exc
    return TrackingStatus(
        event=str(data.get("event", "")),
        sequence=sequence,
        connected=data.get("connected") is True,
        calibrated=data.get("calibrated") is True,
        tracking_fresh=data.get("tracking_fresh") is True,
    )


class TrackingWatchdog:
    """Track real WebSocket frame heartbeats without refreshing duplicates."""

    def __init__(self, timeout: float) -> None:
        if timeout <= 0.0:
            raise ValueError("tracking timeout must be positive")
        self.timeout = float(timeout)
        self.last_sequence: int | None = None
        self.last_frame_time: float | None = None
        self.status_valid = False

    def observe(self, status: TrackingStatus, now: float) -> None:
        valid = status.connected and status.calibrated and status.tracking_fresh
        self.status_valid = valid
        if status.event == "frame" and valid and status.sequence != self.last_sequence:
            self.last_sequence = status.sequence
            self.last_frame_time = float(now)

    def is_fresh(self, now: float) -> bool:
        return (
            self.status_valid
            and self.last_frame_time is not None
            and float(now) - self.last_frame_time <= self.timeout
        )

    def age(self, now: float) -> float | None:
        if self.last_frame_time is None:
            return None
        return max(0.0, float(now) - self.last_frame_time)
