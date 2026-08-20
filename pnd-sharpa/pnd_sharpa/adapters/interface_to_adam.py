"""Shared executable action to complete PND Adam/SharpA joint plans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ws_core.kinematics import PndKinematics


def interface_action_to_adam_plan(
    command: Mapping[str, Any],
    robot_state: dict[str, Any],
    kinematics: PndKinematics,
    *,
    enable_adam: bool = True,
    enable_sharpa: bool = True,
) -> dict[str, np.ndarray]:
    """Run the production PND IK for every row of an already sliced action."""

    action = command.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("executable command missing action")
    left = _absolute_eef(action, "left_wrist")
    right = _absolute_eef(action, "right_wrist")
    hands = action.get("hand_joint")
    if not isinstance(hands, Mapping):
        raise ValueError("executable action missing hand_joint")
    left_hand = _matrix(hands.get("left"), "hand_joint.left", width=22)
    right_hand = _matrix(hands.get("right"), "hand_joint.right", width=22)
    horizon = left.shape[0]
    for name, value in (
        ("right_wrist.eef", right),
        ("hand_joint.left", left_hand),
        ("hand_joint.right", right_hand),
    ):
        if value.shape[0] != horizon:
            raise ValueError(f"{name} horizon does not match left_wrist.eef")

    action62 = np.concatenate((left, right, left_hand, right_hand), axis=1)
    converted = kinematics.convert_state(robot_state)
    adam_rows: list[list[float]] = []
    sharpa_rows: list[list[float]] = []
    previous_qpos: np.ndarray | None = None
    for step in range(horizon):
        targets = kinematics.plan_action(
            action_rel62=action62,
            anchor_state_62d=converted.hand_pose_62d,
            robot_state=robot_state,
            action_step_index=step,
            enable_adam=enable_adam,
            enable_sharpa=enable_sharpa,
            action_frame="absolute_current_hip",
            qpos_previous=previous_qpos,
        )
        if targets.ik_qpos is not None:
            previous_qpos = targets.ik_qpos.copy()
        adam_rows.append(targets.adam_q19)
        sharpa_rows.append(targets.sharpa_q44)
    return {
        "adam": np.asarray(adam_rows, dtype=np.float32),
        "sharpa": np.asarray(sharpa_rows, dtype=np.float32),
        "action_abs62": action62,
    }


def _absolute_eef(action: Mapping[str, Any], name: str) -> np.ndarray:
    wrist = action.get(name)
    if not isinstance(wrist, Mapping):
        raise ValueError(f"executable action missing {name}")
    if wrist.get("eef_def") not in (None, "absolute"):
        raise ValueError(f"{name}.eef_def must be absolute")
    return _matrix(wrist.get("eef"), f"{name}.eef", width=9)


def _matrix(value: Any, name: str, *, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (T, {width})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array)
