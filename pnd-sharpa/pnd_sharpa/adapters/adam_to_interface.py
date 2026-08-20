"""PND robot state to the shared absolute-EEF observation fields."""

from __future__ import annotations

from typing import Any

from ws_core.kinematics import PndKinematics


def adam_state_to_interface(
    robot_state: dict[str, Any],
    kinematics: PndKinematics,
) -> dict[str, Any]:
    converted = kinematics.convert_state(robot_state)
    return {
        "left_wrist": {
            "joint": None,
            "eef": converted.left_wrist_9d_hip.copy(),
            "eef_def": "absolute",
        },
        "right_wrist": {
            "joint": None,
            "eef": converted.right_wrist_9d_hip.copy(),
            "eef_def": "absolute",
        },
        "hand_joint": {
            "left": converted.sharpa_q44[:22].copy(),
            "right": converted.sharpa_q44[22:].copy(),
        },
    }
