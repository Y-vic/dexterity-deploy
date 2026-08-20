from __future__ import annotations

import numpy as np

from ws_core.kinematics import PndKinematics


def test_reachable_wrist_pose_roundtrip_from_zero_initial_position() -> None:
    kinematics = PndKinematics()
    qpos_start = kinematics.model.qpos0.copy()
    qpos_target = qpos_start.copy()
    right_arm_target = [-0.60, -0.72, 0.58, -1.50, 0.84, 0.19, 0.58]
    for address, value in zip(
        kinematics.variable_addrs[7:], right_arm_target, strict=True
    ):
        qpos_target[address] = value

    action = np.zeros(62, dtype=np.float32)
    action[:9] = kinematics.body_pose_hip(qpos_target, "left")
    action[9:18] = kinematics.body_pose_hip(qpos_target, "right")

    _, report = kinematics.solve_ik_step(qpos_start, qpos_start, action)

    assert report["success"]
    assert report["left_position_error_m"] < 1e-4
    assert report["right_position_error_m"] < 1e-4
    assert report["left_orientation_error_deg"] < 0.1
    assert report["right_orientation_error_deg"] < 0.1
