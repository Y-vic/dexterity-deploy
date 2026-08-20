import numpy as np

from sharpa_policy_v3_client.hardware_execution import (
    ExecutionLimits,
    HardwareSnapshot,
    PolicyActionExecutor,
    prepare_executable_action,
)
from sharpa_policy_v3_client.hardware_geometry import (
    UrSharpAWireGeometry,
    rtde_pose_to_transform,
    transform_to_rtde_pose,
)


class FakeBackend:
    def __init__(self, snapshot):
        self.current = snapshot
        self.arm_commands = []
        self.hand_commands = []
        self.stop_count = 0

    def snapshot(self):
        return self.current

    def inverse_kinematics(self, side, target_rtde_pose, near_joint):
        return np.asarray(near_joint) + 0.01

    def command_arms(self, left_joint, right_joint, period_s):
        self.arm_commands.append((left_joint.copy(), right_joint.copy(), period_s))
        return True, True

    def command_hands(self, left_hand, right_hand):
        self.hand_commands.append((left_hand.copy(), right_hand.copy()))
        return True, True

    def safe_stop(self):
        self.stop_count += 1


def snapshot():
    geometry = UrSharpAWireGeometry()
    left_capture = np.eye(4)
    left_capture[:3, 3] = (0.25, 0.18, 0.32)
    right_capture = np.eye(4)
    right_capture[:3, 3] = (0.25, -0.18, 0.32)
    return HardwareSnapshot(
        left_rtde_pose=transform_to_rtde_pose(
            geometry.capture_tcp_to_ur_base_tcp(left_capture, "left")
        ),
        right_rtde_pose=transform_to_rtde_pose(
            geometry.capture_tcp_to_ur_base_tcp(right_capture, "right")
        ),
        left_joint=np.zeros(6),
        right_joint=np.zeros(6),
        left_hand=np.zeros(22),
        right_hand=np.zeros(22),
    )


def action_from_snapshot(current, *, horizon=4, execute_start=1, execute_length=2):
    geometry = UrSharpAWireGeometry()
    left = geometry.rtde_pose_to_wire_pose(current.left_rtde_pose, "left")
    right = geometry.rtde_pose_to_wire_pose(current.right_rtde_pose, "right")
    left_rows = np.repeat(left[None], horizon, axis=0)
    right_rows = np.repeat(right[None], horizon, axis=0)
    hands = np.zeros((horizon, 44), dtype=np.float32)
    return prepare_executable_action(
        action_id="action-1",
        frequency_hz=1000.0,
        action_length=horizon,
        execute_start=execute_start,
        execute_length=execute_length,
        left_wrist_action_type="eef",
        right_wrist_action_type="eef",
        left_wrist=left_rows,
        right_wrist=right_rows,
        hand_joint=hands,
        current_left_wire=left,
        current_right_wire=right,
        max_frequency_hz=2000.0,
    )


def test_executor_runs_only_negotiated_slice():
    current = snapshot()
    backend = FakeBackend(current)
    executor = PolicyActionExecutor(
        backend,
        enabled=True,
        limits=ExecutionLimits(max_frequency_hz=2000.0),
    )

    result = executor.execute(action_from_snapshot(current))

    assert result.success
    assert result.executed_steps == 2
    assert len(backend.arm_commands) == 2
    assert len(backend.hand_commands) == 2
    assert backend.stop_count == 0


def test_executor_is_non_actuating_by_default():
    current = snapshot()
    backend = FakeBackend(current)
    executor = PolicyActionExecutor(backend)

    result = executor.execute(action_from_snapshot(current))

    assert not result.success
    assert result.failure_code == "execution_disabled"
    assert backend.arm_commands == []
    assert backend.hand_commands == []


def test_workspace_violation_stops_before_command():
    current = snapshot()
    backend = FakeBackend(current)
    geometry = UrSharpAWireGeometry()
    action = action_from_snapshot(current, execute_start=0, execute_length=1)
    bad_capture = geometry.wire_pose_to_capture_tcp(action.left_wrist[0], "left")
    bad_capture[2, 3] = -0.1
    bad_left = geometry.capture_tcp_to_wire_pose(bad_capture, "left")
    object.__setattr__(action, "left_wrist", bad_left[None])
    executor = PolicyActionExecutor(
        backend,
        enabled=True,
        limits=ExecutionLimits(max_frequency_hz=2000.0),
    )

    result = executor.execute(action)

    assert not result.success
    assert result.executed_steps == 0
    assert backend.arm_commands == []
    assert backend.stop_count == 1


def test_rtde_snapshot_pose_is_a_rigid_transform():
    value = rtde_pose_to_transform(snapshot().left_rtde_pose)
    np.testing.assert_allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1.0e-7)
