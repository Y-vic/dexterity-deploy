from types import SimpleNamespace

import numpy as np
import pytest

from sharpa_policy_v3_client import hardware_drivers
from sharpa_policy_v3_client.hardware_drivers import (
    FINGER_NAMES,
    HardwareCommandError,
    SharpADevicePair,
    TACTILE_CHANNELS,
    UrRtdePair,
)
from sharpa_policy_v3_client.hardware_geometry import (
    UrSharpAWireGeometry,
    transform_to_rtde_pose,
)


class FakeReceiver:
    def __init__(self, joint, pose):
        self.joint = joint
        self.pose = pose

    def getActualQ(self):
        return self.joint

    def getActualTCPPose(self):
        return self.pose

    def isConnected(self):
        return True

    def isProtectiveStopped(self):
        return False

    def isEmergencyStopped(self):
        return False

    def getSafetyMode(self):
        return 1

    def getRobotMode(self):
        return 7


class FakeHand:
    def __init__(self, offset):
        self.commands = []
        self.state = SimpleNamespace(
            angles=np.arange(22, dtype=np.float32) + offset,
            torques=np.arange(22, dtype=np.float32) - offset,
        )

    def get_states(self):
        return self.state

    def set_joint_position(self, target, interpolation):
        self.commands.append((list(target), interpolation))
        self.state.angles = np.asarray(target, dtype=np.float32)
        return SimpleNamespace(code=0, message="")


class FakeController:
    def __init__(self, receiver):
        self.receiver = receiver
        self.commands = []
        self.stop_calls = 0

    def servoJ(self, target, *parameters):
        self.commands.append((list(target), parameters))
        self.receiver.joint = np.asarray(target, dtype=np.float64)
        return True

    def servoStop(self):
        self.stop_calls += 1


def test_ur_control_connection_stabilizes_each_receiver(monkeypatch):
    events = []

    class FakeReceiveInterface:
        def __init__(self, ip, *_args):
            events.append(("receive", ip))

    class FakeControlInterface:
        FLAG_VERBOSE = 1
        FLAG_UPLOAD_SCRIPT = 2
        FLAG_NO_WAIT = 4

        def __init__(self, ip, _frequency, flags, *_args):
            self.ip = ip
            events.append(("control", ip, flags))

        def isConnected(self):
            events.append(("connected", self.ip))
            return True

        def isProgramRunning(self):
            events.append(("program", self.ip))
            return True

    modules = {
        "rtde_receive": SimpleNamespace(RTDEReceiveInterface=FakeReceiveInterface),
        "rtde_control": SimpleNamespace(RTDEControlInterface=FakeControlInterface),
    }
    monkeypatch.setattr(
        hardware_drivers.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        hardware_drivers.time,
        "sleep",
        lambda delay: events.append(("sleep", delay)),
    )
    driver = UrRtdePair(enable_control=True, control_connection_delay_s=0.5)

    driver.connect()

    assert events == [
        ("receive", "192.168.56.20"),
        ("sleep", 0.5),
        ("control", "192.168.56.20", 7),
        ("connected", "192.168.56.20"),
        ("program", "192.168.56.20"),
        ("receive", "192.168.56.10"),
        ("sleep", 0.5),
        ("control", "192.168.56.10", 7),
        ("connected", "192.168.56.10"),
        ("program", "192.168.56.10"),
    ]


def test_ur_control_connection_reuploads_script_until_ready(monkeypatch):
    events = []
    program_checks = {"192.168.56.20": 0, "192.168.56.10": 0}

    class FakeReceiveInterface:
        def __init__(self, ip, *_args):
            events.append(("receive", ip))

    class FakeControlInterface:
        FLAG_VERBOSE = 1
        FLAG_UPLOAD_SCRIPT = 2
        FLAG_NO_WAIT = 4

        def __init__(self, ip, *_args):
            self.ip = ip
            events.append(("control", ip))

        def isConnected(self):
            return True

        def isProgramRunning(self):
            program_checks[self.ip] += 1
            return self.ip == "192.168.56.10" or program_checks[self.ip] > 1

        def reuploadScript(self):
            events.append(("reupload", self.ip))
            return True

    modules = {
        "rtde_receive": SimpleNamespace(RTDEReceiveInterface=FakeReceiveInterface),
        "rtde_control": SimpleNamespace(RTDEControlInterface=FakeControlInterface),
    }
    monkeypatch.setattr(
        hardware_drivers.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        hardware_drivers.time,
        "sleep",
        lambda delay: events.append(("sleep", delay)),
    )
    driver = UrRtdePair(
        enable_control=True,
        control_connection_delay_s=0.0,
        control_ready_timeout_s=1.0,
        control_ready_poll_s=0.1,
    )

    driver.connect()

    assert program_checks == {"192.168.56.20": 2, "192.168.56.10": 1}
    assert ("reupload", "192.168.56.20") in events
    assert ("reupload", "192.168.56.10") not in events
    assert ("sleep", 0.1) in events


def test_ur_read_publishes_joint_and_wire_eef_for_both_sides():
    geometry = UrSharpAWireGeometry()
    driver = UrRtdePair(geometry=geometry)
    poses = {}
    for side, y in (("left", 0.15), ("right", -0.15)):
        capture = np.eye(4)
        capture[:3, 3] = (0.25, y, 0.30)
        poses[side] = transform_to_rtde_pose(
            geometry.capture_tcp_to_ur_base_tcp(capture, side)
        )
        driver.receivers[side] = FakeReceiver(np.arange(6), poses[side])

    snapshot = driver.read()

    assert snapshot.normal_mode
    np.testing.assert_array_equal(snapshot.left_joint, np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(snapshot.right_joint, np.arange(6, dtype=np.float32))
    np.testing.assert_allclose(
        snapshot.left_wire_eef,
        geometry.rtde_pose_to_wire_pose(poses["left"], "left"),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        snapshot.right_wire_eef,
        geometry.rtde_pose_to_wire_pose(poses["right"], "right"),
        atol=1.0e-6,
    )


def test_ur_command_is_disabled_without_control_connection():
    driver = UrRtdePair(enable_control=False)

    with pytest.raises(HardwareCommandError, match="disabled"):
        driver.command_eef_step(np.zeros(9), np.zeros(9), 1.0 / 30.0)


def test_ur_initialization_interpolates_both_arms_and_seeds_ik_reference():
    driver = UrRtdePair(enable_control=True)
    left_receiver = FakeReceiver(np.zeros(6), np.zeros(6))
    right_receiver = FakeReceiver(np.zeros(6), np.zeros(6))
    driver.receivers = {"left": left_receiver, "right": right_receiver}
    driver.controllers = {
        "left": FakeController(left_receiver),
        "right": FakeController(right_receiver),
    }
    left_target = np.asarray([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])
    right_target = -left_target

    driver.initialize_joints(
        left_target,
        right_target,
        steps=4,
        step_delay_s=0.0,
        tolerance_rad=1.0e-6,
    )

    assert len(driver.controllers["left"].commands) == 4
    assert len(driver.controllers["right"].commands) == 4
    np.testing.assert_allclose(left_receiver.joint, left_target)
    np.testing.assert_allclose(right_receiver.joint, right_target)
    np.testing.assert_allclose(driver.last_target_joint["left"], left_target)
    np.testing.assert_allclose(driver.last_target_joint["right"], right_target)
    assert driver.controllers["left"].stop_calls == 1
    assert driver.controllers["right"].stop_calls == 1


def test_sharpa_snapshot_uses_canonical_pinky_to_thumb_channels():
    driver = SharpADevicePair(sdk_root="/unused")
    driver.hands = {"left": FakeHand(1.0), "right": FakeHand(2.0)}
    channel_layout = {
        "left": (5, 6, 7, 8, 9),
        "right": (0, 1, 2, 3, 4),
    }
    for side, channels in channel_layout.items():
        callback = driver._tactile_callback(side)
        for channel in channels:
            callback(
                {
                    "channel": channel,
                    "content": {
                        "F6": np.arange(6, dtype=np.float32) + channel * 10,
                        "DEFORM": np.full(
                            (240, 240),
                            50 + channel,
                            dtype=np.uint8,
                        ),
                    },
                }
            )

    snapshot = driver.read()

    assert FINGER_NAMES == ("pinky", "ring", "middle", "index", "thumb")
    assert TACTILE_CHANNELS == channel_layout
    assert snapshot.left_joint_valid
    assert snapshot.right_joint_valid
    for side, channels in channel_layout.items():
        np.testing.assert_array_equal(
            getattr(snapshot, f"{side}_wrench"),
            np.stack(
                [np.arange(6, dtype=np.float32) + channel * 10 for channel in channels]
            ),
        )
        assert getattr(snapshot, f"{side}_wrench_valid").all()
        np.testing.assert_array_equal(
            getattr(snapshot, f"{side}_deformation")[:, 0, 0],
            np.asarray([50 + channel for channel in channels], dtype=np.uint8),
        )
        assert getattr(snapshot, f"{side}_deformation_valid").all()


def test_sharpa_command_is_disabled_by_default():
    driver = SharpADevicePair(sdk_root="/unused")

    with pytest.raises(HardwareCommandError, match="disabled"):
        driver.command(np.zeros(22), np.zeros(22))


def test_sharpa_initialization_interpolates_both_hands_to_zero():
    driver = SharpADevicePair(sdk_root="/unused", enable_control=True)
    driver.hands = {"left": FakeHand(1.0), "right": FakeHand(2.0)}

    driver.initialize_joints(
        np.zeros(22),
        np.zeros(22),
        steps=3,
        step_delay_s=0.0,
        tolerance_rad=1.0e-6,
    )

    assert len(driver.hands["left"].commands) == 3
    assert len(driver.hands["right"].commands) == 3
    np.testing.assert_array_equal(driver.hands["left"].state.angles, np.zeros(22))
    np.testing.assert_array_equal(driver.hands["right"].state.angles, np.zeros(22))
