from types import SimpleNamespace

import numpy as np
import pytest

from ur_common.drivers import (
    HardwareCommandError,
    HardwareConnectionError,
    SharpADevicePair,
    UrRtdePair,
    format_ur_pair_diagnostics,
)


class FakeReceiver:
    def __init__(self):
        self.joint = np.zeros(6, dtype=np.float64)

    def getActualQ(self):
        return self.joint

    def getActualTCPPose(self):
        return np.zeros(6, dtype=np.float64)

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


class FakeController:
    def __init__(self, receiver):
        self.receiver = receiver
        self.commands = []

    def servoJ(self, target, *_args):
        self.commands.append(target)
        self.receiver.joint = np.asarray(target, dtype=np.float64)

    def servoStop(self):
        pass


class FakePayloadReceiver(FakeReceiver):
    def __init__(self, mass=2.5, cog=(0.0, 0.0, 0.0)):
        super().__init__()
        self.payload_mass = float(mass)
        self.payload_cog = np.asarray(cog, dtype=np.float64)
        self.safety_mode = 1
        self.safety_status_bits = 1
        self.protective_stopped = False
        self.tcp_wrench = np.zeros(6, dtype=np.float64)

    def isProtectiveStopped(self):
        return self.protective_stopped

    def getSafetyMode(self):
        return self.safety_mode

    def getSafetyStatusBits(self):
        return self.safety_status_bits

    def getPayload(self):
        return self.payload_mass

    def getPayloadCog(self):
        return self.payload_cog

    def getActualTCPForce(self):
        return self.tcp_wrench


class FakePayloadController(FakeController):
    def __init__(self, receiver, update_readback=True):
        super().__init__(receiver)
        self.payload_commands = []
        self.update_readback = bool(update_readback)

    def setPayload(self, mass, cog):
        self.payload_commands.append((float(mass), list(cog)))
        if self.update_readback:
            self.receiver.payload_mass = float(mass)
            self.receiver.payload_cog = np.asarray(cog, dtype=np.float64)
        return True


class FakeHand:
    def __init__(self):
        self.angles = np.zeros(22, dtype=np.float32)
        self.commands = []

    def get_states(self):
        return SimpleNamespace(angles=self.angles, torques=np.zeros(22))

    def set_joint_position(self, target, _interpolate):
        self.commands.append(np.asarray(target, dtype=np.float64))
        self.angles = np.asarray(target, dtype=np.float32)
        return SimpleNamespace(code=0, message="")


class FakeControlledHand(FakeHand):
    def __init__(self, *, mode=10, source=7, ignore_mode_write=False):
        super().__init__()
        self.mode = mode
        self.source = source
        self.ignore_mode_write = ignore_mode_write

    def set_control_mode(self, mode):
        if not self.ignore_mode_write:
            self.mode = mode
        return SimpleNamespace(code=0, message="")

    def get_control_mode(self):
        return SimpleNamespace(code=0, message=""), self.mode

    def set_control_source(self, source):
        self.source = source
        return SimpleNamespace(code=0, message="")

    def get_control_source(self):
        return SimpleNamespace(code=0, message=""), self.source


def test_ur_rejects_large_live_target_without_sending_servoj():
    driver = UrRtdePair(enable_control=True, enable_motion_limits=True, max_joint_step_rad=0.35)
    driver.receivers = {side: FakeReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakeController(driver.receivers[side]) for side in ("left", "right")
    }

    target = np.full(6, 1.2)
    with pytest.raises(HardwareCommandError, match="joint target step exceeds"):
        driver.command_joint_step(target, target, 1.0 / 30.0)

    for controller in driver.controllers.values():
        assert controller.commands == []


def test_ur_step_limit_allows_small_floating_point_tolerance():
    driver = UrRtdePair(
        enable_control=True, enable_motion_limits=True, max_joint_step_rad=0.35
    )
    driver.receivers = {side: FakeReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakeController(driver.receivers[side]) for side in ("left", "right")
    }
    target = np.full(6, 0.35 + 5.0e-10)

    driver.command_joint_step(target, target, 1.0 / 30.0)

    for controller in driver.controllers.values():
        np.testing.assert_allclose(controller.commands[-1], target)


def test_ur_can_disable_step_limit_without_disabling_absolute_limits():
    driver = UrRtdePair(
        enable_control=True,
        enable_motion_limits=True,
        enable_joint_step_limit=False,
        max_joint_step_rad=0.35,
    )
    driver.receivers = {side: FakeReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakeController(driver.receivers[side]) for side in ("left", "right")
    }

    target = np.full(6, 1.2)
    driver.command_joint_step(target, target, 1.0 / 30.0)

    for controller in driver.controllers.values():
        np.testing.assert_allclose(controller.commands[-1], target)

    outside_absolute_limit = np.full(6, 2.0 * np.pi + 0.1)
    with pytest.raises(HardwareCommandError, match="absolute limit"):
        driver.command_joint_step(
            outside_absolute_limit, outside_absolute_limit, 1.0 / 30.0
        )


def test_ur_payload_configuration_disabled_never_writes_controller():
    driver = UrRtdePair(enable_control=True, enable_payload_configuration=False)
    driver.receivers = {side: FakePayloadReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakePayloadController(driver.receivers[side])
        for side in ("left", "right")
    }

    driver._configure_payloads()

    for controller in driver.controllers.values():
        assert controller.payload_commands == []


def test_ur_payload_configuration_sets_each_side_and_verifies_readback():
    driver = UrRtdePair(
        enable_control=True,
        enable_payload_configuration=True,
        left_payload_mass_kg=3.1,
        left_payload_cog_m=[0.01, -0.02, 0.03],
        right_payload_mass_kg=3.4,
        right_payload_cog_m=[-0.01, 0.02, 0.04],
    )
    driver.receivers = {side: FakePayloadReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakePayloadController(driver.receivers[side])
        for side in ("left", "right")
    }

    driver._configure_payloads()

    assert driver.controllers["left"].payload_commands == [
        (3.1, [0.01, -0.02, 0.03])
    ]
    assert driver.controllers["right"].payload_commands == [
        (3.4, [-0.01, 0.02, 0.04])
    ]


def test_ur_payload_configuration_rejects_unverifiable_receiver_before_write():
    driver = UrRtdePair(
        enable_control=True,
        enable_payload_configuration=True,
        left_payload_mass_kg=3.1,
        right_payload_mass_kg=3.1,
    )
    driver.receivers = {side: FakeReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakePayloadController(driver.receivers[side])
        for side in ("left", "right")
    }

    with pytest.raises(HardwareConnectionError, match="refusing unverified payload write"):
        driver._configure_payloads()

    for controller in driver.controllers.values():
        assert controller.payload_commands == []


def test_ur_payload_configuration_fails_when_readback_does_not_match():
    driver = UrRtdePair(
        enable_control=True,
        enable_payload_configuration=True,
        left_payload_mass_kg=3.1,
        right_payload_mass_kg=3.1,
        payload_verify_timeout_s=0.001,
        payload_verify_poll_s=0.001,
    )
    driver.receivers = {side: FakePayloadReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakePayloadController(driver.receivers[side], update_readback=False)
        for side in ("left", "right")
    }

    with pytest.raises(HardwareConnectionError, match="readback did not match"):
        driver._configure_payloads()


def test_ur_protective_stop_snapshot_reports_side_payload_and_wrench():
    driver = UrRtdePair()
    left = FakePayloadReceiver(mass=2.5, cog=(0.01, 0.02, 0.03))
    right = FakePayloadReceiver(mass=2.7, cog=(-0.01, 0.0, 0.04))
    left.protective_stopped = True
    left.safety_mode = 3
    left.safety_status_bits = (1 << 2) | (1 << 10)
    left.tcp_wrench = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
    driver.receivers = {"left": left, "right": right}

    snapshot = driver.read()
    report = format_ur_pair_diagnostics(snapshot)

    assert snapshot.normal_mode is False
    assert snapshot.left_diagnostic.normal_mode is False
    assert snapshot.right_diagnostic.normal_mode is True
    assert "left: normal=false" in report
    assert "safety_mode=3(PROTECTIVE_STOP)" in report
    assert "PROTECTIVE_STOPPED|STOPPED_DUE_TO_SAFETY" in report
    assert "payload=2.500kg" in report
    assert "cog_m=[0.010,0.020,0.030]" in report
    assert "tcp_wrench=[10.000,20.000,30.000,1.000,2.000,3.000]" in report
    assert "right: normal=true" in report


def test_ur_synchronize_command_state_adopts_fresh_measured_joints():
    driver = UrRtdePair()
    left = FakePayloadReceiver()
    right = FakePayloadReceiver()
    left.joint = np.linspace(-0.5, 0.5, 6)
    right.joint = np.linspace(0.6, -0.4, 6)
    driver.receivers = {"left": left, "right": right}
    driver.last_target_joint = {
        "left": np.full(6, 1.0),
        "right": np.full(6, -1.0),
    }

    snapshot = driver.synchronize_command_state()

    np.testing.assert_allclose(
        driver.last_target_joint["left"], snapshot.left_joint
    )
    np.testing.assert_allclose(
        driver.last_target_joint["right"], snapshot.right_joint
    )
    assert driver.last_target_joint["left"].dtype == np.float64
    assert driver.last_target_joint["right"].dtype == np.float64


def test_ur_synchronize_command_state_rejects_abnormal_snapshot():
    driver = UrRtdePair()
    left = FakePayloadReceiver()
    right = FakePayloadReceiver()
    left.protective_stopped = True
    left.safety_mode = 3
    driver.receivers = {"left": left, "right": right}

    with pytest.raises(HardwareCommandError, match="outside normal mode"):
        driver.synchronize_command_state()

    assert driver.last_target_joint == {"left": None, "right": None}


def test_sharpa_slews_large_live_target_instead_of_rejecting_it():
    driver = SharpADevicePair(
        sdk_root="/unused", enable_control=True, max_hand_step_rad=0.5
    )
    driver.hands = {side: FakeHand() for side in ("left", "right")}

    dispatched = driver.command(np.full(22, 1.7), np.full(22, -1.7))

    np.testing.assert_allclose(driver.hands["left"].commands[-1], 0.5)
    np.testing.assert_allclose(driver.hands["right"].commands[-1], -0.5)
    np.testing.assert_allclose(dispatched["left"], 0.5)
    np.testing.assert_allclose(dispatched["right"], -0.5)


def test_sharpa_command_reports_each_blocking_sdk_stage():
    driver = SharpADevicePair(
        sdk_root="/unused", enable_control=True, max_hand_step_rad=0.5
    )
    driver.hands = {side: FakeHand() for side in ("left", "right")}
    events = []

    driver.command(
        np.full(22, 0.1),
        np.full(22, -0.1),
        event_callback=lambda event, **fields: events.append((event, fields)),
    )

    names = [event for event, _fields in events]
    assert names == [
        "sharpa_sdk_state_read_started",
        "sharpa_sdk_state_read_completed",
        "sharpa_sdk_hand_command_started",
        "sharpa_sdk_hand_command_completed",
        "sharpa_sdk_hand_command_started",
        "sharpa_sdk_hand_command_completed",
    ]
    assert [
        fields["side"]
        for event, fields in events
        if event == "sharpa_sdk_hand_command_started"
    ] == ["left", "right"]


def test_sharpa_commands_connected_left_without_right_hand():
    driver = SharpADevicePair(
        sdk_root="/unused", enable_control=True, max_hand_step_rad=0.5
    )
    driver.hands = {"left": FakeHand()}

    dispatched = driver.command(
        np.full(22, 0.4), np.full(22, 9.0),
        left_enable=True, right_enable=False,
    )

    assert set(dispatched) == {"left"}
    np.testing.assert_allclose(driver.hands["left"].commands[-1], 0.4)


def test_sharpa_position_preflight_repairs_stale_mode_on_both_hands():
    driver = SharpADevicePair(sdk_root="/unused", enable_control=True)
    driver.sdk = SimpleNamespace(
        ControlMode=SimpleNamespace(POSITION=13),
        ControlSource=SimpleNamespace(SDK=4),
    )
    driver.hands = {
        side: FakeControlledHand(mode=10, source=7)
        for side in ("left", "right")
    }

    driver.set_position_control()

    assert all(hand.mode == 13 for hand in driver.hands.values())
    assert all(hand.source == 4 for hand in driver.hands.values())


def test_sharpa_position_preflight_rejects_false_success_from_sdk():
    driver = SharpADevicePair(sdk_root="/unused", enable_control=True)
    driver.sdk = SimpleNamespace(
        ControlMode=SimpleNamespace(POSITION=13),
        ControlSource=SimpleNamespace(SDK=4),
    )
    driver.hands = {
        "left": FakeControlledHand(mode=10, source=7, ignore_mode_write=True),
        "right": FakeControlledHand(mode=10, source=7),
    }

    with pytest.raises(HardwareCommandError, match="readback mismatch"):
        driver.set_position_control()


def test_ur_single_arm_command_does_not_servo_disabled_arm():
    driver = UrRtdePair(enable_control=True)
    driver.receivers = {side: FakeReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakeController(driver.receivers[side]) for side in ("left", "right")
    }

    driver.command_joint_step(
        np.full(6, 0.2), np.full(6, 9.0), 1.0 / 30.0,
        left_enable=True, right_enable=False,
    )

    assert len(driver.controllers["left"].commands) == 1
    assert driver.controllers["right"].commands == []
