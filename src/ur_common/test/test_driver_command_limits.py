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
        self.interpolation_modes = []

    def get_states(self):
        return SimpleNamespace(angles=self.angles, torques=np.zeros(22))

    def set_joint_position(self, target, interpolate):
        self.commands.append(np.asarray(target, dtype=np.float64))
        self.interpolation_modes.append(bool(interpolate))
        self.angles = np.asarray(target, dtype=np.float32)
        return SimpleNamespace(code=0, message="")

    def get_device_info(self):
        return SimpleNamespace(
            status=SimpleNamespace(error_code=0x0241, error_joint=7)
        )

    def get_fault_code(self):
        return {0x0503: [7, 9]}


class FakeControlledHand(FakeHand):
    def __init__(
        self, *, mode=10, source=7, ignore_mode_write=False,
        fail_redundant_mode_write=False, fail_redundant_source_write=False,
        enabled=True,
    ):
        super().__init__()
        self.mode = mode
        self.source = source
        self.enabled = enabled
        self.ignore_mode_write = ignore_mode_write
        self.fail_redundant_mode_write = fail_redundant_mode_write
        self.fail_redundant_source_write = fail_redundant_source_write
        self.mode_writes = 0
        self.source_writes = 0

    def set_control_mode(self, mode):
        self.mode_writes += 1
        if self.fail_redundant_mode_write and self.mode == mode:
            return SimpleNamespace(code=2, message="Failure to connect to server")
        if not self.ignore_mode_write:
            self.mode = mode
            self.enabled = True
        return SimpleNamespace(code=0, message="")

    def get_control_mode(self):
        return SimpleNamespace(code=0, message=""), self.mode

    def get_enable_state(self):
        return SimpleNamespace(code=0, message=""), self.enabled

    def set_control_source(self, source):
        self.source_writes += 1
        if self.fail_redundant_source_write and self.source == source:
            return SimpleNamespace(code=2, message="Failure to connect to server")
        self.source = source
        return SimpleNamespace(code=0, message="")

    def get_control_source(self):
        return SimpleNamespace(code=0, message=""), self.source


class FakeConnectHand(FakeControlledHand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started = False
        self.speed_coefficient = 0.2
        self.current_coefficient = 0.2
        self.setting_writes = []

    def set_speed_coeff(self, value):
        self.setting_writes.append('speed')
        self.speed_coefficient = value
        return SimpleNamespace(code=0, message="")

    def get_speed_coeff(self):
        return SimpleNamespace(code=0, message=""), self.speed_coefficient

    def set_current_coeff(self, value):
        self.setting_writes.append('current')
        self.current_coefficient = value
        return SimpleNamespace(code=0, message="")

    def get_current_coeff(self):
        return SimpleNamespace(code=0, message=""), self.current_coefficient

    def set_control_source(self, source):
        self.setting_writes.append('source')
        return super().set_control_source(source)

    def set_tactile_callback(self, _callback):
        pass

    def start(self):
        self.started = True
        return True

    def stop(self):
        self.started = False


class FakeWaveManager:
    def __init__(self, devices, *, fail_serial=""):
        self.devices = list(devices)
        self.fail_serial = fail_serial
        self.connected = {}

    def get_all_devices(self):
        return list(self.devices)

    def is_connected(self, serial_number):
        return serial_number in self.connected

    def connect(self, serial_number, _skip_tactile=False):
        if serial_number == self.fail_serial:
            raise RuntimeError("socket refused")
        hand = FakeConnectHand()
        self.connected[serial_number] = hand
        return hand

    def disconnect(self, serial_number):
        self.connected.pop(serial_number, None)

    def disconnect_all(self):
        self.connected.clear()


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
    assert driver.hands["left"].interpolation_modes[-1] is True
    assert driver.hands["right"].interpolation_modes[-1] is True


def test_sharpa_fault_snapshot_keeps_heartbeat_and_active_faults_per_side():
    driver = SharpADevicePair(sdk_root="/unused", enable_control=True)
    driver.sdk = SimpleNamespace(
        FaultCode=SimpleNamespace(
            ERROR_POWER_HAND_OVERCURRENT=0x0241,
            ERROR_TORQUE_LIMIT_EXCEEDED=0x0503,
        )
    )
    driver.hands = {side: FakeHand() for side in ("left", "right")}

    snapshot = driver.read_faults()

    assert snapshot["fault_timestamp_ns"] > 0
    for side in ("left", "right"):
        payload = snapshot["sides"][side]
        assert payload["connected"] is True
        assert payload["heartbeat_fault_code"] == 0x0241
        assert payload["heartbeat_fault_name"] == "ERROR_POWER_HAND_OVERCURRENT"
        assert payload["heartbeat_error_joint"] == 7
        assert payload["active_faults"] == [
            {
                "code": 0x0503,
                "name": "ERROR_TORQUE_LIMIT_EXCEEDED",
                "indices": [7, 9],
            }
        ]


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
    for event, fields in events:
        if event == "sharpa_sdk_hand_command_completed":
            np.testing.assert_allclose(fields["target_rad"],
                                       driver.hands[fields["side"]].commands[-1])


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


def test_sharpa_partial_connect_reports_side_stage_and_reconnects_only_missing():
    left = SimpleNamespace(
        sn="left-sn", ip="192.168.10.10", hand_side=0, device_type=9
    )
    right = SimpleNamespace(
        sn="right-sn", ip="192.168.10.20", hand_side=1, device_type=9
    )
    manager = FakeWaveManager([left, right], fail_serial="right-sn")
    driver = SharpADevicePair(sdk_root="/unused", enable_control=True)
    driver.sdk = SimpleNamespace(
        DeviceType=SimpleNamespace(HAND=9),
        HandSide=SimpleNamespace(LEFT=0),
        ControlMode=SimpleNamespace(FLOATING=10, POSITION=13),
        ControlSource=SimpleNamespace(SDK=4),
    )
    driver.manager = manager

    driver._connect_devices([left, right])

    assert set(driver.hands) == {"left"}
    assert driver.hands["left"].mode == 13
    assert driver.hands["left"].mode_writes == 1
    assert "SharpA right manager.connect failed" in driver.connection_error
    assert "ip=192.168.10.20" in driver.connection_error

    left_hand = driver.hands["left"]
    manager.fail_serial = ""
    assert driver.reconnect_missing() is True

    assert driver.hands["left"] is left_hand
    assert driver.hands["left"].mode_writes == 1
    assert driver.hands["right"].mode == 13
    assert driver.connection_error == ""


def test_sharpa_skip_tactile_keeps_joints_without_tactile_connection_or_arrays():
    class Manager(FakeWaveManager):
        def connect(self, serial_number, skip_tactile=False):
            assert skip_tactile
            hand = super().connect(serial_number, skip_tactile)
            hand.set_tactile_callback = lambda _: pytest.fail("tactile callback must be disabled")
            return hand
    device = SimpleNamespace(sn="left-sn", ip="test", hand_side=0, device_type=9)
    driver = SharpADevicePair(sdk_root="/unused", skip_tactile=True)
    driver.sdk = SimpleNamespace(ControlMode=SimpleNamespace(POSITION=13),
                                 ControlSource=SimpleNamespace(SDK=4),
                                 HandSide=SimpleNamespace(LEFT=0))
    driver.manager = Manager([device])
    driver._connect_devices([device])
    assert "left" in driver.hands
    snapshot = driver.read()
    assert snapshot.left_joint_valid and snapshot.left_tau_valid
    assert snapshot.left_deformation.nbytes == snapshot.right_deformation.nbytes == 0
    assert not snapshot.left_deformation_valid.any()


def test_tactile_freshness_is_per_finger_side_and_modality(monkeypatch):
    from ur_common.drivers import TACTILE_CHANNELS
    now = [10.]
    monkeypatch.setattr('ur_common.drivers.time.monotonic', lambda: now[0])
    driver = SharpADevicePair(sdk_root='/unused')
    hand = SimpleNamespace(get_states=lambda: SimpleNamespace(angles=[0.]*22, torques=[0.]*22))
    driver.hands = {'left': hand, 'right': hand}
    callback = driver._tactile_callback('left')
    first, second = TACTILE_CHANNELS['left'][:2]
    callback(dict(channel=first, content={'F6': [1.]*6, 'DEFORM': [1]*(240*240)}))
    snapshot = driver.read()
    assert list(snapshot.left_wrench_valid) == [True, False, False, False, False]
    assert snapshot.left_deformation_valid[0]
    assert not snapshot.right_wrench_valid.any()
    now[0] = 10.2
    callback(dict(channel=second, content={'F6': [2.]*6}))
    callback(dict(channel=first, content={'DEFORM': [2]*(240*240)}))
    now[0] = 10.26
    snapshot = driver.read()
    assert list(snapshot.left_wrench_valid) == [False, True, False, False, False]
    assert list(snapshot.left_deformation_valid) == [True, False, False, False, False]
    assert not snapshot.left_wrench[0].any()
    assert (snapshot.left_wrench[1] == 2).all()
    driver.hands.pop('left')
    snapshot = driver.read()
    assert not snapshot.left_wrench_valid.any() and not snapshot.left_deformation_valid.any()


@pytest.mark.parametrize('bad_force', [[1.]*5, [float('nan')]*6, None, 'bad'])
def test_invalid_tactile_revokes_cached_force_without_losing_deformation(bad_force):
    from ur_common.drivers import TACTILE_CHANNELS
    driver = SharpADevicePair(sdk_root='/unused')
    driver.hands['left'] = SimpleNamespace(get_states=lambda: SimpleNamespace(angles=[0.]*22, torques=[0.]*22))
    callback = driver._tactile_callback('left')
    channel = TACTILE_CHANNELS['left'][0]
    callback(dict(channel=channel, content={'F6': [1.]*6, 'DEFORM': [1]*(240*240)}))
    callback(dict(channel=channel, content={'F6': bad_force}))
    snapshot = driver.read()
    assert not snapshot.left_wrench_valid.any()
    assert snapshot.left_deformation_valid[0]
    assert not snapshot.left_wrench.any()


def connection_driver(hand):
    device = SimpleNamespace(sn='left-sn', ip='192.168.10.10', hand_side=0, device_type=9)
    manager = FakeWaveManager([device])
    def connect(serial, _skip_tactile=False):
        manager.connected[serial] = hand
        return hand
    manager.connect = connect
    driver = SharpADevicePair(sdk_root='/unused', skip_tactile=True)
    driver.sdk = SimpleNamespace(ControlMode=SimpleNamespace(POSITION=13),
                                 ControlSource=SimpleNamespace(SDK=4),
                                 HandSide=SimpleNamespace(LEFT=0))
    driver.manager = manager
    return driver, device


def test_sharpa_connection_does_not_retrigger_ready_mode_or_source():
    hand = FakeConnectHand(mode=13, source=4, enabled=True,
                           fail_redundant_mode_write=True, fail_redundant_source_write=True)
    driver, device = connection_driver(hand)
    driver._connect_device(device, side='left')
    assert driver.connection_error == ''
    assert hand.started
    assert hand.mode_writes == hand.source_writes == 0
    assert not hand.commands


def test_sharpa_connection_configures_coefficients_before_changing_source():
    hand = FakeConnectHand()
    driver, device = connection_driver(hand)
    driver._connect_device(device, side='left')
    assert driver.connection_error == ''
    assert hand.setting_writes == ['speed', 'current', 'source']
    assert hand.mode == 13 and hand.enabled and hand.source == 4
    assert not hand.commands


def test_sharpa_connection_does_not_resend_existing_sdk_source_after_mode_change():
    hand = FakeConnectHand(mode=10, source=4, fail_redundant_source_write=True)
    driver, device = connection_driver(hand)
    driver._connect_device(device, side='left')
    assert driver.connection_error == ''
    assert hand.started and hand.mode_writes == 1
    assert hand.source_writes == 0


def test_sharpa_connection_accepts_float32_coefficient_readbacks_without_rewriting():
    hand = FakeConnectHand()
    hand.speed_coefficient = np.float32(0.5)
    hand.current_coefficient = np.float32(0.6)
    driver, device = connection_driver(hand)
    driver._connect_device(device, side='left')
    assert driver.connection_error == ''
    assert hand.setting_writes == ['source']


def test_sharpa_connection_waits_for_mode_transition_without_retriggering(monkeypatch):
    clock = [0.0]
    hand = FakeConnectHand(mode=10, source=4)
    def mode_readback():
        return SimpleNamespace(code=0, message=''), 13 if hand.mode_writes and clock[0] >= 0.07 else 10
    hand.get_control_mode = mode_readback
    monkeypatch.setattr('ur_common.drivers.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('ur_common.drivers.time.sleep', lambda duration: clock.__setitem__(0, clock[0] + duration))
    driver, device = connection_driver(hand)
    driver._connect_device(device, side='left')
    assert driver.connection_error == ''
    assert clock[0] >= 0.07
    assert hand.started and hand.mode_writes == 1 and hand.source_writes == 0


def test_sharpa_connection_enters_position_when_already_position_but_disabled():
    hand = FakeConnectHand(mode=13, source=4, enabled=False)
    driver, device = connection_driver(hand)
    driver._connect_device(device, side='left')
    assert driver.connection_error == ''
    assert hand.mode_writes == 1 and hand.enabled
    assert hand.source_writes == 0


@pytest.mark.parametrize('setting', ['control_mode', 'control_source', 'speed_coeff', 'current_coeff'])
def test_sharpa_connection_rejects_ignored_setting_without_starting(setting, monkeypatch):
    hand = FakeConnectHand()
    setattr(hand, 'set_' + setting, lambda _: SimpleNamespace(code=0, message=''))
    driver, device = connection_driver(hand)
    clock = [0.0]
    monkeypatch.setattr('ur_common.drivers.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('ur_common.drivers.time.sleep', lambda duration: clock.__setitem__(0, clock[0] + duration))
    driver._connect_device(device, side='left')
    assert 'readback' in driver.connection_error
    assert setting in driver.connection_error
    assert not hand.started
    assert not driver.hands and not driver.manager.connected


def test_sharpa_connection_rejects_failed_initial_read_without_changing_device():
    hand = FakeConnectHand()
    hand.get_control_mode = lambda: (SimpleNamespace(code=2, message='Failure to connect to server'), 13)
    driver, device = connection_driver(hand)
    driver._connect_device(device, side='left')
    assert 'get_control_mode' in driver.connection_error
    assert 'Failure to connect' in driver.connection_error
    assert hand.mode_writes == 0 and not hand.setting_writes and not hand.started


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


def test_strict_sharpa_never_slews_policy_command_and_resets_session():
    from ur_common.command_guards import hand_bounds, ExecutionContractError
    driver = SharpADevicePair(sdk_root='/unused',enable_control=True)
    driver.hands={side:FakeHand() for side in ('left','right')}
    from ur_common.command_guards import firmware_limit_profile
    for side in ('left', 'right'):
        info = SimpleNamespace(**firmware_limit_profile()['hands'][side]['device'])
        driver.hands[side].get_device_info = lambda info=info: info
    lo,hi=hand_bounds();q=np.clip(np.zeros(44,np.float32),lo,hi)
    for i,side in enumerate(('left','right')):driver.hands[side].angles=q[i*22:i*22+22].copy()
    target=q.copy();target[0]=.35
    sent=driver.command(target[:22],target[22:],strict_targets=True,policy_session_id='one')
    np.testing.assert_array_equal(sent['left'],target[:22])
    # Tracking lag is not a request to alter the next target.
    driver.hands['left'].angles=q[:22].copy()
    target[0]=.45
    sent=driver.command(target[:22],target[22:],strict_targets=True,policy_session_id='one')
    assert sent['left'][0] == target[0]
    before=[len(driver.hands[side].commands) for side in ('left','right')]
    target[22+5]=lo[22+5]-.1
    with pytest.raises(ExecutionContractError,match='range_rejected'):
        driver.command(target[:22],target[22:],strict_targets=True,policy_session_id='one')
    assert before == [len(driver.hands[side].commands) for side in ('left','right')]
    target=q.copy();target[0]=1.0
    driver.hands['left'].angles=q[:22].copy()
    with pytest.raises(ExecutionContractError,match='startup_pose_incompatible'):
        driver.command(target[:22],target[22:],strict_targets=True,policy_session_id='two')


def test_strict_sharpa_rejects_unknown_firmware_before_either_hand_moves():
    from ur_common.command_guards import firmware_limit_profile, ExecutionContractError
    driver = SharpADevicePair(sdk_root='/unused', enable_control=True)
    driver.hands = {side: FakeHand() for side in ('left', 'right')}
    for side in ('left', 'right'):
        info = SimpleNamespace(**firmware_limit_profile()['hands'][side]['device'])
        if side == 'right':
            info.firmware_version = 'unknown'
        driver.hands[side].get_device_info = lambda info=info: info
    with pytest.raises(ExecutionContractError, match='firmware_limits_mismatch'):
        driver.command(np.zeros(22), np.zeros(22), strict_targets=True, policy_session_id='one')
    assert all(not hand.commands for hand in driver.hands.values())


def test_ur_side_receipts_preserve_partial_sdk_failure_and_guard_history():
    driver = UrRtdePair(enable_control=True, enable_motion_limits=True, max_joint_step_rad=.35)
    driver.receivers = {side: FakeReceiver() for side in ('left', 'right')}
    driver.controllers = {side: FakeController(driver.receivers[side]) for side in ('left', 'right')}
    def fail(*args):
        raise RuntimeError('right SDK disconnected')
    driver.controllers['right'].servoJ = fail
    events = []
    with pytest.raises(HardwareCommandError, match='right SDK disconnected'):
        driver.command_joint_step(np.full(6, .1), np.full(6, .2), 1/30,
            event_callback=lambda event, **fields: events.append(dict(event=event, **fields)))
    receipts = {row['side']: row for row in events if row['event'] == 'ur_sdk_arm_command_completed'}
    assert receipts['left']['success'] is True and receipts['right']['success'] is False
    assert receipts['left']['target_rad'] == [.1]*6
    assert receipts['right']['target_rad'] == [.2]*6
    assert receipts['right']['completed_timestamp_ns'] >= receipts['right']['command_timestamp_ns']
    context = next(row for row in events if row['event'] == 'ur_guard_context')
    assert context['max_joint_step_rad'] == .35
    assert context['previous_targets'] == dict(left=None, right=None)


@pytest.mark.parametrize("rejected_side", ["left", "right"])
def test_ur_false_servoj_return_is_a_failed_receipt(rejected_side):
    driver = UrRtdePair(enable_control=True)
    driver.receivers = {side: FakeReceiver() for side in ("left", "right")}
    driver.controllers = {
        side: FakeController(driver.receivers[side]) for side in ("left", "right")
    }
    stopped = []
    driver.controllers[rejected_side].servoJ = lambda *_args: False
    driver.controllers[rejected_side].servoStop = lambda: stopped.append(rejected_side)
    events = []
    with pytest.raises(HardwareCommandError, match=f"{rejected_side} UR rejected servoJ"):
        driver.command_joint_step(
            np.full(6, .1), np.full(6, .1), .1,
            event_callback=lambda event, **fields: events.append(dict(event=event, **fields)),
        )
    receipt = next(
        row for row in events
        if row["event"] == "ur_sdk_arm_command_completed" and row["side"] == rejected_side
    )
    assert receipt["success"] is False
    assert "rejected servoJ" in receipt["error"]
    assert stopped == [rejected_side]
    assert driver.last_target_joint[rejected_side] is None
    assert np.array_equal(driver.receivers[rejected_side].joint, np.zeros(6))
