import threading
from types import SimpleNamespace

import numpy as np
import pytest
from rclpy.clock import Clock

from ur_common.drivers import UrArmDiagnostic, UrPairSnapshot
from ur_msgs.msg import ArmCommand
from ur_node.ur_node import UrNode


class FakeDriver:
    def read(self):
        return SimpleNamespace(
            left_joint=np.zeros(6, dtype=np.float64),
            right_joint=np.zeros(6, dtype=np.float64),
        )


def test_resolve_targets_rejects_before_joint_clipping():
    node = UrNode.__new__(UrNode)
    node.enable_motion_limits = True
    node.enable_joint_step_limit = True
    node.enable_joint_clip = True
    node.max_joint_step_rad = 0.35
    node.max_abs_joint_rad = 6.28
    node.driver = FakeDriver()
    message = SimpleNamespace(
        left_joint=np.full(6, 1.2),
        right_joint=np.full(6, 1.2),
        left_enable=True,
        right_enable=True,
    )

    with pytest.raises(ValueError, match="left joint target step exceeds"):
        node.resolve_targets(message)


def test_resolve_targets_passes_through_when_joint_step_and_clip_are_disabled():
    node = UrNode.__new__(UrNode)
    node.enable_motion_limits = True
    node.enable_joint_step_limit = False
    node.enable_joint_clip = False
    node.max_joint_step_rad = 0.35
    node.max_abs_joint_rad = 6.28
    node.driver = FakeDriver()
    message = SimpleNamespace(
        left_joint=np.full(6, 1.2),
        right_joint=np.full(6, -1.2),
        left_enable=True,
        right_enable=True,
    )

    left, right = node.resolve_targets(message)

    np.testing.assert_array_equal(left, message.left_joint)
    np.testing.assert_array_equal(right, message.right_joint)


def test_resolve_targets_rejects_absolute_joint_limit():
    node = UrNode.__new__(UrNode)
    node.enable_motion_limits = True
    node.enable_joint_step_limit = False
    node.enable_joint_clip = False
    node.max_joint_step_rad = 0.35
    node.max_abs_joint_rad = 1.0
    node.driver = FakeDriver()
    message = SimpleNamespace(
        left_joint=[1.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        right_joint=np.zeros(6),
        left_enable=True,
        right_enable=False,
    )

    with pytest.raises(ValueError, match="absolute limit"):
        node.resolve_targets(message)


def test_safety_signature_ignores_live_wrench_changes():
    def diagnostic(wrench):
        return SimpleNamespace(
            normal_mode=False,
            safety_mode=3,
            safety_status_bits=(1 << 2) | (1 << 10),
            robot_mode=7,
            protective_stopped=True,
            emergency_stopped=False,
            probe_errors=(),
            tcp_wrench=np.asarray(wrench),
        )

    first = SimpleNamespace(
        left_diagnostic=diagnostic([1, 2, 3, 4, 5, 6]),
        right_diagnostic=diagnostic([6, 5, 4, 3, 2, 1]),
    )
    second = SimpleNamespace(
        left_diagnostic=diagnostic([10, 20, 30, 40, 50, 60]),
        right_diagnostic=diagnostic([60, 50, 40, 30, 20, 10]),
    )

    assert UrNode._safety_signature(first) == UrNode._safety_signature(second)


class FakeLogger:
    def __init__(self):
        self.info_messages = []
        self.error_messages = []

    def info(self, message):
        self.info_messages.append(str(message))

    def error(self, message):
        self.error_messages.append(str(message))


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def diagnostic(*, normal, protective=False, safety_mode=None):
    return UrArmDiagnostic(
        normal_mode=normal,
        safety_mode=(1 if normal else safety_mode),
        safety_status_bits=(1 if normal else (1 << 2)),
        robot_mode=7,
        protective_stopped=protective,
        emergency_stopped=False,
        payload_mass_kg=1.3,
        payload_cog_m=np.asarray([0.0, 0.0, 0.08]),
        tcp_wrench=np.zeros(6),
        probe_errors=(),
    )


def snapshot(*, normal):
    left = diagnostic(
        normal=normal, protective=not normal, safety_mode=3 if not normal else 1
    )
    right = diagnostic(normal=True)
    return UrPairSnapshot(
        timestamp_ns=123,
        left_joint=np.linspace(-0.3, 0.2, 6).astype(np.float32),
        right_joint=np.linspace(0.4, -0.1, 6).astype(np.float32),
        left_rtde_pose=np.zeros(6),
        right_rtde_pose=np.zeros(6),
        normal_mode=normal,
        left_diagnostic=left,
        right_diagnostic=right,
    )


class RecoveryDriver:
    def __init__(self, samples):
        self.samples = list(samples)
        self.events = []

    def close(self):
        self.events.append("close")

    def connect(self):
        self.events.append("connect")

    def read(self):
        self.events.append("read")
        return self.samples.pop(0)

    def synchronize_command_state(self, current):
        self.events.append("synchronize")
        self.synchronized = current

    def safe_stop(self):
        self.events.append("safe_stop")


class FakeKinematics:
    def arm_fk_eef9(self, _joint, _side):
        return np.zeros(9, dtype=np.float32)


def bare_recovery_node(driver, *, stopped=False):
    node = UrNode.__new__(UrNode)
    node.driver = driver
    node.driver_error = "reconnect requested"
    node.last_command_message = "old command state"
    node.stopped = bool(stopped)
    node._driver_lock = threading.RLock()
    node._reconnect_lock = threading.Lock()
    node._reconnecting = True
    node._rtde_recovery_needed = True
    node._rtde_recovery_ready = True
    node._destroying = False
    logger = FakeLogger()
    node.get_logger = lambda: logger
    return node, logger


def test_rtde_reconnect_requires_normal_sample_before_command_sync():
    driver = RecoveryDriver([snapshot(normal=True)])
    node, logger = bare_recovery_node(driver)

    node._reconnect()

    assert driver.events == ["close", "connect", "read", "synchronize"]
    assert driver.synchronized.normal_mode is True
    assert node.driver_error == ""
    assert node._rtde_recovery_needed is False
    assert node._rtde_recovery_ready is False
    assert node._reconnecting is False
    assert logger.info_messages


def test_rtde_reconnect_keeps_gate_closed_when_controller_is_not_normal():
    driver = RecoveryDriver([snapshot(normal=False)])
    node, _logger = bare_recovery_node(driver, stopped=True)

    node._reconnect()

    assert driver.events == ["close", "connect", "safe_stop", "read"]
    assert node.driver_error == ""
    assert node._rtde_recovery_needed is True
    assert node._rtde_recovery_ready is False
    assert node._reconnecting is False
    assert "waiting for NORMAL" in node.last_command_message


def test_protective_stop_waits_for_normal_before_scheduling_reconnect():
    driver = RecoveryDriver([snapshot(normal=False), snapshot(normal=True)])
    node = UrNode.__new__(UrNode)
    node.driver = driver
    node.driver_error = ""
    node.last_command_acked = True
    node.last_command_message = ""
    node._driver_lock = threading.RLock()
    node._reconnecting = False
    node._rtde_recovery_needed = False
    node._rtde_recovery_ready = False
    node._destroying = False
    node._last_safety_signature = None
    node.kinematics = FakeKinematics()
    node.state_pub = FakePublisher()
    node.get_clock = lambda: Clock()
    logger = FakeLogger()
    node.get_logger = lambda: logger
    reconnect_requests = []
    node.schedule_reconnect = lambda: reconnect_requests.append(True)

    node.publish_state()

    assert node._rtde_recovery_needed is True
    assert node._rtde_recovery_ready is False
    assert reconnect_requests == []
    assert node.state_pub.messages[-1].normal_mode is False
    assert "PROTECTIVE_STOP" in node.state_pub.messages[-1].status_message

    node.publish_state()

    assert node._rtde_recovery_needed is True
    assert node._rtde_recovery_ready is True
    assert reconnect_requests == [True]
    assert node.state_pub.messages[-1].normal_mode is True


def test_publish_state_during_reconnect_does_not_create_persistent_error():
    node = UrNode.__new__(UrNode)
    node.driver_error = ""
    node.last_command_acked = True
    node._reconnecting = True
    node.state_pub = FakePublisher()
    node.get_clock = lambda: Clock()

    node.publish_state()

    assert node.driver_error == ""
    assert len(node.state_pub.messages) == 1
    message = node.state_pub.messages[0]
    assert message.valid is False
    assert message.normal_mode is False
    assert message.status_message == "UR RTDE reconnecting"


def test_command_is_rejected_while_rtde_recovery_gate_is_closed():
    node = UrNode.__new__(UrNode)
    node.stopped = False
    node.driver_error = ""
    node._reconnecting = False
    node._rtde_recovery_needed = True
    node.result_pub = FakePublisher()
    node.get_clock = lambda: Clock()
    node.last_command_acked = True
    node.last_command_message = ""
    message = ArmCommand()

    node.on_command(message)

    assert len(node.result_pub.messages) == 1
    result = node.result_pub.messages[0]
    assert result.success is False
    assert result.failure_code == "ur_recovering"


def test_actor_dry_run_never_calls_ur_command_write():
    node = UrNode.__new__(UrNode)
    node._driver_lock = threading.RLock()
    writes = []
    node.driver = SimpleNamespace(
        command_joint_step=lambda *_args, **_kwargs: writes.append(True)
    )
    node.resolve_targets = lambda _message: (np.zeros(6), np.zeros(6))
    results = []
    node.publish_result = (
        lambda _message, success, code, detail: results.append(
            (success, code, detail)
        )
    )
    message = SimpleNamespace(
        dry_run=True,
        source=ArmCommand.SOURCE_ACTOR,
        period_s=1.0 / 30.0,
    )

    node.on_command(message)

    assert writes == []
    assert results == [(True, "", "")]
