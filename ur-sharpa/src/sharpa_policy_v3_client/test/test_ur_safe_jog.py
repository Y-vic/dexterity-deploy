import math
from types import SimpleNamespace

import pytest

import sharpa_policy_v3_client.ur_safe_jog as ur_safe_jog
from sharpa_policy_v3_client.ur_safe_jog import (
    JogRequest,
    RobotSnapshot,
    build_target_pose,
    validate_execution_snapshot,
)


def request(**overrides):
    values = {
        "side": "right",
        "robot_ip": "192.168.56.10",
        "axis": "z",
        "distance_mm": 2.0,
    }
    values.update(overrides)
    return JogRequest(**values)


def snapshot(**overrides):
    values = {
        "actual_q": (0.0,) * 6,
        "tcp_pose": (0.1, -0.2, 0.3, 0.0, 0.0, 0.0),
        "robot_mode": 7,
        "safety_mode": 1,
        "runtime_state": 1,
    }
    values.update(overrides)
    return RobotSnapshot(**values)


def test_request_builds_explicit_confirmation_token():
    assert request().confirmation_token == "right:z:+2mm"
    assert request(distance_mm=-2.5).confirmation_token == "right:z:-2.5mm"


@pytest.mark.parametrize(
    "overrides",
    [
        {"side": "center"},
        {"robot_ip": "not-an-ip"},
        {"robot_ip": "::1"},
        {"axis": "roll"},
        {"distance_mm": 0.0},
        {"distance_mm": 5.01},
        {"distance_mm": math.nan},
        {"speed_m_s": 0.0},
        {"speed_m_s": 0.021},
        {"acceleration_m_s2": 0.0},
        {"acceleration_m_s2": 0.101},
    ],
)
def test_request_rejects_unsafe_values(overrides):
    with pytest.raises(ValueError):
        request(**overrides)


@pytest.mark.parametrize(
    ("axis", "distance_mm", "expected_position"),
    [
        ("x", 2.0, (0.102, -0.2, 0.3)),
        ("y", -3.0, (0.1, -0.203, 0.3)),
        ("z", 5.0, (0.1, -0.2, 0.305)),
    ],
)
def test_build_target_pose_changes_one_base_axis(
    axis,
    distance_mm,
    expected_position,
):
    target = build_target_pose(
        (0.1, -0.2, 0.3, 0.4, 0.5, 0.6),
        request(axis=axis, distance_mm=distance_mm),
    )
    assert target[:3] == pytest.approx(expected_position)
    assert target[3:] == pytest.approx((0.4, 0.5, 0.6))


def test_build_target_pose_rejects_invalid_pose():
    with pytest.raises(ValueError, match="exactly 6"):
        build_target_pose((0.0,) * 5, request())
    with pytest.raises(ValueError, match="finite"):
        build_target_pose((0.0, 0.0, math.inf, 0.0, 0.0, 0.0), request())


def test_execution_snapshot_accepts_stopped_normal_robot():
    validate_execution_snapshot(snapshot())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"safety_mode": 3}, "safety mode"),
        ({"robot_mode": 5}, "robot mode"),
        ({"runtime_state": 2}, "STOPPED"),
    ],
)
def test_execution_snapshot_rejects_unsafe_state(overrides, match):
    with pytest.raises(RuntimeError, match=match):
        validate_execution_snapshot(snapshot(**overrides))


class FakeReceiver:
    def __init__(self, robot_ip, frequency):
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.disconnected = False

    def isConnected(self):
        return True

    def getActualQ(self):
        return [0.0] * 6

    def getActualTCPPose(self):
        return [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]

    def getRobotMode(self):
        return 7

    def getSafetyMode(self):
        return 1

    def getRuntimeState(self):
        return 1

    def disconnect(self):
        self.disconnected = True


class UnexpectedControl:
    FLAG_UPLOAD_SCRIPT = 1

    def __init__(self, *args):
        raise AssertionError("dry-run must not construct the control interface")


def test_main_is_read_only_by_default(monkeypatch, capsys):
    monkeypatch.setattr(
        ur_safe_jog,
        "_load_rtde",
        lambda: (
            SimpleNamespace(RTDEReceiveInterface=FakeReceiver),
            SimpleNamespace(RTDEControlInterface=UnexpectedControl),
        ),
    )

    result = ur_safe_jog.main(
        ["--side", "right", "--axis", "z", "--distance-mm", "2"]
    )

    assert result == 0
    assert '"dry_run":true' in capsys.readouterr().out


def test_main_rejects_execute_without_exact_confirmation(monkeypatch):
    monkeypatch.setattr(
        ur_safe_jog,
        "_load_rtde",
        lambda: pytest.fail("invalid confirmation must fail before RTDE is loaded"),
    )

    with pytest.raises(SystemExit) as error:
        ur_safe_jog.main(
            [
                "--side",
                "right",
                "--axis",
                "z",
                "--distance-mm",
                "2",
                "--execute",
                "--confirm",
                "MOVE",
            ]
        )

    assert error.value.code == 2
