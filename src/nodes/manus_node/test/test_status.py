"""Exercise status publication without starting ROS or vendor hardware."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from manus_node import ManusHealthTracker


def publish_status(health, now, *, error="", client_code=None):
    source = Path(__file__).parents[1] / "manus_node/manus_node.py"
    method = next(
        node for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_publish_status"
    )
    context = {
        "time": SimpleNamespace(monotonic=lambda: now),
        "String": SimpleNamespace,
        "json": json,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), context)
    messages = []
    node = SimpleNamespace(
        _health=health,
        _process=SimpleNamespace(poll=lambda: client_code),
        _retarget=SimpleNamespace(poll=lambda: None),
        _hardware_started=True,
        _hardware_requested=True,
        _hardware_error=error,
        _last_action_monotonic=health.last_action,
        _last_side_action_monotonic={"left": health.last_action, "right": 0.0},
        get_parameter=lambda name: SimpleNamespace(value=False),
        status=SimpleNamespace(publish=messages.append),
    )
    context["_publish_status"](node)
    return json.loads(messages[0].data)


def test_live_processes_without_frames_report_raw_timeout_after_grace():
    health = ManusHealthTracker(startup_grace_s=8.0)
    health.reset(10.0)
    assert publish_status(health, 17.9)["error"] == ""
    status = publish_status(health, 18.1)
    assert status["client_alive"] and status["retarget_alive"]
    assert status["mocap_age_s"] is None
    assert not status["valid"]
    assert "2044" in status["error"]


def test_raw_frames_without_retarget_output_report_action_timeout():
    health = ManusHealthTracker(startup_grace_s=0.0)
    health.reset(10.0)
    health.observe_raw(25, 25, 11.0)
    assert "6668" in publish_status(health, 11.1)["error"]


def test_missing_glove_is_reported_without_disabling_healthy_side():
    health = ManusHealthTracker(startup_grace_s=0.0)
    health.reset(10.0)
    health.observe_raw(25, 0, 10.1)
    health.observe_raw(25, 0, 11.2)
    health.observe_action(11.2)
    status = publish_status(health, 11.3)
    assert status["valid"] and status["left_valid"]
    assert not status["right_valid"]
    assert "right glove" in status["error"]
    health.observe_raw(25, 25, 11.4)
    health.observe_action(11.4)
    assert publish_status(health, 11.5)["error"] == ""


@pytest.mark.parametrize(
    "error,client_code,expected",
    [("decode failed", None, "decode failed"), ("", 1, "Manus client exited with code 1")],
)
def test_specific_errors_keep_precedence(error, client_code, expected):
    health = ManusHealthTracker(startup_grace_s=0.0)
    health.reset(10.0)
    assert publish_status(health, 20.0, error=error, client_code=client_code)["error"] == expected
