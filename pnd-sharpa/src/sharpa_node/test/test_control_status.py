import json

import pytest

from sharpa_node.common import MODE_DAMPING, MODE_TELEOP, MODE_ZERO, parse_control_status


@pytest.mark.parametrize("state", ["t_init", "t_adam"])
def test_inactive_teleop_states_command_sharpa_zero(state):
    status = parse_control_status(json.dumps({"state": state}))

    assert status.mode == MODE_ZERO
    assert status.teleop_state == state
    assert not status.sharpa_active
    assert status.known


@pytest.mark.parametrize("state", ["t_init_sharpa", "t_adam_sharpa"])
def test_active_teleop_states_forward_sharpa_targets(state):
    status = parse_control_status(json.dumps({"state": state}))

    assert status.mode == MODE_TELEOP
    assert status.teleop_state == state
    assert status.sharpa_active
    assert status.known


def test_robot_damping_keeps_sharpa_in_damping_mode():
    status = parse_control_status(json.dumps({"state": "damping"}))

    assert status.mode == MODE_DAMPING
    assert not status.sharpa_active
    assert status.known
