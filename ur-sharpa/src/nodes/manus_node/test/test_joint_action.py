from types import SimpleNamespace

import numpy as np
import pytest

from manus_node import ManusCore


def hand(side, offset=0.0):
    return SimpleNamespace(
        name=[f"{side}_{name.upper()}" for name in ManusCore.JOINT_NAMES],
        position=[offset + index for index in range(len(ManusCore.JOINT_NAMES))],
    )


def test_joint_action_accepts_side_prefixed_protobuf_names():
    action = SimpleNamespace(joint_left=hand("left"), joint_right=hand("right", 100.0))

    result = ManusCore.joint_action_to_q(action)

    np.testing.assert_array_equal(result[:22], np.arange(22, dtype=np.float32))
    np.testing.assert_array_equal(result[22:], np.arange(100, 122, dtype=np.float32))


def test_joint_action_rejects_missing_joint_instead_of_publishing_zero():
    left = hand("left")
    left.name.pop()
    left.position.pop()
    action = SimpleNamespace(joint_left=left, joint_right=hand("right"))

    with pytest.raises(ValueError, match="left joints missing: pinky_dip"):
        ManusCore.joint_action_to_q(action)


def test_one_malformed_side_does_not_prevent_parsing_the_other():
    right = hand("right")
    right.name.clear()
    right.position.clear()
    action = SimpleNamespace(joint_left=hand("left"), joint_right=right)

    left = ManusCore.joint_action_side_to_q(action, "left")

    np.testing.assert_array_equal(left, np.arange(22, dtype=np.float32))
    with pytest.raises(ValueError, match="right joints missing"):
        ManusCore.joint_action_side_to_q(action, "right")


def test_workbench_manus_routes_by_authoritative_owner():
    from types import SimpleNamespace
    import ast
    from pathlib import Path
    source = Path(__file__).parents[1] / 'manus_node/manus_node.py'
    method = next(node for node in ast.walk(ast.parse(source.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == '_control')
    context = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), context)
    node=SimpleNamespace(_direct_pub='direct',_candidate_pub='candidate')
    for source, expected in ((2,'direct'),(3,'candidate'),(0,'candidate')):
        context['_control'](node,SimpleNamespace(active_source=source))
        assert node.pub==expected
