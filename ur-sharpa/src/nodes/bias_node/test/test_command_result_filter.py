import threading
import time
from types import SimpleNamespace

import numpy as np

from bias_node.bias_node import (
    BIAS_PREPARE_STEP,
    BiasNode,
    BiasRun,
    PHASE_UR,
    REQUEST_HOME_BOTH,
    REQUEST_HOME_UR,
)
from bias_node.bias_node import PHASE_SHARPA, REQUEST_HOME_SHARPA
from ur_msgs.msg import ArmCommand


def _active_node():
    node = object.__new__(BiasNode)
    node._lock = threading.RLock()
    node._counts = {"nacks": 0}
    node._last_nack = ""
    node._run = BiasRun(
        request=REQUEST_HOME_UR,
        started_at=time.monotonic(),
        phase=PHASE_UR,
    )
    failures = []
    node._fail_locked = failures.append
    return node, failures


def _nack(source):
    return SimpleNamespace(
        source=source,
        success=False,
        device="ur",
        failure_code="arbitration_rejected",
        failure_message="command source is not active",
    )


def test_vive_nack_does_not_fail_active_bias_run():
    node, failures = _active_node()

    node._on_command_result(_nack(ArmCommand.SOURCE_VIVE))

    assert failures == []
    assert node._counts["nacks"] == 0
    assert node._run.phase == PHASE_UR


def test_bias_nack_fails_active_bias_run():
    node, failures = _active_node()

    node._on_command_result(_nack(ArmCommand.SOURCE_BIAS))

    assert failures == [
        "ur rejected a bias step: arbitration_rejected command source is not active"
    ]
    assert node._counts["nacks"] == 1


def test_sharpa_reset_targets_saved_bias_from_nonzero_current_pose():
    node = object.__new__(BiasNode)
    node.sharpa_steps = 4
    node._measured_hand_joints = lambda: np.full(44, 0.7)
    saved_bias = np.linspace(-0.2, 0.6, 44)
    node._load_hand_bias_target_locked = lambda: saved_bias
    run = BiasRun(
        request=REQUEST_HOME_SHARPA,
        started_at=time.monotonic(),
        phase=PHASE_SHARPA,
    )

    node._build_plans_locked(run)

    np.testing.assert_array_equal(run.hand_target, saved_bias)
    np.testing.assert_allclose(run.hand_plan[-1], saved_bias, atol=1e-15)


def test_both_reset_waits_for_sharpa_preflight_before_moving_arms():
    node = object.__new__(BiasNode)
    node.arrival_timeout_s = 3.0
    node._counts = {"arm_steps": 0, "hand_steps": 0}
    arm_steps = []
    hand_steps = []
    node._publish_arm_step = lambda left, right, step: arm_steps.append(step)
    node._publish_hand_step = lambda target, step: hand_steps.append(step)
    run = BiasRun(
        request=REQUEST_HOME_BOTH,
        started_at=time.monotonic(),
        phase=PHASE_UR,
        arm_plan=np.zeros((2, 12)),
        hand_plan=np.zeros((2, 44)),
        hand_prepare_target=np.full(44, 0.3),
    )

    node._stream_step_locked(run)

    assert arm_steps == []
    assert hand_steps == [BIAS_PREPARE_STEP]
    assert run.step == 0


def test_sharpa_preflight_ack_unlocks_both_reset_stream():
    node, _failures = _active_node()
    node._run = BiasRun(
        request=REQUEST_HOME_BOTH,
        started_at=time.monotonic(),
        phase=PHASE_UR,
        arm_plan=np.zeros((2, 12)),
        hand_plan=np.zeros((2, 44)),
        hand_prepare_target=np.zeros(44),
        hand_prepare_sent_at=time.monotonic(),
    )
    ack = SimpleNamespace(
        source=ArmCommand.SOURCE_BIAS,
        success=True,
        device="sharpa",
        step_index=BIAS_PREPARE_STEP,
        failure_code="",
        failure_message="",
    )

    node._on_command_result(ack)

    assert node._run.hand_ready is True
