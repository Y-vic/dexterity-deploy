import threading
import time
from types import SimpleNamespace

import numpy as np

from bias_node.bias_node import (
    BiasNode,
    BiasRun,
    PHASE_FAILED,
    PHASE_PREPARING,
    PHASE_UR,
    REQUEST_HOME_BOTH,
    REQUEST_HOME_SHARPA,
)
from ur_msgs.msg import ArmCommand, ControlStatus


def _run(*, request=REQUEST_HOME_BOTH, steps=2):
    return BiasRun(
        request=request,
        action_id="reset-current",
        started_at=time.monotonic(),
        phase=PHASE_UR,
        arm_plan=np.zeros((steps, 12)),
        hand_plan=np.zeros((steps, 44)),
    )


def _stream_node(run):
    node = object.__new__(BiasNode)
    node._lock = threading.RLock()
    node._run = run
    node._counts = {"arm_steps": 0, "hand_steps": 0, "nacks": 0}
    node._last_nack = ""
    node.ur_step_delay_s = 0.02
    node.sharpa_step_delay_s = 0.02
    node.arrival_timeout_s = 3.0
    node.command_ack_timeout_s = 1.0
    node._tick_period = 0.02
    node.arm_publications = []
    node.hand_publications = []
    node._publish_arm_step = lambda left, right, step, action: node.arm_publications.append(
        (step, action)
    )
    node._publish_hand_step = lambda target, step, action: node.hand_publications.append(
        (step, action)
    )
    node.failures = []

    def fail(detail, *_args, **_kwargs):
        run.phase = PHASE_FAILED
        run.pending_step = None
        run.pending_devices.clear()
        node.failures.append(detail)

    node._fail_locked = fail
    return node


def _result(*, device, success=True, action_id="reset-current", step=0, source=None):
    return SimpleNamespace(
        source=ArmCommand.SOURCE_BIAS if source is None else source,
        action_id=action_id,
        revision=0,
        step_index=step,
        success=success,
        device=device,
        failure_code="sdk_failed" if not success else "",
        failure_message="device rejected command" if not success else "",
    )


def test_paired_step_does_not_advance_until_ur_and_sharpa_ack():
    run = _run()
    node = _stream_node(run)

    node._stream_step_locked(run)

    assert node.arm_publications == [(0, "reset-current")]
    assert node.hand_publications == [(0, "reset-current")]
    assert run.step == 0
    assert run.pending_devices == {"ur", "sharpa"}

    node._on_command_result(_result(device="ur"))
    assert run.step == 0
    assert run.pending_devices == {"sharpa"}

    node._on_command_result(_result(device="sharpa"))
    assert run.step == 1
    assert run.pending_step is None


def test_sharpa_nack_fails_pair_before_ur_can_advance_again():
    run = _run()
    node = _stream_node(run)
    node._stream_step_locked(run)
    node._on_command_result(_result(device="ur"))

    node._on_command_result(_result(device="sharpa", success=False))

    assert run.phase == PHASE_FAILED
    assert run.step == 0
    assert len(node.arm_publications) == 1
    assert node._counts["nacks"] == 1
    assert "sharpa rejected bias step 0" in node.failures[0]


def test_old_run_ack_and_non_bias_result_are_ignored():
    run = _run()
    node = _stream_node(run)
    node._stream_step_locked(run)

    node._on_command_result(_result(device="ur", action_id="reset-old"))
    node._on_command_result(_result(device="ur", source=ArmCommand.SOURCE_VIVE))

    assert run.step == 0
    assert run.pending_devices == {"ur", "sharpa"}
    assert node.failures == []


def test_missing_ack_times_out_without_publishing_another_step():
    run = _run()
    node = _stream_node(run)
    node._stream_step_locked(run)
    run.pending_since = time.monotonic() - 2.0

    node._stream_step_locked(run)

    assert run.phase == PHASE_FAILED
    assert len(node.arm_publications) == 1
    assert len(node.hand_publications) == 1
    assert "ACK timeout" in node.failures[0]


def test_prepare_waits_for_new_position_ready_frame_before_building_plans():
    started = time.monotonic()
    run = BiasRun(
        request=REQUEST_HOME_BOTH,
        action_id="reset-current",
        started_at=started,
        phase=PHASE_PREPARING,
    )
    node = object.__new__(BiasNode)
    node._run = run
    node.prepare_timeout_s = 5.0
    node._ur_active_state = int(ControlStatus.UR_HOMING)
    node._sharpa_active_state = int(ControlStatus.SHARPA_HOMING)
    node._ur_state_time = started + 0.01
    node._hand_state_time = started + 0.01
    node._ur_state = SimpleNamespace(timestamp_ns=run.started_wall_ns + 1)
    node._hand_state = SimpleNamespace(
        timestamp_ns=run.started_wall_ns + 1,
        control_error="",
        position_control_ready=False,
    )
    node._measured_arm_joints = lambda: {"left": np.zeros(6), "right": np.zeros(6)}
    node._measured_hand_joints = lambda: np.zeros(44)
    built = []

    def build(candidate):
        built.append(True)
        candidate.arm_plan = np.zeros((2, 12))
        candidate.hand_plan = np.zeros((2, 44))

    node._build_plans_locked = build
    node._fail_locked = lambda detail: (_ for _ in ()).throw(AssertionError(detail))
    node.get_logger = lambda: SimpleNamespace(info=lambda *_args: None)

    node._prepare_locked(run)
    assert built == []
    assert run.phase == PHASE_PREPARING

    node._hand_state.position_control_ready = True
    node._prepare_locked(run)
    assert built == [True]
    assert run.phase == PHASE_UR


def test_sharpa_reset_targets_saved_bias_from_nonzero_current_pose():
    node = object.__new__(BiasNode)
    node.sharpa_steps = 4
    node._measured_hand_joints = lambda: np.full(44, 0.7)
    saved_bias = np.linspace(-0.2, 0.6, 44)
    node._load_hand_bias_target_locked = lambda: saved_bias
    run = BiasRun(
        request=REQUEST_HOME_SHARPA,
        action_id="reset-current",
        started_at=time.monotonic(),
        phase=PHASE_PREPARING,
    )

    node._build_plans_locked(run)

    np.testing.assert_array_equal(run.hand_target, saved_bias)
    np.testing.assert_allclose(run.hand_plan[-1], saved_bias, atol=1e-15)
