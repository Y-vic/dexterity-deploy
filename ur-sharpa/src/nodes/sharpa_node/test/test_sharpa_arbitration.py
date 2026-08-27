import numpy as np
import pytest

from sharpa_node.arbitration import SharpaState, Source, arbitrate
from sharpa_node.sharpa_node import (
    SuccessfulHandCommandCache,
    validate_hand_command,
)


@pytest.mark.parametrize("state", [SharpaState.HOMING, SharpaState.HOMED])
def test_bias_is_available_only_for_homing_and_homed(state):
    assert arbitrate(state, Source.ACTOR, Source.BIAS)


@pytest.mark.parametrize(
    "state", [SharpaState.STOPPED, SharpaState.STARTING, SharpaState.RUNNING]
)
def test_bias_is_rejected_outside_reset_states(state):
    assert not arbitrate(state, Source.ACTOR, Source.BIAS)


@pytest.mark.parametrize("source", [Source.VIVE, Source.ACTOR])
def test_streaming_sources_are_blocked_while_stopped(source):
    assert not arbitrate(SharpaState.STOPPED, source, source)
    assert not arbitrate(SharpaState.HOMING, source, source)
    assert not arbitrate(SharpaState.HOMED, source, source)
    assert not arbitrate(SharpaState.STARTING, source, source)


def test_running_accepts_only_active_streaming_source():
    assert arbitrate(SharpaState.RUNNING, Source.ACTOR, Source.ACTOR)
    assert not arbitrate(SharpaState.RUNNING, Source.ACTOR, Source.VIVE)


def test_unknown_source_is_rejected():
    assert not arbitrate(SharpaState.RUNNING, Source.UNKNOWN, Source.UNKNOWN)


def test_successful_hand_command_cache_holds_full_command():
    cache = SuccessfulHandCommandCache()
    cache.record(
        {"left": np.full(22, 0.2), "right": np.full(22, -0.3)},
        source=2,
        timestamp_ns=1_000_000_000,
    )
    q_cmd, valid, timestamp_ns, source = cache.snapshot()
    np.testing.assert_allclose(q_cmd[:22], 0.2)
    np.testing.assert_allclose(q_cmd[22:], -0.3)
    assert valid is True
    assert timestamp_ns == 1_000_000_000
    assert source == 2
    # Like pnd-sharpa, the most recent successful SDK target is a zero-order
    # hold until an execution-state transition explicitly invalidates it.
    assert cache.snapshot()[1] is True


def test_partial_hand_command_is_not_a_valid_full_action():
    cache = SuccessfulHandCommandCache()
    cache.record(
        {"left": np.full(22, 0.2)},
        source=2,
        timestamp_ns=1_000_000_000,
    )
    q_cmd, valid, timestamp_ns, source = cache.snapshot()
    assert valid is False
    assert np.isnan(q_cmd).all()
    assert timestamp_ns == 0
    assert source == 0


def test_two_independent_hand_commands_do_not_form_one_full_action():
    cache = SuccessfulHandCommandCache()
    cache.record({"left": np.full(22, 0.2)}, source=2, timestamp_ns=1_000_000_000)
    cache.record({"right": np.full(22, -0.3)}, source=2, timestamp_ns=1_010_000_000)

    q_cmd, valid, timestamp_ns, source = cache.snapshot()

    assert valid is False
    assert np.isnan(q_cmd).all()
    assert timestamp_ns == 0
    assert source == 0


def test_invalidate_clears_the_held_successful_command():
    cache = SuccessfulHandCommandCache()
    cache.record(
        {"left": np.full(22, 0.2), "right": np.full(22, -0.3)},
        source=2,
        timestamp_ns=1_000_000_000,
    )

    cache.invalidate()

    q_cmd, valid, timestamp_ns, source = cache.snapshot()
    assert valid is False
    assert np.isnan(q_cmd).all()
    assert timestamp_ns == 0
    assert source == 0


def test_actor_dry_run_hand_command_is_validated_without_sdk_state():
    command = type(
        "Command",
        (),
        {
            "source": 3,
            "q_cmd": [0.0] * 44,
            "left_enable": True,
            "right_enable": True,
            "dry_run": True,
        },
    )()

    values, source, left_enable, right_enable, dry_run = validate_hand_command(command)

    assert values.shape == (44,)
    assert source == 3
    assert left_enable and right_enable and dry_run


def test_non_actor_cannot_request_sharpa_dry_run():
    command = type(
        "Command",
        (),
        {
            "source": 2,
            "q_cmd": [0.0] * 44,
            "left_enable": True,
            "right_enable": True,
            "dry_run": True,
        },
    )()

    with pytest.raises(ValueError, match="only Actor"):
        validate_hand_command(command)
