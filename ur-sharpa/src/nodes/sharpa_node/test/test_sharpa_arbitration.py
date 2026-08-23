import numpy as np
import pytest

from sharpa_node.arbitration import SharpaState, Source, arbitrate
from sharpa_node.sharpa_node import SuccessfulHandCommandCache


@pytest.mark.parametrize("state", list(SharpaState))
def test_bias_is_available_for_homing_and_reset(state):
    assert arbitrate(state, Source.ACTOR, Source.BIAS)


@pytest.mark.parametrize("source", [Source.VIVE, Source.ACTOR])
def test_streaming_sources_are_blocked_while_stopped(source):
    assert not arbitrate(SharpaState.STOPPED, source, source)


def test_running_accepts_only_active_streaming_source():
    assert arbitrate(SharpaState.RUNNING, Source.ACTOR, Source.ACTOR)
    assert not arbitrate(SharpaState.RUNNING, Source.ACTOR, Source.VIVE)


def test_unknown_source_is_rejected():
    assert not arbitrate(SharpaState.RUNNING, Source.UNKNOWN, Source.UNKNOWN)


def test_successful_hand_command_cache_keeps_fresh_full_command():
    cache = SuccessfulHandCommandCache(freshness_timeout_s=0.25)
    cache.record(
        {"left": np.full(22, 0.2), "right": np.full(22, -0.3)},
        source=2,
        timestamp_ns=1_000_000_000,
    )
    q_cmd, valid, timestamp_ns, source = cache.snapshot(1_200_000_000)
    np.testing.assert_allclose(q_cmd[:22], 0.2)
    np.testing.assert_allclose(q_cmd[22:], -0.3)
    assert valid is True
    assert timestamp_ns == 1_000_000_000
    assert source == 2
    assert cache.snapshot(1_300_000_000)[1] is False


def test_partial_hand_command_is_not_a_valid_full_action():
    cache = SuccessfulHandCommandCache()
    cache.record(
        {"left": np.full(22, 0.2)},
        source=2,
        timestamp_ns=1_000_000_000,
    )
    assert cache.snapshot(1_000_000_000)[1] is False


def test_two_fresh_independent_hand_commands_form_one_full_action():
    cache = SuccessfulHandCommandCache()
    cache.record({"left": np.full(22, 0.2)}, source=2, timestamp_ns=1_000_000_000)
    cache.record({"right": np.full(22, -0.3)}, source=2, timestamp_ns=1_010_000_000)

    q_cmd, valid, timestamp_ns, source = cache.snapshot(1_020_000_000)

    assert valid is True
    np.testing.assert_allclose(q_cmd[:22], 0.2)
    np.testing.assert_allclose(q_cmd[22:], -0.3)
    assert timestamp_ns == 1_010_000_000
    assert source == 2
