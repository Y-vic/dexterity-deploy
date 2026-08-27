import pytest

from ur_node.arbitration import Source, UrState, arbitrate


@pytest.mark.parametrize("state", [UrState.HOMING, UrState.HOMED])
def test_bias_is_available_only_for_homing_and_homed(state):
    assert arbitrate(state, Source.ACTOR, Source.BIAS)


@pytest.mark.parametrize("state", [UrState.STOPPED, UrState.STARTING, UrState.RUNNING])
def test_bias_is_rejected_outside_reset_states(state):
    assert not arbitrate(state, Source.ACTOR, Source.BIAS)


@pytest.mark.parametrize("source", [Source.VIVE, Source.ACTOR])
def test_streaming_sources_are_blocked_before_running(source):
    assert not arbitrate(UrState.STOPPED, source, source)
    assert not arbitrate(UrState.HOMED, source, source)
    assert not arbitrate(UrState.HOMING, source, source)
    assert not arbitrate(UrState.STARTING, source, source)


def test_running_accepts_only_active_streaming_source():
    assert arbitrate(UrState.RUNNING, Source.ACTOR, Source.ACTOR)
    assert not arbitrate(UrState.RUNNING, Source.ACTOR, Source.VIVE)


def test_unknown_source_is_rejected():
    assert not arbitrate(UrState.RUNNING, Source.UNKNOWN, Source.UNKNOWN)
