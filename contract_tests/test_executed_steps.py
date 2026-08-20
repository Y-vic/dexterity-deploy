"""Guardrails for the executed_steps contract.

If either embodiment reverts to slice-local semantics, this test fails.
Wire both embodiments' feedback-builders into this test in Phase 2/3.
"""

from __future__ import annotations

import pytest

from sharpa_interface.server.executed_steps import make_progress


def test_cumulative_semantics():
    p = make_progress(execute_start=100, executed_in_slice=7, revision=3)
    assert p.executed_steps == 107


def test_zero_start():
    p = make_progress(execute_start=0, executed_in_slice=5, revision=0)
    assert p.executed_steps == 5


def test_rejects_negative():
    with pytest.raises(ValueError):
        make_progress(execute_start=-1, executed_in_slice=0, revision=0)
    with pytest.raises(ValueError):
        make_progress(execute_start=0, executed_in_slice=-1, revision=0)


def test_monotonic_within_slice():
    prev = -1
    for k in range(10):
        p = make_progress(execute_start=200, executed_in_slice=k, revision=0)
        assert p.executed_steps > prev
        prev = p.executed_steps
