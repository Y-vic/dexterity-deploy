"""Inference timing boundaries, cancellation and request-ID reuse."""

import pytest

from policy_client_node.request_metrics import PolicyRequestMetrics


def test_pending_elapsed_and_model_average_use_separate_request_samples():
    metrics = PolicyRequestMetrics()
    metrics.begin("rollout-1", 0, 1, now=10.0)
    pending = metrics.snapshot(now=10.4)
    assert pending["inference_pending"]
    assert pending["pending_request_id"] == 0
    assert pending["pending_session_id"] == "rollout-1"
    assert pending["pending_elapsed_s"] == pytest.approx(0.4)
    assert pending["latency_s"] is None
    assert metrics.complete("rollout-1", 0, 1, 0.3)
    metrics.begin("rollout-1", 1, 1, now=11.0)
    assert metrics.complete("rollout-1", 1, 1, 0.5)
    completed = metrics.snapshot(now=12.0)
    assert completed["inference_count"] == 2
    assert completed["last_request_id"] == 1
    assert completed["latency_s"] == pytest.approx(0.5)
    assert completed["average_latency_s"] == pytest.approx(0.4)
    assert not completed["inference_pending"]
    assert completed["pending_elapsed_s"] is None
    assert completed["pending_request_id"] is None


def test_stop_and_rollout_reset_preserve_model_history_but_reject_stale_completion():
    metrics = PolicyRequestMetrics()
    metrics.record(4, 0.2)
    metrics.begin("rollout-1", 5, 1, now=10.0)
    metrics.cancel()
    assert not metrics.complete("rollout-1", 5, 1, 99.0)
    metrics.begin("rollout-2", 0, 2, now=20.0)
    assert not metrics.complete("rollout-1", 0, 2, 99.0)
    assert metrics.complete("rollout-2", 0, 2, 0.4)
    assert metrics.snapshot()["average_latency_s"] == pytest.approx(0.3)
    assert metrics.inference_count == 2


def test_epoch_rejects_reused_request_id_and_model_unload_clears_all_metrics():
    metrics = PolicyRequestMetrics()
    metrics.record(0, 0.2)
    metrics.begin("same-session", 0, 2, now=20.0)
    assert not metrics.complete("same-session", 0, 1, 99.0)
    assert metrics.snapshot(now=20.1)["inference_pending"]
    metrics.reset()
    assert not metrics.complete("same-session", 0, 2, 99.0)
    result = metrics.snapshot()
    assert result["inference_count"] == 0
    assert result["last_request_id"] == 0
    assert result["latency_s"] is None
    assert result["average_latency_s"] is None
    assert not result["inference_pending"]


@pytest.mark.parametrize("latency", [-1.0, float("nan"), float("inf")])
def test_invalid_transport_latency_cannot_poison_model_average(latency):
    metrics = PolicyRequestMetrics()
    with pytest.raises(ValueError, match="latency"):
        metrics.record(0, latency)
    assert metrics.inference_count == 0
