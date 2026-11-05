"""Request timing owned by one loaded model, independent of ROS and clocks."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any


@dataclass
class PolicyRequestMetrics:
    """Count accepted responses and time only the current inference request.

    Model replacement/unload resets statistics. Rollout resets and control
    changes cancel the pending request but preserve the model's history.
    The epoch disambiguates request IDs reused by a reset of the same session.
    """

    inference_count: int = 0
    last_request_id: int = 0
    latency_s: float | None = None
    _total_latency_s: float = field(default=0.0, init=False, repr=False)
    _pending: tuple[str, int, int] | None = field(default=None, init=False, repr=False)
    _started_at: float | None = field(default=None, init=False, repr=False)

    @property
    def inference_pending(self) -> bool:
        return self._pending is not None

    def begin(
        self, session_id: str, request_id: int, epoch: int, *, now: float | None = None
    ) -> None:
        self._pending = (str(session_id), int(request_id), int(epoch))
        self._started_at = time.monotonic() if now is None else float(now)

    def complete(
        self, session_id: str, request_id: int, epoch: int, latency_s: float
    ) -> bool:
        if self._pending != (str(session_id), int(request_id), int(epoch)):
            return False
        self.record(request_id, latency_s)
        self.cancel()
        return True

    def record(self, request_id: int, latency_s: float) -> None:
        latency_s = float(latency_s)
        if not math.isfinite(latency_s) or latency_s < 0:
            raise ValueError("request latency must be finite and non-negative")
        self.inference_count += 1
        self.last_request_id = int(request_id)
        self.latency_s = latency_s
        self._total_latency_s += latency_s

    def cancel(self) -> None:
        self._pending = None
        self._started_at = None

    def reset(self) -> None:
        self.cancel()
        self.inference_count = 0
        self.last_request_id = 0
        self.latency_s = None
        self._total_latency_s = 0.0

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        pending = self._pending
        current_time = time.monotonic() if now is None else float(now)
        return {
            "inference_count": self.inference_count,
            "last_request_id": self.last_request_id,
            "latency_s": self.latency_s,
            "average_latency_s": (
                self._total_latency_s / self.inference_count if self.inference_count else None
            ),
            "inference_pending": pending is not None,
            "pending_session_id": pending[0] if pending is not None else "",
            "pending_request_id": pending[1] if pending is not None else None,
            "pending_elapsed_s": (
                max(0.0, current_time - self._started_at)
                if self._started_at is not None else None
            ),
        }
