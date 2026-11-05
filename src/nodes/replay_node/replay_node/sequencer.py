"""Replay configuration and finite segment sequencing, independent of ROS."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from .sample import ReplaySample


@dataclass(frozen=True)
class ReplayConfig:
    execution_steps: int = 40
    frequency_hz: float = 30.0
    loop: bool = False
    dry_run: bool = True
    execution_scope: str = "full"

    def __post_init__(self) -> None:
        if (
            isinstance(self.execution_steps, bool)
            or not isinstance(self.execution_steps, Integral)
            or not 1 <= int(self.execution_steps) <= 1000
        ):
            raise ValueError("execution_steps must be in [1, 1000]")
        if (
            isinstance(self.frequency_hz, bool)
            or not isinstance(self.frequency_hz, Real)
            or not math.isfinite(float(self.frequency_hz))
            or not 5.0 <= float(self.frequency_hz) <= 125.0
        ):
            raise ValueError("frequency_hz must be in [5, 125]")
        if type(self.loop) is not bool or type(self.dry_run) is not bool:
            raise ValueError("loop and dry_run must be boolean")
        if self.loop:
            raise ValueError("loop replay is disabled; clear progress to run again")
        if self.execution_scope not in {"full", "hands_only"}:
            raise ValueError("execution_scope must be full or hands_only")


class ReplaySequencer:
    """Pure segment and identity state used by the ROS wrapper."""

    def __init__(self, sample: ReplaySample, config: ReplayConfig) -> None:
        self.sample = sample
        self.config = config
        self.execution_id = ""
        self.next_row = 0
        self.cycle = 0
        self.segment_index = 0
        self.request_id = 0
        self.reset()

    def reset(self) -> None:
        self.execution_id = uuid.uuid4().hex
        self.next_row = 0
        self.cycle = 0
        self.segment_index = 0
        self.request_id = 0

    def next_identity(self, label: str) -> dict[str, Any]:
        request_id = self.request_id
        self.request_id += 1
        safe_label = str(label).strip().replace(" ", "_") or "action"
        return {
            "session_id": f"replay_{self.sample.name}_{self.execution_id}",
            "request_id": request_id,
            "action_id": f"replay_{self.execution_id}_{safe_label}_{request_id}",
            "revision": self.cycle,
        }

    def next_segment(self) -> dict[str, Any] | None:
        if self.next_row >= self.sample.row_count:
            return None
        start = self.next_row
        end = min(start + int(self.config.execution_steps), self.sample.row_count)
        self.segment_index += 1
        self.next_row = end
        return {
            **self.next_identity(f"segment_{self.cycle}_{start}_{end}"),
            "frequency_hz": float(self.config.frequency_hz),
            "action_length": self.sample.row_count,
            "execute_start": start,
            "execute_length": end - start,
            "left_arm_joint": self.sample.arm_joint[start:end, :6].reshape(-1).tolist(),
            "right_arm_joint": self.sample.arm_joint[start:end, 6:].reshape(-1).tolist(),
            "hand_joint": self.sample.hand_joint[start:end].reshape(-1).tolist(),
        }
