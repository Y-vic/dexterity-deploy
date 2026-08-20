"""Independent modality buffers and metadata-driven observation assembly."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .execution import initial_execution_feedback
from .metadata import OBSERVATION_SCHEMA, validate_metadata_format
from .observation import validate_policy_observation


BUFFER_NAMES = (
    "ego_cam",
    "left_wrist_cam",
    "right_wrist_cam",
    "state",
    "tau",
    "wrench",
    "deformation",
)


@dataclass(frozen=True)
class BufferSelection:
    history: list[dict[str, Any]]
    current: dict[str, Any] | None


class TemporalBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("buffer capacity must be positive")
        self._items: deque[dict[str, Any]] = deque(maxlen=capacity)

    def push(self, frame: Mapping[str, Any]) -> None:
        if "timestamp_ns" not in frame:
            raise ValueError("buffer frame missing timestamp_ns")
        stamp = frame["timestamp_ns"]
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0:
            raise ValueError("buffer timestamp_ns must be a nonnegative integer")
        if self._items and stamp < self._items[-1]["timestamp_ns"]:
            raise ValueError("buffer timestamps must be monotonic")
        self._items.append(deepcopy(dict(frame)))

    def select(self, history_len: int, current: bool) -> BufferSelection:
        if history_len < 0:
            raise ValueError("history_len must be nonnegative")
        count = history_len + int(current)
        if len(self._items) < count:
            raise RuntimeError(f"buffer needs {count} frames, has {len(self._items)}")
        if not count:
            return BufferSelection([], None)
        selected = list(self._items)[-count:]
        current_frame = deepcopy(selected[-1]) if current else None
        history = selected[:-1] if current else selected
        return BufferSelection(deepcopy(history), current_frame)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class PolicyInputBuffers:
    """Own one fixed buffer per input stream; metadata only selects windows."""

    DEFAULT_CAPACITIES = {
        "ego_cam": 3,
        "left_wrist_cam": 3,
        "right_wrist_cam": 3,
        "state": 19,
        "tau": 19,
        "wrench": 19,
        "deformation": 3,
    }

    def __init__(self, capacities: Mapping[str, int] | None = None) -> None:
        configured = {**self.DEFAULT_CAPACITIES, **dict(capacities or {})}
        missing = [name for name in BUFFER_NAMES if name not in configured]
        if missing:
            raise ValueError(f"missing buffer capacities: {', '.join(missing)}")
        self._buffers = {name: TemporalBuffer(int(configured[name])) for name in BUFFER_NAMES}

    def push(self, name: str, frame: Mapping[str, Any]) -> None:
        try:
            buffer = self._buffers[name]
        except KeyError as exc:
            raise ValueError(f"unknown policy input buffer: {name}") from exc
        buffer.push(frame)

    def clear(self) -> None:
        for buffer in self._buffers.values():
            buffer.clear()

    def lengths(self) -> dict[str, int]:
        return {name: len(buffer) for name, buffer in self._buffers.items()}

    def build_observation(
        self,
        metadata_format: Mapping[str, Any],
        *,
        session_id: str,
        request_id: int,
        timestamp_ns: int,
        prompt: str = "",
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = validate_metadata_format(metadata_format)
        image = {
            name: _camera_observation(self._buffers[name], metadata["image"][name])
            for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
        }
        state_selection = self._buffers["state"].select(
            metadata["state"]["history_len"], metadata["state"]["current"]
        )
        sensor = {
            name: _sensor_observation(self._buffers[name], metadata["sensor"][name])
            for name in ("tau", "wrench", "deformation")
        }
        observation = {
            "schema": OBSERVATION_SCHEMA,
            "metadata_format_id": metadata["format_id"],
            "session_id": session_id,
            "request_id": request_id,
            "timestamp_ns": timestamp_ns,
            "prompt": prompt,
            "image": image,
            "state": {
                "history": _state_history(state_selection.history, metadata["state"]),
                "current": state_selection.current,
            },
            "sensor": sensor,
            "execution_feedback": dict(execution_feedback or initial_execution_feedback()),
        }
        return validate_policy_observation(observation, metadata)


def _camera_observation(buffer: TemporalBuffer, requirement: Mapping[str, Any]) -> dict[str, Any]:
    selected = buffer.select(int(requirement["history_len"]), bool(requirement["current"]))
    return {"history": selected.history, "current": selected.current}


def _sensor_observation(buffer: TemporalBuffer, requirement: Mapping[str, Any]) -> dict[str, Any]:
    selected = buffer.select(int(requirement["history_len"]), bool(requirement["current"]))
    return {
        "history": _sensor_history(selected.history),
        "current": selected.current,
    }


def _state_history(frames: list[dict[str, Any]], requirement: Mapping[str, Any]) -> dict[str, Any] | None:
    if not frames:
        return None
    return {
        "timestamp_ns": np.asarray([frame["timestamp_ns"] for frame in frames], dtype=np.int64),
        "left_wrist": _wrist_history(frames, "left_wrist", requirement["left_wrist"]),
        "right_wrist": _wrist_history(frames, "right_wrist", requirement["right_wrist"]),
        "hand_joint": {
            side: _stack_optional(frames, ("hand_joint", side), bool(requirement["hand_joint"][side]))
            for side in ("left", "right")
        },
        "valid": np.asarray([frame["valid"] for frame in frames], dtype=np.bool_),
    }


def _wrist_history(frames: list[dict[str, Any]], side: str, requirement: Mapping[str, Any]) -> dict[str, Any]:
    eef_required = bool(requirement["eef"])
    if eef_required:
        definitions = [frame[side]["eef_def"] for frame in frames]
        if any(value not in (None, "absolute") for value in definitions):
            raise RuntimeError(f"buffer values for {side}.eef must be absolute")
    return {
        "joint": _stack_optional(frames, (side, "joint"), bool(requirement["joint"])),
        "eef": _stack_optional(frames, (side, "eef"), eef_required),
        "eef_def": "absolute" if eef_required else None,
    }


def _sensor_history(frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not frames:
        return None
    return {
        "left": np.stack([frame["left"] for frame in frames]),
        "right": np.stack([frame["right"] for frame in frames]),
        "timestamp_ns": np.asarray([frame["timestamp_ns"] for frame in frames], dtype=np.int64),
        "valid": {
            "left": np.stack([frame["valid"]["left"] for frame in frames]),
            "right": np.stack([frame["valid"]["right"] for frame in frames]),
        },
    }


def _stack_optional(frames: list[dict[str, Any]], path: tuple[str, str], required: bool) -> np.ndarray | None:
    if not required:
        return None
    first, second = path
    values = [frame[first][second] for frame in frames]
    if any(value is None for value in values):
        raise RuntimeError(f"buffer values for {first}.{second} are not ready")
    return np.stack(values)
