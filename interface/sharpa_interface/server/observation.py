"""Validation for the metadata-driven SharpA observation dict."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from .execution import validate_execution_feedback
from .metadata import OBSERVATION_SCHEMA, validate_metadata_format


class ObservationValidationError(ValueError):
    pass


def validate_policy_observation(value: Any, metadata_format: Any) -> dict[str, Any]:
    """Validate required keys and requested values while allowing extensions."""

    metadata = validate_metadata_format(metadata_format)
    obs = _required(
        value,
        "observation",
        (
            "schema",
            "metadata_format_id",
            "session_id",
            "request_id",
            "timestamp_ns",
            "prompt",
            "image",
            "state",
            "sensor",
            "execution_feedback",
        ),
    )
    if obs["schema"] != OBSERVATION_SCHEMA:
        raise ObservationValidationError(f"observation.schema must be {OBSERVATION_SCHEMA}")
    if obs["metadata_format_id"] != metadata["format_id"]:
        raise ObservationValidationError("observation.metadata_format_id is not active")
    _string(obs["session_id"], "observation.session_id", nonempty=True)
    _integer(obs["request_id"], "observation.request_id")
    _integer(obs["timestamp_ns"], "observation.timestamp_ns")
    _string(obs["prompt"], "observation.prompt")

    images = _required(
        obs["image"],
        "observation.image",
        ("ego_cam", "left_wrist_cam", "right_wrist_cam"),
    )
    for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam"):
        _camera_stream(images[name], metadata["image"][name], f"observation.image.{name}")

    _state(obs["state"], metadata["state"])
    sensors = _required(obs["sensor"], "observation.sensor", ("tau", "wrench", "deformation"))
    _sensor(sensors["tau"], metadata["sensor"]["tau"], "observation.sensor.tau", np.float32, (22,), (22,))
    _sensor(sensors["wrench"], metadata["sensor"]["wrench"], "observation.sensor.wrench", np.float32, (5, 6), (5,))
    _sensor(
        sensors["deformation"],
        metadata["sensor"]["deformation"],
        "observation.sensor.deformation",
        np.uint8,
        (5, 240, 240),
        (5,),
    )
    validate_execution_feedback(obs["execution_feedback"])
    return dict(obs)


def _camera_stream(value: Any, requirement: Mapping[str, Any], path: str) -> None:
    stream = _required(value, path, ("history", "current"))
    history = stream["history"]
    if not isinstance(history, list):
        raise ObservationValidationError(f"{path}.history must be a list")
    expected = int(requirement["history_len"])
    if len(history) != expected:
        raise ObservationValidationError(f"{path}.history must contain {expected} frames")
    stamps = [_camera_frame(frame, f"{path}.history[{index}]") for index, frame in enumerate(history)]
    _chronological(stamps, f"{path}.history")
    current = stream["current"]
    if requirement["current"]:
        stamp = _camera_frame(current, f"{path}.current")
        if stamps and stamp < stamps[-1]:
            raise ObservationValidationError(f"{path}.current precedes history")
    elif current is not None:
        raise ObservationValidationError(f"{path}.current must be None when not requested")


def _camera_frame(value: Any, path: str) -> int:
    frame = _required(value, path, ("encoding", "data", "timestamp_ns", "valid"))
    if frame["encoding"] != "jpeg" or not isinstance(frame["data"], bytes):
        raise ObservationValidationError(f"{path} must contain JPEG bytes")
    valid = _boolean(frame["valid"], f"{path}.valid")
    if valid and not frame["data"]:
        raise ObservationValidationError(f"{path}.data cannot be empty when valid")
    return _integer(frame["timestamp_ns"], f"{path}.timestamp_ns")


def _state(value: Any, requirement: Mapping[str, Any]) -> None:
    state = _required(value, "observation.state", ("history", "current"))
    history_len = int(requirement["history_len"])
    history = state["history"]
    last_stamp: int | None = None
    if history_len:
        batch = _required(history, "observation.state.history", ("timestamp_ns", "left_wrist", "right_wrist", "hand_joint", "valid"))
        stamps = _array(batch["timestamp_ns"], "observation.state.history.timestamp_ns", np.int64, (history_len,))
        _chronological(stamps.tolist(), "observation.state.history")
        last_stamp = int(stamps[-1])
        _wrist(batch["left_wrist"], requirement["left_wrist"], "observation.state.history.left_wrist", (history_len,))
        _wrist(batch["right_wrist"], requirement["right_wrist"], "observation.state.history.right_wrist", (history_len,))
        _hands(batch["hand_joint"], requirement["hand_joint"], "observation.state.history.hand_joint", (history_len,))
        _array(batch["valid"], "observation.state.history.valid", np.bool_, (history_len,))
    elif history is not None:
        raise ObservationValidationError("observation.state.history must be None")

    current = state["current"]
    if requirement["current"]:
        frame = _required(current, "observation.state.current", ("timestamp_ns", "left_wrist", "right_wrist", "hand_joint", "valid"))
        stamp = _integer(frame["timestamp_ns"], "observation.state.current.timestamp_ns")
        if last_stamp is not None and stamp < last_stamp:
            raise ObservationValidationError("observation.state.current precedes history")
        _wrist(frame["left_wrist"], requirement["left_wrist"], "observation.state.current.left_wrist", ())
        _wrist(frame["right_wrist"], requirement["right_wrist"], "observation.state.current.right_wrist", ())
        _hands(frame["hand_joint"], requirement["hand_joint"], "observation.state.current.hand_joint", ())
        _boolean(frame["valid"], "observation.state.current.valid")
    elif current is not None:
        raise ObservationValidationError("observation.state.current must be None")


def _wrist(value: Any, requirement: Mapping[str, Any], path: str, leading: tuple[int, ...]) -> None:
    wrist = _required(value, path, ("joint", "eef", "eef_def"))
    _optional_float_array(wrist["joint"], bool(requirement["joint"]), leading, None, f"{path}.joint")
    eef_required = bool(requirement["eef"])
    _optional_float_array(wrist["eef"], eef_required, leading, 9, f"{path}.eef")
    eef_def = wrist["eef_def"]
    if eef_required and eef_def not in (None, "absolute"):
        raise ObservationValidationError(f"{path}.eef_def must be absolute or None")
    if not eef_required and eef_def is not None:
        raise ObservationValidationError(f"{path}.eef_def must be None when EEF is not requested")


def _hands(value: Any, requirement: Mapping[str, Any], path: str, leading: tuple[int, ...]) -> None:
    hands = _required(value, path, ("left", "right"))
    for side in ("left", "right"):
        _optional_float_array(hands[side], bool(requirement[side]), leading, 22, f"{path}.{side}")


def _optional_float_array(value: Any, required: bool, leading: tuple[int, ...], width: int | None, path: str) -> None:
    if not required:
        if value is not None:
            raise ObservationValidationError(f"{path} must be None when not requested")
        return
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ObservationValidationError(f"{path} must be a float32 numpy.ndarray")
    if value.ndim != len(leading) + 1 or value.shape[: len(leading)] != leading:
        raise ObservationValidationError(f"{path} has an invalid leading shape")
    if width is None and value.shape[-1] < 1:
        raise ObservationValidationError(f"{path} must contain at least one joint")
    if width is not None and value.shape[-1] != width:
        raise ObservationValidationError(f"{path} must end in {width}")
    if not np.all(np.isfinite(value)):
        raise ObservationValidationError(f"{path} contains NaN or Inf")


def _sensor(value: Any, requirement: Mapping[str, Any], path: str, dtype: Any, sample_shape: tuple[int, ...], valid_shape: tuple[int, ...]) -> None:
    stream = _required(value, path, ("history", "current"))
    history_len = int(requirement["history_len"])
    history = stream["history"]
    last_stamp: int | None = None
    if history_len:
        batch = _required(history, f"{path}.history", ("left", "right", "timestamp_ns", "valid"))
        _side_arrays(batch, f"{path}.history", dtype, (history_len, *sample_shape), (history_len, *valid_shape))
        stamps = _array(batch["timestamp_ns"], f"{path}.history.timestamp_ns", np.int64, (history_len,))
        _chronological(stamps.tolist(), f"{path}.history")
        last_stamp = int(stamps[-1])
    elif history is not None:
        raise ObservationValidationError(f"{path}.history must be None")
    current = stream["current"]
    if requirement["current"]:
        frame = _required(current, f"{path}.current", ("left", "right", "timestamp_ns", "valid"))
        _side_arrays(frame, f"{path}.current", dtype, sample_shape, valid_shape)
        stamp = _integer(frame["timestamp_ns"], f"{path}.current.timestamp_ns")
        if last_stamp is not None and stamp < last_stamp:
            raise ObservationValidationError(f"{path}.current precedes history")
    elif current is not None:
        raise ObservationValidationError(f"{path}.current must be None")


def _side_arrays(value: Mapping[str, Any], path: str, dtype: Any, shape: tuple[int, ...], valid_shape: tuple[int, ...]) -> None:
    validity = _required(value["valid"], f"{path}.valid", ("left", "right"))
    for side in ("left", "right"):
        array = _array(value[side], f"{path}.{side}", dtype, shape)
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ObservationValidationError(f"{path}.{side} contains NaN or Inf")
        _array(validity[side], f"{path}.valid.{side}", np.bool_, valid_shape)


def _array(value: Any, path: str, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ObservationValidationError(f"{path} must have dtype {np.dtype(dtype)} and shape {shape}")
    return value


def _required(value: Any, path: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservationValidationError(f"{path} must be an object")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ObservationValidationError(f"{path} missing fields: {', '.join(missing)}")
    return value


def _string(value: Any, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ObservationValidationError(f"{path} must be a string")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 0:
        raise ObservationValidationError(f"{path} must be a nonnegative integer")
    return int(value)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ObservationValidationError(f"{path} must be a boolean")
    return bool(value)


def _chronological(values: list[int], path: str) -> None:
    if any(value < 0 for value in values) or any(a > b for a, b in zip(values, values[1:])):
        raise ObservationValidationError(f"{path} timestamps must be chronological")
