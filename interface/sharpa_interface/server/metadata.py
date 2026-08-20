"""Strict validators for SharpA policy server v3 metadata.

Ported from UR YNS sharpa_policy_v3_client/metadata.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .serialization import MAX_MESSAGE_SIZE


SERVER_SCHEMA = "sharpa_policy_server.v3"
OBSERVATION_SCHEMA = "sharpa_policy_observation.v3"
ACTION_SCHEMA = "sharpa_policy_action.v4"
METADATA_FORMAT_SCHEMA = "sharpa_policy_metadata_format.v1"
RESET_SCHEMA = "sharpa_policy_reset.v1"
TRANSPORT = "websocket+binary_msgpack"

IMAGE_NAMES = ("ego_cam", "left_wrist_cam", "right_wrist_cam")
SENSOR_NAMES = ("tau", "wrench", "deformation")


class MetadataValidationError(ValueError):
    pass


def _object(value: Any, path: str, fields: tuple[str, ...]) -> Mapping[str, Any]:
    """Return a mapping containing every required field.

    Interface dicts are intentionally extensible: producers may append fields,
    but may not omit any field frozen by the minimum contract.
    """
    if not isinstance(value, Mapping):
        raise MetadataValidationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise MetadataValidationError(f"{path} keys must be strings")
    missing = sorted(set(fields).difference(value))
    if missing:
        raise MetadataValidationError(f"{path} missing fields: {', '.join(missing)}")
    return value


def _string(value: Any, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise MetadataValidationError(f"{path} must be a string")
    if nonempty and not value:
        raise MetadataValidationError(f"{path} must not be empty")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise MetadataValidationError(f"{path} must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MetadataValidationError(f"{path} must be an integer")
    if value < minimum:
        raise MetadataValidationError(f"{path} must be >= {minimum}")
    return value


def _history_selector(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path, ("history_len", "current"))
    return {
        **dict(obj),
        "history_len": _integer(obj["history_len"], f"{path}.history_len"),
        "current": _boolean(obj["current"], f"{path}.current"),
    }


def validate_metadata_format(value: Any) -> dict[str, Any]:
    obj = _object(value, "metadata_format", ("schema", "format_id", "image", "state", "sensor"))
    schema = _string(obj["schema"], "metadata_format.schema")
    if schema != METADATA_FORMAT_SCHEMA:
        raise MetadataValidationError(f"metadata_format.schema must be {METADATA_FORMAT_SCHEMA}")
    image_obj = _object(obj["image"], "metadata_format.image", IMAGE_NAMES)
    image = {
        **dict(image_obj),
        **{
            name: _history_selector(
                image_obj[name], f"metadata_format.image.{name}"
            )
            for name in IMAGE_NAMES
        },
    }
    state_obj = _object(obj["state"], "metadata_format.state",
                        ("history_len", "current", "left_wrist", "right_wrist", "hand_joint"))
    def wrist_sel(v: Any, p: str) -> dict[str, bool]:
        o = _object(v, p, ("joint", "eef"))
        return {
            **dict(o),
            "joint": _boolean(o["joint"], f"{p}.joint"),
            "eef": _boolean(o["eef"], f"{p}.eef"),
        }
    def hand_sel(v: Any, p: str) -> dict[str, bool]:
        o = _object(v, p, ("left", "right"))
        return {
            **dict(o),
            "left": _boolean(o["left"], f"{p}.left"),
            "right": _boolean(o["right"], f"{p}.right"),
        }
    state = {
        **dict(state_obj),
        "history_len": _integer(state_obj["history_len"], "metadata_format.state.history_len"),
        "current": _boolean(state_obj["current"], "metadata_format.state.current"),
        "left_wrist": wrist_sel(state_obj["left_wrist"], "metadata_format.state.left_wrist"),
        "right_wrist": wrist_sel(state_obj["right_wrist"], "metadata_format.state.right_wrist"),
        "hand_joint": hand_sel(state_obj["hand_joint"], "metadata_format.state.hand_joint"),
    }
    sensor_obj = _object(obj["sensor"], "metadata_format.sensor", SENSOR_NAMES)
    sensor = {
        **dict(sensor_obj),
        **{
            name: _history_selector(
                sensor_obj[name], f"metadata_format.sensor.{name}"
            )
            for name in SENSOR_NAMES
        },
    }
    return {
        **dict(obj),
        "schema": schema,
        "format_id": _string(obj["format_id"], "metadata_format.format_id", nonempty=True),
        "image": image,
        "state": state,
        "sensor": sensor,
    }


def validate_server_metadata(value: Any) -> dict[str, Any]:
    fields = ("schema", "policy_family", "checkpoint_id", "checkpoint_path", "task_id", "run_id",
              "dataset_path", "prompt", "transport", "observation_schema", "action_schema", "host",
              "port", "infer_path", "health_path", "metadata_path", "reset_path",
              "max_message_size", "metadata_format")
    obj = _object(value, "metadata", fields)
    must_equal = {"schema": SERVER_SCHEMA, "transport": TRANSPORT,
                  "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA,
                  "infer_path": "/infer", "health_path": "/healthz",
                  "metadata_path": "/metadata", "reset_path": "/reset"}
    normalized: dict[str, Any] = dict(obj)
    for f in fields:
        if f in {"port", "max_message_size", "metadata_format"}:
            continue
        normalized[f] = _string(obj[f], f"metadata.{f}",
                                 nonempty=f not in {"checkpoint_path", "dataset_path", "prompt"})
    for f, expected in must_equal.items():
        if normalized[f] != expected:
            raise MetadataValidationError(f"metadata.{f} must be {expected}")
    port = _integer(obj["port"], "metadata.port", minimum=1)
    if port > 65535:
        raise MetadataValidationError("metadata.port must be <= 65535")
    mms = _integer(obj["max_message_size"], "metadata.max_message_size", minimum=1)
    if mms > MAX_MESSAGE_SIZE:
        raise MetadataValidationError("metadata.max_message_size exceeds the 64 MiB client hard limit")
    normalized["port"] = port
    normalized["max_message_size"] = mms
    normalized["metadata_format"] = validate_metadata_format(obj["metadata_format"])
    return normalized


def validate_reset_response(value: Any, *, expected_session_id: str | None = None,
                             expected_request_id: int | None = None) -> dict[str, Any]:
    obj = _object(value, "reset_result", ("schema", "session_id", "request_id", "reset", "metadata_format"))
    schema = _string(obj["schema"], "reset_result.schema")
    if schema != RESET_SCHEMA:
        raise MetadataValidationError(f"reset_result.schema must be {RESET_SCHEMA}")
    session_id = _string(obj["session_id"], "reset_result.session_id", nonempty=True)
    request_id_raw = obj["request_id"]
    request_id = (
        None
        if request_id_raw is None
        else _integer(request_id_raw, "reset_result.request_id")
    )
    if obj["reset"] is not True:
        raise MetadataValidationError("reset_result.reset must be true")
    if expected_session_id is not None and session_id != expected_session_id:
        raise MetadataValidationError("reset_result.session_id does not match the request")
    if expected_request_id is not None and request_id != expected_request_id:
        raise MetadataValidationError("reset_result.request_id does not match the request")
    return {
        **dict(obj),
        "schema": schema,
        "session_id": session_id,
        "request_id": request_id,
        "reset": True,
        "metadata_format": validate_metadata_format(obj["metadata_format"]),
    }
