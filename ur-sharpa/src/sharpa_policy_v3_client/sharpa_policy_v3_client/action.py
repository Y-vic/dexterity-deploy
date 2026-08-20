"""Strict parsing for SharpA policy server v4 actions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt


ACTION_SCHEMA = "sharpa_policy_action.v4"
EEF_DIMENSION = 9
HAND_JOINT_DIMENSION = 44
_ROT6D_NORM_EPSILON = 1.0e-6
_ROT6D_MAX_ABS_COSINE = 1.0 - 1.0e-4


class ActionValidationError(ValueError):
    """An action does not satisfy the negotiated v3 contract."""


@dataclass(frozen=True)
class ActionExecution:
    frequency_hz: float
    action_length: int
    execute_start: int
    execute_length: int

    @property
    def execute_stop(self) -> int:
        return self.execute_start + self.execute_length


@dataclass(frozen=True)
class ActionDiagnostics:
    policy_family: str
    checkpoint_id: str
    checkpoint_path: str
    inference_latency_ms: float


@dataclass(frozen=True)
class ParsedPolicyActionV3:
    schema: str
    session_id: str
    request_id: int
    action_id: str
    revision: int
    timestamp_ns: int
    execution: ActionExecution
    left_wrist_action_type: str
    right_wrist_action_type: str
    left_wrist: npt.NDArray[np.float32]
    right_wrist: npt.NDArray[np.float32]
    hand_joint: npt.NDArray[np.float32]
    diagnostics: ActionDiagnostics
    next_metadata_format: dict[str, Any] | None

    @property
    def left_wrist_dimension(self) -> int:
        return int(self.left_wrist.shape[1])

    @property
    def right_wrist_dimension(self) -> int:
        return int(self.right_wrist.shape[1])

    @property
    def execution_slice(self) -> slice:
        return slice(self.execution.execute_start, self.execution.execute_stop)


def _object(
    value: Any,
    path: str,
    fields: tuple[str, ...],
    *,
    allow_unknown: bool = True,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionValidationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ActionValidationError(f"{path} keys must be strings")
    expected = set(fields)
    actual = set(value)
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    if missing:
        raise ActionValidationError(f"{path} missing fields: {', '.join(missing)}")
    if unknown and not allow_unknown:
        raise ActionValidationError(f"{path} has unknown fields: {', '.join(unknown)}")
    return value


def _string(value: Any, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ActionValidationError(f"{path} must be a string")
    if nonempty and not value:
        raise ActionValidationError(f"{path} must not be empty")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionValidationError(f"{path} must be an integer")
    if value < minimum:
        raise ActionValidationError(f"{path} must be >= {minimum}")
    return value


def _finite_float(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ActionValidationError(f"{path} must be a number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ActionValidationError(f"{path} must be finite")
    if minimum is not None and normalized < minimum:
        raise ActionValidationError(f"{path} must be >= {minimum}")
    return normalized


def _array(value: Any, path: str, action_length: int) -> npt.NDArray[np.float32]:
    if not isinstance(value, np.ndarray):
        raise ActionValidationError(f"{path} must be a numpy.ndarray")
    if value.dtype != np.dtype(np.float32):
        raise ActionValidationError(f"{path} dtype must be float32")
    if value.ndim != 2:
        raise ActionValidationError(f"{path} must be two-dimensional")
    if value.shape[0] != action_length:
        raise ActionValidationError(
            f"{path} first dimension must equal execution.action_length"
        )
    if not np.isfinite(value).all():
        raise ActionValidationError(f"{path} must contain only finite values")
    output = np.array(value, dtype=np.float32, order="C", copy=True)
    output.setflags(write=False)
    return output


def _validate_rot6d(value: npt.NDArray[np.float32], path: str) -> None:
    first = value[:, 3:6]
    second = value[:, 6:9]
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    if np.any(first_norm <= _ROT6D_NORM_EPSILON):
        raise ActionValidationError(f"{path} Rot6D first column must be nonzero")
    if np.any(second_norm <= _ROT6D_NORM_EPSILON):
        raise ActionValidationError(f"{path} Rot6D second column must be nonzero")
    cosine = np.sum(first * second, axis=1) / (first_norm * second_norm)
    if np.any(np.abs(cosine) >= _ROT6D_MAX_ABS_COSINE):
        raise ActionValidationError(
            f"{path} Rot6D columns must not be collinear or near-collinear"
        )


def _wrist_array(
    value: Any,
    path: str,
    action_type: str,
    action_length: int,
    expected_joint_dimension: int | None,
) -> npt.NDArray[np.float32]:
    array = _array(value, path, action_length)
    dimension = int(array.shape[1])
    if action_type in {"eef", "relative_eef"}:
        if dimension != EEF_DIMENSION:
            raise ActionValidationError(f"{path} dimension must be {EEF_DIMENSION}")
        _validate_rot6d(array, path)
        return array
    if dimension == 0:
        raise ActionValidationError(f"{path} joint dimension must be positive")
    if expected_joint_dimension is not None and dimension != expected_joint_dimension:
        raise ActionValidationError(
            f"{path} joint dimension must be {expected_joint_dimension}"
        )
    return array


def _expected_integer(value: int | None, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def parse_policy_action(
    value: Any,
    *,
    expected_session_id: str | None = None,
    expected_request_id: int | None = None,
    expected_action_id: str | None = None,
    expected_revision: int | None = None,
    expected_left_joint_dimension: int | None = None,
    expected_right_joint_dimension: int | None = None,
) -> ParsedPolicyActionV3:
    """Validate and normalize one unpacked ``sharpa_policy_action.v4`` object."""

    expected_request_id = _expected_integer(
        expected_request_id,
        "expected_request_id",
    )
    expected_revision = _expected_integer(expected_revision, "expected_revision")
    expected_left_joint_dimension = _expected_integer(
        expected_left_joint_dimension,
        "expected_left_joint_dimension",
    )
    expected_right_joint_dimension = _expected_integer(
        expected_right_joint_dimension,
        "expected_right_joint_dimension",
    )
    if expected_left_joint_dimension == 0:
        raise ActionValidationError("expected_left_joint_dimension must be positive")
    if expected_right_joint_dimension == 0:
        raise ActionValidationError("expected_right_joint_dimension must be positive")

    obj = _object(
        value,
        "action_result",
        (
            "schema",
            "session_id",
            "request_id",
            "action_id",
            "revision",
            "timestamp_ns",
            "execution",
            "action",
            "auxiliary",
            "diagnostics",
            "next_metadata_format",
        ),
    )
    schema = _string(obj["schema"], "action_result.schema")
    if schema != ACTION_SCHEMA:
        raise ActionValidationError(f"action_result.schema must be {ACTION_SCHEMA}")
    session_id = _string(
        obj["session_id"],
        "action_result.session_id",
        nonempty=True,
    )
    request_id = _integer(obj["request_id"], "action_result.request_id")
    action_id = _string(
        obj["action_id"],
        "action_result.action_id",
        nonempty=True,
    )
    revision = _integer(obj["revision"], "action_result.revision")
    timestamp_ns = _integer(obj["timestamp_ns"], "action_result.timestamp_ns")

    if expected_session_id is not None and session_id != expected_session_id:
        raise ActionValidationError(
            "action_result.session_id does not match the request"
        )
    if expected_request_id is not None and request_id != expected_request_id:
        raise ActionValidationError(
            "action_result.request_id does not match the request"
        )
    if expected_action_id is not None and action_id != expected_action_id:
        raise ActionValidationError(
            "action_result.action_id does not match the expected action"
        )
    if expected_revision is not None and revision != expected_revision:
        raise ActionValidationError(
            "action_result.revision does not match the expected revision"
        )

    execution_obj = _object(
        obj["execution"],
        "action_result.execution",
        ("frequency_hz", "action_length", "execute_start", "execute_length"),
    )
    frequency_hz = _finite_float(
        execution_obj["frequency_hz"],
        "action_result.execution.frequency_hz",
    )
    if frequency_hz <= 0.0:
        raise ActionValidationError(
            "action_result.execution.frequency_hz must be greater than zero"
        )
    action_length = _integer(
        execution_obj["action_length"],
        "action_result.execution.action_length",
        minimum=1,
    )
    execute_start = _integer(
        execution_obj["execute_start"],
        "action_result.execution.execute_start",
    )
    execute_length = _integer(
        execution_obj["execute_length"],
        "action_result.execution.execute_length",
        minimum=1,
    )
    if execute_start + execute_length > action_length:
        raise ActionValidationError(
            "action_result.execution slice exceeds action_length"
        )
    execution = ActionExecution(
        frequency_hz=frequency_hz,
        action_length=action_length,
        execute_start=execute_start,
        execute_length=execute_length,
    )

    action_obj = _object(
        obj["action"],
        "action_result.action",
        ("left_wrist", "right_wrist", "hand_joint"),
    )

    def wrist_action(
        side: str,
        expected_joint_dimension: int | None,
    ) -> tuple[str, npt.NDArray[np.float32]]:
        wrist = _object(
            action_obj[side],
            f"action_result.action.{side}",
            ("joint", "eef", "eef_def"),
        )
        eef_def = "absolute" if wrist["eef_def"] is None else wrist["eef_def"]
        joint = wrist["joint"]
        eef = wrist["eef"]
        if joint is None and eef is None:
            raise ActionValidationError(
                f"action_result.action.{side} must provide joint or eef"
            )
        if joint is not None:
            return "joint", _wrist_array(
                joint,
                f"action_result.action.{side}.joint",
                "joint",
                action_length,
                expected_joint_dimension,
            )
        if eef_def != "absolute":
            raise ActionValidationError(
                f"action_result.action.{side}.eef_def must be absolute"
            )
        return "eef", _wrist_array(
            eef,
            f"action_result.action.{side}.eef",
            "eef",
            action_length,
            None,
        )

    left_type, left_wrist = wrist_action(
        "left_wrist", expected_left_joint_dimension
    )
    right_type, right_wrist = wrist_action(
        "right_wrist", expected_right_joint_dimension
    )
    hand_obj = _object(
        action_obj["hand_joint"],
        "action_result.action.hand_joint",
        ("left", "right"),
    )
    left_hand = _array(
        hand_obj["left"], "action_result.action.hand_joint.left", action_length
    )
    right_hand = _array(
        hand_obj["right"], "action_result.action.hand_joint.right", action_length
    )
    if left_hand.shape[1] != 22 or right_hand.shape[1] != 22:
        raise ActionValidationError(
            "action_result.action.hand_joint sides must each have dimension 22"
        )
    hand_joint = np.concatenate((left_hand, right_hand), axis=1)
    hand_joint.setflags(write=False)

    auxiliary = _object(
        obj["auxiliary"], "action_result.auxiliary", ("video", "tactile")
    )
    _object(
        auxiliary["video"],
        "action_result.auxiliary.video",
        ("ego", "left_wrist", "right_wrist"),
    )
    _object(
        auxiliary["tactile"],
        "action_result.auxiliary.tactile",
        ("deformation", "wrench", "hand_tau"),
    )

    diagnostics_obj = _object(
        obj["diagnostics"],
        "action_result.diagnostics",
        (
            "policy_family",
            "checkpoint_id",
            "checkpoint_path",
            "inference_latency_ms",
        ),
        allow_unknown=True,
    )
    diagnostics = ActionDiagnostics(
        policy_family=_string(
            diagnostics_obj["policy_family"],
            "action_result.diagnostics.policy_family",
            nonempty=True,
        ),
        checkpoint_id=_string(
            diagnostics_obj["checkpoint_id"],
            "action_result.diagnostics.checkpoint_id",
            nonempty=True,
        ),
        checkpoint_path=_string(
            diagnostics_obj["checkpoint_path"],
            "action_result.diagnostics.checkpoint_path",
        ),
        inference_latency_ms=_finite_float(
            diagnostics_obj["inference_latency_ms"],
            "action_result.diagnostics.inference_latency_ms",
            minimum=0.0,
        ),
    )

    next_metadata_format_value = obj["next_metadata_format"]
    if next_metadata_format_value is not None and not isinstance(
        next_metadata_format_value,
        dict,
    ):
        raise ActionValidationError(
            "action_result.next_metadata_format must be an object or null"
        )
    next_metadata_format = (
        None
        if next_metadata_format_value is None
        else deepcopy(next_metadata_format_value)
    )

    return ParsedPolicyActionV3(
        schema=schema,
        session_id=session_id,
        request_id=request_id,
        action_id=action_id,
        revision=revision,
        timestamp_ns=timestamp_ns,
        execution=execution,
        left_wrist_action_type=left_type,
        right_wrist_action_type=right_type,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        hand_joint=hand_joint,
        diagnostics=diagnostics,
        next_metadata_format=next_metadata_format,
    )


parse_action = parse_policy_action
