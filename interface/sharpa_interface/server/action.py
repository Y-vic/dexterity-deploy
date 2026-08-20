"""Minimum extensible dict contract for SharpA policy actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import numpy.typing as npt

from .metadata import validate_metadata_format


ACTION_SCHEMA = "sharpa_policy_action.v4"
EXECUTABLE_ACTION_SCHEMA = "sharpa_executable_action.v1"
ERROR_SCHEMA = "sharpa_policy_error.v1"
EEF_DEFINITIONS = frozenset({"absolute", "relative"})


class ActionValidationError(ValueError):
    pass


class PolicyServerError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        request_id: int | None,
        retryable: bool,
    ) -> None:
        super().__init__(f"policy server {code}: {message}")
        self.code = code
        self.message = message
        self.request_id = request_id
        self.retryable = retryable


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
class WristAction:
    joint: npt.NDArray[np.float32] | None
    eef: npt.NDArray[np.float32] | None
    eef_def: str


@dataclass(frozen=True)
class HandAction:
    left: npt.NDArray[np.float32] | None
    right: npt.NDArray[np.float32] | None


@dataclass(frozen=True)
class ParsedPolicyAction:
    session_id: str
    request_id: int
    action_id: str
    revision: int
    timestamp_ns: int
    execution: ActionExecution
    left_wrist: WristAction
    right_wrist: WristAction
    hand_joint: HandAction
    auxiliary: dict[str, Any]
    diagnostics: dict[str, Any] | None
    next_metadata_format: dict[str, Any] | None

    @property
    def execution_slice(self) -> slice:
        return slice(self.execution.execute_start, self.execution.execute_stop)

    def executable_action(self) -> dict[str, Any]:
        """Return only the rows requested by the server for this cycle."""

        rows = self.execution_slice
        return {
            "left_wrist": {
                "joint": _slice(self.left_wrist.joint, rows),
                "eef": _slice(self.left_wrist.eef, rows),
                "eef_def": self.left_wrist.eef_def,
            },
            "right_wrist": {
                "joint": _slice(self.right_wrist.joint, rows),
                "eef": _slice(self.right_wrist.eef, rows),
                "eef_def": self.right_wrist.eef_def,
            },
            "hand_joint": {
                "left": _slice(self.hand_joint.left, rows),
                "right": _slice(self.hand_joint.right, rows),
            },
        }

    def execution_command(self) -> dict[str, Any]:
        """Build the identity-preserving message sent to ``action_ik``."""

        return {
            "schema": EXECUTABLE_ACTION_SCHEMA,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "action_id": self.action_id,
            "revision": self.revision,
            "timestamp_ns": self.timestamp_ns,
            "execution": {
                "frequency_hz": self.execution.frequency_hz,
                "action_length": self.execution.action_length,
                "execute_start": self.execution.execute_start,
                "execute_length": self.execution.execute_length,
            },
            "action": self.executable_action(),
        }


# Import compatibility while callers migrate their type annotations.
ParsedPolicyActionV3 = ParsedPolicyAction


def empty_auxiliary() -> dict[str, Any]:
    """Return the required auxiliary columns for policies without predictions."""

    return {
        "video": {"ego": None, "left_wrist": None, "right_wrist": None},
        "tactile": {"deformation": None, "wrench": None, "hand_tau": None},
    }


def parse_policy_action(
    value: Any,
    *,
    expected_session_id: str,
    expected_request_id: int,
    expected_eef_def: str | None = "absolute",
) -> ParsedPolicyAction:
    envelope = _required_mapping(value, "action_result", ("schema",))
    if envelope["schema"] == ERROR_SCHEMA:
        _raise_server_error(envelope, expected_request_id)
    result = _required_mapping(
        envelope,
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
    if result["schema"] != ACTION_SCHEMA:
        raise ActionValidationError(f"action_result.schema must be {ACTION_SCHEMA}")
    if result["session_id"] != expected_session_id:
        raise ActionValidationError("action_result.session_id does not match request")

    request_id = _integer(result["request_id"], "action_result.request_id")
    if request_id != expected_request_id:
        raise ActionValidationError("action_result.request_id does not match request")

    execution_raw = _required_mapping(
        result["execution"],
        "action_result.execution",
        ("frequency_hz", "action_length", "execute_start", "execute_length"),
    )
    execution = ActionExecution(
        frequency_hz=_positive_float(
            execution_raw["frequency_hz"], "action_result.execution.frequency_hz"
        ),
        action_length=_integer(
            execution_raw["action_length"],
            "action_result.execution.action_length",
            minimum=1,
        ),
        execute_start=_integer(
            execution_raw["execute_start"], "action_result.execution.execute_start"
        ),
        execute_length=_integer(
            execution_raw["execute_length"],
            "action_result.execution.execute_length",
            minimum=1,
        ),
    )
    if execution.execute_stop > execution.action_length:
        raise ActionValidationError("action_result execution slice exceeds action_length")

    action = _required_mapping(
        result["action"],
        "action_result.action",
        ("left_wrist", "right_wrist", "hand_joint"),
    )
    left_wrist = _wrist(
        action["left_wrist"], execution.action_length, "action_result.action.left_wrist"
    )
    right_wrist = _wrist(
        action["right_wrist"], execution.action_length, "action_result.action.right_wrist"
    )
    if expected_eef_def is not None:
        for side, wrist in (("left", left_wrist), ("right", right_wrist)):
            if wrist.eef is not None and wrist.eef_def != expected_eef_def:
                raise ActionValidationError(
                    f"action_result.action.{side}_wrist.eef_def must be "
                    f"{expected_eef_def!r} at the interface boundary"
                )

    hands = _required_mapping(
        action["hand_joint"],
        "action_result.action.hand_joint",
        ("left", "right"),
    )
    hand_joint = HandAction(
        left=_optional_array(
            hands["left"],
            "action_result.action.hand_joint.left",
            execution.action_length,
            width=22,
        ),
        right=_optional_array(
            hands["right"],
            "action_result.action.hand_joint.right",
            execution.action_length,
            width=22,
        ),
    )

    auxiliary = _required_mapping(
        result["auxiliary"],
        "action_result.auxiliary",
        ("video", "tactile"),
    )
    video = _required_mapping(
        auxiliary["video"],
        "action_result.auxiliary.video",
        ("ego", "left_wrist", "right_wrist"),
    )
    tactile = _required_mapping(
        auxiliary["tactile"],
        "action_result.auxiliary.tactile",
        ("deformation", "wrench", "hand_tau"),
    )
    normalized_auxiliary = {
        **dict(auxiliary),
        "video": dict(video),
        "tactile": dict(tactile),
    }

    diagnostics_raw = result["diagnostics"]
    if diagnostics_raw is not None and not isinstance(diagnostics_raw, Mapping):
        raise ActionValidationError("action_result.diagnostics must be an object or None")
    next_format_raw = result["next_metadata_format"]
    next_format = (
        validate_metadata_format(next_format_raw) if next_format_raw is not None else None
    )
    return ParsedPolicyAction(
        session_id=_nonempty_string(result["session_id"], "action_result.session_id"),
        request_id=request_id,
        action_id=_nonempty_string(result["action_id"], "action_result.action_id"),
        revision=_integer(result["revision"], "action_result.revision"),
        timestamp_ns=_integer(result["timestamp_ns"], "action_result.timestamp_ns"),
        execution=execution,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        hand_joint=hand_joint,
        auxiliary=normalized_auxiliary,
        diagnostics=dict(diagnostics_raw) if diagnostics_raw is not None else None,
        next_metadata_format=next_format,
    )


def _wrist(value: Any, length: int, path: str) -> WristAction:
    wrist = _required_mapping(value, path, ("joint", "eef", "eef_def"))
    eef_def_raw = wrist["eef_def"]
    eef_def = "absolute" if eef_def_raw is None else eef_def_raw
    if eef_def not in EEF_DEFINITIONS:
        raise ActionValidationError(f"{path}.eef_def must be absolute, relative, or None")
    return WristAction(
        joint=_optional_array(wrist["joint"], f"{path}.joint", length, width=None),
        eef=_optional_array(wrist["eef"], f"{path}.eef", length, width=9),
        eef_def=eef_def,
    )


def _optional_array(
    value: Any,
    path: str,
    length: int,
    *,
    width: int | None,
) -> npt.NDArray[np.float32] | None:
    if value is None:
        return None
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
        raise ActionValidationError(f"{path} must be a float32 numpy.ndarray or None")
    if value.ndim != 2 or value.shape[0] != length:
        raise ActionValidationError(f"{path} must have shape ({length}, D)")
    if width is not None and value.shape[1] != width:
        raise ActionValidationError(f"{path} must have shape ({length}, {width})")
    if width is None and value.shape[1] < 1:
        raise ActionValidationError(f"{path} must contain at least one joint")
    if not np.all(np.isfinite(value)):
        raise ActionValidationError(f"{path} contains NaN or Inf")
    return value


def _raise_server_error(response: Mapping[str, Any], expected_request_id: int) -> None:
    request_id = response.get("request_id")
    if request_id is not None:
        request_id = _integer(request_id, "error.request_id")
        if request_id != expected_request_id:
            raise ActionValidationError("error.request_id does not match request")
    error = _required_mapping(
        response.get("error"), "error.error", ("code", "message", "retryable")
    )
    if type(error["retryable"]) is not bool:
        raise ActionValidationError("error.error.retryable must be a boolean")
    raise PolicyServerError(
        _nonempty_string(error["code"], "error.error.code"),
        _nonempty_string(error["message"], "error.error.message"),
        request_id=request_id,
        retryable=error["retryable"],
    )


def _required_mapping(value: Any, path: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionValidationError(f"{path} must be an object")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ActionValidationError(f"{path} missing fields: {', '.join(missing)}")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(f"{path} must be a nonempty string")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ActionValidationError(f"{path} must be an integer")
    result = int(value)
    if result < minimum:
        raise ActionValidationError(f"{path} must be >= {minimum}")
    return result


def _positive_float(value: Any, path: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ActionValidationError(f"{path} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ActionValidationError(f"{path} must be finite and positive")
    return result


def _slice(
    value: npt.NDArray[np.float32] | None,
    rows: slice,
) -> npt.NDArray[np.float32] | None:
    return None if value is None else value[rows].copy()
