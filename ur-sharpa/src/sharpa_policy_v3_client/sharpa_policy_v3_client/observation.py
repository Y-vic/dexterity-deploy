"""Build strict sharpa_policy_observation.v3 request payloads."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping, Sequence, TypeVar

import numpy as np

from .buffers import (
    CAMERA_NAMES,
    DEFORMATION_FRAME_BYTE_SIZE,
    INT64_MAX,
    TAU_FRAME_BYTE_SIZE,
    WRENCH_FRAME_BYTE_SIZE,
    BufferUnderflowError,
    CameraFrame,
    DeformationFrame,
    ObservationBuffers,
    StateFrame,
    TauFrame,
    TimestampedBuffer,
    WrenchFrame,
    required_frame_count,
)


OBSERVATION_SCHEMA = "sharpa_policy_observation.v3"


class ObservationValidationError(ValueError):
    """Raised when buffered facts cannot satisfy the v3 wire schema."""


class ObservationNotReady(RuntimeError):
    """Raised when a requested modality lacks enough distinct source frames."""

    def __init__(self, path: str, *, required: int, available: int) -> None:
        self.path = str(path)
        self.required = int(required)
        self.available = int(available)
        super().__init__(
            f"{self.path} is not ready: {self.available} frames available, "
            f"{self.required} required"
        )


class ObservationCapacityError(ObservationValidationError):
    """Raised when a MetadataFormat cannot fit configured local resources."""

    def __init__(
        self,
        path: str,
        resource: str,
        *,
        required: int,
        available: int,
    ) -> None:
        self.path = str(path)
        self.resource = str(resource)
        self.required = int(required)
        self.available = int(available)
        super().__init__(
            f"{self.path} requires {self.required} {self.resource}; "
            f"only {self.available} available"
        )


FrameT = TypeVar(
    "FrameT",
    CameraFrame,
    StateFrame,
    TauFrame,
    WrenchFrame,
    DeformationFrame,
)


def _wire_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ObservationValidationError(f"{field} must be an integer")
    output = int(value)
    if output < 0 or output > INT64_MAX:
        raise ObservationValidationError(
            f"{field} must be in [0, {INT64_MAX}]"
        )
    return output


def _wire_bool(value: object, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ObservationValidationError(f"{field} must be boolean")
    return bool(value)


def _validate_execution_feedback(value: object | None) -> dict[str, Any]:
    if value is None:
        return {
            "last_action_id": None,
            "executed_steps": 0,
            "success": True,
        }
    if not isinstance(value, Mapping):
        raise ObservationValidationError("execution_feedback must be a mapping")
    expected = {"last_action_id", "executed_steps", "success"}
    if set(value) != expected:
        raise ObservationValidationError(
            "execution_feedback must contain exactly last_action_id, "
            "executed_steps, and success"
        )
    action_id = value["last_action_id"]
    if action_id is not None and (
        not isinstance(action_id, str) or not action_id
    ):
        raise ObservationValidationError(
            "execution_feedback.last_action_id must be a non-empty string or None"
        )
    return {
        "last_action_id": action_id,
        "executed_steps": _wire_int(
            value["executed_steps"],
            "execution_feedback.executed_steps",
        ),
        "success": _wire_bool(
            value["success"],
            "execution_feedback.success",
        ),
    }


def _select(
    buffer: TimestampedBuffer[FrameT],
    *,
    path: str,
    history_len: int,
    current: bool,
) -> tuple[tuple[FrameT, ...], FrameT | None]:
    try:
        return buffer.select(history_len=history_len, current=current)
    except BufferUnderflowError as exc:
        raise ObservationNotReady(
            path,
            required=exc.required,
            available=exc.available,
        ) from exc


def _state_fixed_frame_bytes(config: Mapping[str, Any]) -> int:
    total = 0
    if config["left_wrist"]["eef"]:
        total += 9 * np.dtype(np.float32).itemsize
    if config["right_wrist"]["eef"]:
        total += 9 * np.dtype(np.float32).itemsize
    if config["hand_joint"]["left"]:
        total += 22 * np.dtype(np.float32).itemsize
    if config["hand_joint"]["right"]:
        total += 22 * np.dtype(np.float32).itemsize
    return total


def _minimum_fixed_wire_bytes(metadata_format: Mapping[str, Any]) -> int:
    state = metadata_format["state"]
    state_frames = required_frame_count(state["history_len"], state["current"])
    total = state_frames * _state_fixed_frame_bytes(state)
    total += state["history_len"] * (
        np.dtype(np.int64).itemsize + np.dtype(np.bool_).itemsize
    )
    fixed_sensor_sizes = {
        "tau": TAU_FRAME_BYTE_SIZE,
        "wrench": WRENCH_FRAME_BYTE_SIZE,
        "deformation": DEFORMATION_FRAME_BYTE_SIZE,
    }
    for name, frame_bytes in fixed_sensor_sizes.items():
        selector = metadata_format["sensor"][name]
        frame_count = required_frame_count(
            selector["history_len"],
            selector["current"],
        )
        total += frame_count * frame_bytes
        total += selector["history_len"] * np.dtype(np.int64).itemsize
    return int(total)


def validate_format_capacity(
    buffers: ObservationBuffers,
    metadata_format: object,
    *,
    max_message_size: int,
) -> dict[str, Any]:
    """Validate local fixed costs; dynamic JPEG and joint bytes remain runtime-bound."""

    if not isinstance(buffers, ObservationBuffers):
        raise TypeError("buffers must be an ObservationBuffers")
    if isinstance(max_message_size, bool) or not isinstance(
        max_message_size,
        Integral,
    ):
        raise TypeError("max_message_size must be an integer")
    message_limit = int(max_message_size)
    if message_limit <= 0:
        raise ValueError("max_message_size must be positive")

    from .metadata import validate_metadata_format

    active_format = validate_metadata_format(metadata_format)
    sources = (
        *(
            (
                f"image.{camera_name}",
                active_format["image"][camera_name],
                buffers.camera(camera_name),
            )
            for camera_name in CAMERA_NAMES
        ),
        ("state", active_format["state"], buffers.state),
        ("sensor.tau", active_format["sensor"]["tau"], buffers.tau),
        (
            "sensor.wrench",
            active_format["sensor"]["wrench"],
            buffers.wrench,
        ),
        (
            "sensor.deformation",
            active_format["sensor"]["deformation"],
            buffers.deformation,
        ),
    )
    for path, selector, buffer in sources:
        required_frames = required_frame_count(
            selector["history_len"],
            selector["current"],
        )
        if required_frames > buffer.frame_capacity:
            raise ObservationCapacityError(
                path,
                "frames",
                required=required_frames,
                available=buffer.frame_capacity,
            )

    fixed_buffer_requirements = (
        (
            "state",
            required_frame_count(
                active_format["state"]["history_len"],
                active_format["state"]["current"],
            )
            * _state_fixed_frame_bytes(active_format["state"]),
            buffers.state.byte_capacity,
        ),
        (
            "sensor.tau",
            required_frame_count(
                active_format["sensor"]["tau"]["history_len"],
                active_format["sensor"]["tau"]["current"],
            )
            * TAU_FRAME_BYTE_SIZE,
            buffers.tau.byte_capacity,
        ),
        (
            "sensor.wrench",
            required_frame_count(
                active_format["sensor"]["wrench"]["history_len"],
                active_format["sensor"]["wrench"]["current"],
            )
            * WRENCH_FRAME_BYTE_SIZE,
            buffers.wrench.byte_capacity,
        ),
        (
            "sensor.deformation",
            required_frame_count(
                active_format["sensor"]["deformation"]["history_len"],
                active_format["sensor"]["deformation"]["current"],
            )
            * DEFORMATION_FRAME_BYTE_SIZE,
            buffers.deformation.byte_capacity,
        ),
    )
    for path, required_bytes, available_bytes in fixed_buffer_requirements:
        if required_bytes > available_bytes:
            raise ObservationCapacityError(
                path,
                "buffer bytes",
                required=required_bytes,
                available=available_bytes,
            )

    minimum_message_size = _minimum_fixed_wire_bytes(active_format) + 1
    if minimum_message_size > message_limit:
        raise ObservationCapacityError(
            "observation",
            "message bytes",
            required=minimum_message_size,
            available=message_limit,
        )
    return active_format


def _camera_wire(frame: CameraFrame) -> dict[str, Any]:
    return {
        "encoding": frame.encoding,
        "data": frame.data,
        "timestamp_ns": frame.timestamp_ns,
        "valid": frame.valid,
    }


def _requested_state_frame_bytes(
    frame: StateFrame,
    config: Mapping[str, Any],
) -> int:
    requested = (
        (config["left_wrist"]["joint"], frame.left_joint),
        (config["left_wrist"]["eef"], frame.left_eef),
        (config["right_wrist"]["joint"], frame.right_joint),
        (config["right_wrist"]["eef"], frame.right_eef),
        (config["hand_joint"]["left"], frame.left_hand_joint),
        (config["hand_joint"]["right"], frame.right_hand_joint),
    )
    return sum(
        int(value.nbytes)
        for enabled, value in requested
        if enabled and value is not None
    )


def _selected_raw_bytes(
    image_frames: Mapping[
        str,
        tuple[tuple[CameraFrame, ...], CameraFrame | None],
    ],
    state_history: Sequence[StateFrame],
    state_current: StateFrame | None,
    state_config: Mapping[str, Any],
    sensor_frames: Sequence[
        tuple[Sequence[TauFrame | WrenchFrame | DeformationFrame], object | None]
    ],
) -> int:
    total = 0
    for history, current in image_frames.values():
        total += sum(frame.byte_size for frame in history)
        if current is not None:
            total += current.byte_size
    total += sum(
        _requested_state_frame_bytes(frame, state_config)
        for frame in state_history
    )
    if state_current is not None:
        total += _requested_state_frame_bytes(state_current, state_config)
    total += len(state_history) * (
        np.dtype(np.int64).itemsize + np.dtype(np.bool_).itemsize
    )
    for history, current in sensor_frames:
        total += sum(frame.byte_size for frame in history)
        total += len(history) * np.dtype(np.int64).itemsize
        if current is not None:
            total += current.byte_size
    return int(total)


def _stack_component(
    frames: Sequence[StateFrame],
    attribute: str,
    path: str,
) -> np.ndarray:
    values: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for frame in frames:
        value = getattr(frame, attribute)
        if value is None:
            raise ObservationValidationError(
                f"{path} is requested but missing at timestamp "
                f"{frame.timestamp_ns}"
            )
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ObservationValidationError(
                f"{path} shape changed from {expected_shape} to {value.shape}"
            )
        values.append(value)
    return np.stack(values, axis=0).astype(np.float32, copy=False)


def _current_component(
    frame: StateFrame,
    attribute: str,
    path: str,
) -> np.ndarray:
    value = getattr(frame, attribute)
    if value is None:
        raise ObservationValidationError(
            f"{path} is requested but missing at timestamp {frame.timestamp_ns}"
        )
    return value.copy()


def _validate_state_joint_widths(
    history: Sequence[StateFrame],
    current: StateFrame | None,
    config: Mapping[str, Any],
) -> None:
    frames = tuple(history) + ((current,) if current is not None else ())
    for side, attribute in (
        ("left_wrist", "left_joint"),
        ("right_wrist", "right_joint"),
    ):
        if not config[side]["joint"]:
            continue
        expected_shape: tuple[int, ...] | None = None
        for frame in frames:
            value = getattr(frame, attribute)
            if value is None:
                continue
            if expected_shape is None:
                expected_shape = value.shape
            elif value.shape != expected_shape:
                raise ObservationValidationError(
                    f"state.{side}.joint shape changed from {expected_shape} "
                    f"to {value.shape} at timestamp {frame.timestamp_ns}"
                )


def _state_history_wire(
    frames: Sequence[StateFrame],
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not frames:
        return None
    left = config["left_wrist"]
    right = config["right_wrist"]
    hand = config["hand_joint"]
    return {
        "timestamp_ns": np.asarray(
            [frame.timestamp_ns for frame in frames],
            dtype=np.int64,
        ),
        "left_wrist": {
            "joint": (
                _stack_component(
                    frames,
                    "left_joint",
                    "state.history.left_wrist.joint",
                )
                if left["joint"]
                else None
            ),
            "eef": (
                _stack_component(
                    frames,
                    "left_eef",
                    "state.history.left_wrist.eef",
                )
                if left["eef"]
                else None
            ),
            "eef_def": "absolute" if left["eef"] else None,
        },
        "right_wrist": {
            "joint": (
                _stack_component(
                    frames,
                    "right_joint",
                    "state.history.right_wrist.joint",
                )
                if right["joint"]
                else None
            ),
            "eef": (
                _stack_component(
                    frames,
                    "right_eef",
                    "state.history.right_wrist.eef",
                )
                if right["eef"]
                else None
            ),
            "eef_def": "absolute" if right["eef"] else None,
        },
        "hand_joint": {
            "left": (
                _stack_component(
                    frames,
                    "left_hand_joint",
                    "state.history.hand_joint.left",
                )
                if hand["left"]
                else None
            ),
            "right": (
                _stack_component(
                    frames,
                    "right_hand_joint",
                    "state.history.hand_joint.right",
                )
                if hand["right"]
                else None
            ),
        },
        "valid": np.asarray([frame.valid for frame in frames], dtype=np.bool_),
    }


def _state_current_wire(
    frame: StateFrame | None,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if frame is None:
        return None
    left = config["left_wrist"]
    right = config["right_wrist"]
    hand = config["hand_joint"]
    return {
        "timestamp_ns": frame.timestamp_ns,
        "left_wrist": {
            "joint": (
                _current_component(
                    frame,
                    "left_joint",
                    "state.current.left_wrist.joint",
                )
                if left["joint"]
                else None
            ),
            "eef": (
                _current_component(
                    frame,
                    "left_eef",
                    "state.current.left_wrist.eef",
                )
                if left["eef"]
                else None
            ),
            "eef_def": "absolute" if left["eef"] else None,
        },
        "right_wrist": {
            "joint": (
                _current_component(
                    frame,
                    "right_joint",
                    "state.current.right_wrist.joint",
                )
                if right["joint"]
                else None
            ),
            "eef": (
                _current_component(
                    frame,
                    "right_eef",
                    "state.current.right_wrist.eef",
                )
                if right["eef"]
                else None
            ),
            "eef_def": "absolute" if right["eef"] else None,
        },
        "hand_joint": {
            "left": (
                _current_component(
                    frame,
                    "left_hand_joint",
                    "state.current.hand_joint.left",
                )
                if hand["left"]
                else None
            ),
            "right": (
                _current_component(
                    frame,
                    "right_hand_joint",
                    "state.current.hand_joint.right",
                )
                if hand["right"]
                else None
            ),
        },
        "valid": frame.valid,
    }


def _tau_history_wire(frames: Sequence[TauFrame]) -> dict[str, Any] | None:
    if not frames:
        return None
    return {
        "left": np.stack([frame.left for frame in frames], axis=0),
        "right": np.stack([frame.right for frame in frames], axis=0),
        "timestamp_ns": np.asarray(
            [frame.timestamp_ns for frame in frames],
            dtype=np.int64,
        ),
        "valid": {
            "left": np.stack([frame.left_valid for frame in frames], axis=0),
            "right": np.stack([frame.right_valid for frame in frames], axis=0),
        },
    }


def _tau_current_wire(frame: TauFrame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "left": frame.left.copy(),
        "right": frame.right.copy(),
        "timestamp_ns": frame.timestamp_ns,
        "valid": {
            "left": frame.left_valid.copy(),
            "right": frame.right_valid.copy(),
        },
    }


def _wrench_history_wire(
    frames: Sequence[WrenchFrame],
) -> dict[str, Any] | None:
    if not frames:
        return None
    return {
        "left": np.stack([frame.left for frame in frames], axis=0),
        "right": np.stack([frame.right for frame in frames], axis=0),
        "timestamp_ns": np.asarray(
            [frame.timestamp_ns for frame in frames],
            dtype=np.int64,
        ),
        "valid": {
            "left": np.stack([frame.left_valid for frame in frames], axis=0),
            "right": np.stack([frame.right_valid for frame in frames], axis=0),
        },
    }


def _wrench_current_wire(frame: WrenchFrame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "left": frame.left.copy(),
        "right": frame.right.copy(),
        "timestamp_ns": frame.timestamp_ns,
        "valid": {
            "left": frame.left_valid.copy(),
            "right": frame.right_valid.copy(),
        },
    }


def _deformation_history_wire(
    frames: Sequence[DeformationFrame],
) -> dict[str, Any] | None:
    if not frames:
        return None
    return {
        "left": np.stack([frame.left for frame in frames], axis=0),
        "right": np.stack([frame.right for frame in frames], axis=0),
        "timestamp_ns": np.asarray(
            [frame.timestamp_ns for frame in frames],
            dtype=np.int64,
        ),
        "valid": {
            "left": np.stack([frame.left_valid for frame in frames], axis=0),
            "right": np.stack([frame.right_valid for frame in frames], axis=0),
        },
    }


def _deformation_current_wire(
    frame: DeformationFrame | None,
) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "left": frame.left.copy(),
        "right": frame.right.copy(),
        "timestamp_ns": frame.timestamp_ns,
        "valid": {
            "left": frame.left_valid.copy(),
            "right": frame.right_valid.copy(),
        },
    }


class ObservationBuilder:
    """Construct observations from one continuously populated buffer set."""

    def __init__(self, buffers: ObservationBuffers) -> None:
        if not isinstance(buffers, ObservationBuffers):
            raise TypeError("buffers must be an ObservationBuffers")
        self.buffers = buffers

    def build(
        self,
        metadata_format: object,
        *,
        session_id: str,
        request_id: int,
        timestamp_ns: int,
        prompt: str = "",
        execution_feedback: object | None = None,
        max_message_size: int | None = None,
    ) -> dict[str, Any]:
        return build_observation(
            self.buffers,
            metadata_format,
            session_id=session_id,
            request_id=request_id,
            timestamp_ns=timestamp_ns,
            prompt=prompt,
            execution_feedback=execution_feedback,
            max_message_size=max_message_size,
        )


def build_observation(
    buffers: ObservationBuffers,
    metadata_format: object,
    *,
    session_id: str,
    request_id: int,
    timestamp_ns: int,
    prompt: str = "",
    execution_feedback: object | None = None,
    max_message_size: int | None = None,
) -> dict[str, Any]:
    """Build one strict v3 observation without padding unavailable history."""

    if not isinstance(buffers, ObservationBuffers):
        raise TypeError("buffers must be an ObservationBuffers")
    if not isinstance(session_id, str) or not session_id:
        raise ObservationValidationError("session_id must be a non-empty string")
    if not isinstance(prompt, str):
        raise ObservationValidationError("prompt must be a string")
    normalized_request_id = _wire_int(request_id, "request_id")
    normalized_timestamp_ns = _wire_int(timestamp_ns, "timestamp_ns")
    feedback = _validate_execution_feedback(execution_feedback)

    from .metadata import validate_metadata_format

    active_format = validate_metadata_format(metadata_format)
    format_id = active_format["format_id"]
    if max_message_size is not None:
        if isinstance(max_message_size, bool) or not isinstance(
            max_message_size,
            Integral,
        ):
            raise TypeError("max_message_size must be an integer or None")
        if int(max_message_size) <= 0:
            raise ValueError("max_message_size must be positive")

    with buffers.locked():
        selected_images: dict[
            str,
            tuple[tuple[CameraFrame, ...], CameraFrame | None],
        ] = {}
        for camera_name in CAMERA_NAMES:
            config = active_format["image"][camera_name]
            selected_images[camera_name] = _select(
                buffers.camera(camera_name),
                path=f"image.{camera_name}",
                history_len=config["history_len"],
                current=config["current"],
            )

        state_config = active_format["state"]
        state_history, state_current = _select(
            buffers.state,
            path="state",
            history_len=state_config["history_len"],
            current=state_config["current"],
        )
        _validate_state_joint_widths(
            state_history,
            state_current,
            state_config,
        )

        tau_config = active_format["sensor"]["tau"]
        tau_history, tau_current = _select(
            buffers.tau,
            path="sensor.tau",
            history_len=tau_config["history_len"],
            current=tau_config["current"],
        )
        wrench_config = active_format["sensor"]["wrench"]
        wrench_history, wrench_current = _select(
            buffers.wrench,
            path="sensor.wrench",
            history_len=wrench_config["history_len"],
            current=wrench_config["current"],
        )
        deformation_config = active_format["sensor"]["deformation"]
        deformation_history, deformation_current = _select(
            buffers.deformation,
            path="sensor.deformation",
            history_len=deformation_config["history_len"],
            current=deformation_config["current"],
        )
        if max_message_size is not None:
            selected_bytes = _selected_raw_bytes(
                selected_images,
                state_history,
                state_current,
                state_config,
                (
                    (tau_history, tau_current),
                    (wrench_history, wrench_current),
                    (deformation_history, deformation_current),
                ),
            )
            if selected_bytes + 1 > int(max_message_size):
                raise ObservationCapacityError(
                    "observation",
                    "selected raw bytes",
                    required=selected_bytes + 1,
                    available=int(max_message_size),
                )

        image = {
            camera_name: {
                "history": [_camera_wire(frame) for frame in history],
                "current": _camera_wire(current) if current is not None else None,
            }
            for camera_name, (history, current) in selected_images.items()
        }
        state = {
            "history": _state_history_wire(state_history, state_config),
            "current": _state_current_wire(state_current, state_config),
        }
        sensor = {
            "tau": {
                "history": _tau_history_wire(tau_history),
                "current": _tau_current_wire(tau_current),
            },
            "wrench": {
                "history": _wrench_history_wire(wrench_history),
                "current": _wrench_current_wire(wrench_current),
            },
            "deformation": {
                "history": _deformation_history_wire(deformation_history),
                "current": _deformation_current_wire(deformation_current),
            },
        }

    return {
        "schema": OBSERVATION_SCHEMA,
        "metadata_format_id": format_id,
        "session_id": session_id,
        "request_id": normalized_request_id,
        "timestamp_ns": normalized_timestamp_ns,
        "prompt": prompt,
        "image": image,
        "state": state,
        "sensor": sensor,
        "execution_feedback": feedback,
    }
