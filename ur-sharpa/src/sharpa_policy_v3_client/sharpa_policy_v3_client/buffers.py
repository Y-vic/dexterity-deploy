"""Thread-safe timestamp-ordered buffers for policy observation sources."""

from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
import threading
from typing import Generic, Iterator, TypeVar

import numpy as np


CAMERA_NAMES = ("ego_cam", "left_wrist_cam", "right_wrist_cam")
INT64_MAX = np.iinfo(np.int64).max
TAU_FRAME_BYTE_SIZE = 2 * 22 * np.dtype(np.float32).itemsize + 2 * 22
WRENCH_FRAME_BYTE_SIZE = 2 * 5 * 6 * np.dtype(np.float32).itemsize + 2 * 5
DEFORMATION_FRAME_BYTE_SIZE = 2 * 5 * 240 * 240 + 2 * 5
_ROT6D_NORM_EPSILON = 1.0e-6
_ROT6D_MAX_ABS_COSINE = 1.0 - 1.0e-4


class FrameValidationError(ValueError):
    """Raised when a source frame cannot satisfy the fixed local contract."""


class BufferCapacityError(ValueError):
    """Raised when one frame cannot fit in a buffer's byte budget."""


class BufferUnderflowError(LookupError):
    """Raised when a requested history/current selection is not available."""

    def __init__(self, *, required: int, available: int) -> None:
        self.required = int(required)
        self.available = int(available)
        super().__init__(
            f"buffer has {self.available} frames, requires {self.required}"
        )


def _timestamp_ns(value: object, field: str = "timestamp_ns") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise FrameValidationError(f"{field} must be an integer")
    timestamp = int(value)
    if timestamp < 0 or timestamp > INT64_MAX:
        raise FrameValidationError(f"{field} must be in [0, {INT64_MAX}]")
    return timestamp


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise FrameValidationError(f"{field} must be boolean")
    return bool(value)


def _array(
    value: object,
    *,
    field: str,
    dtype: np.dtype | type[np.generic],
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    nonempty: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FrameValidationError(f"{field} must be a numpy.ndarray")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise FrameValidationError(
            f"{field} dtype is {value.dtype}, expected {expected_dtype}"
        )
    if shape is not None and value.shape != shape:
        raise FrameValidationError(
            f"{field} shape is {value.shape}, expected {shape}"
        )
    if ndim is not None and value.ndim != ndim:
        raise FrameValidationError(
            f"{field} ndim is {value.ndim}, expected {ndim}"
        )
    if nonempty and value.size == 0:
        raise FrameValidationError(f"{field} must not be empty")
    if np.issubdtype(expected_dtype, np.floating) and not np.all(
        np.isfinite(value)
    ):
        raise FrameValidationError(f"{field} contains NaN or Inf")
    output = np.ascontiguousarray(value).copy()
    output.setflags(write=False)
    return output


def _optional_float_vector(value: object, field: str) -> np.ndarray | None:
    if value is None:
        return None
    return _array(
        value,
        field=field,
        dtype=np.float32,
        ndim=1,
        nonempty=True,
    )


def _optional_fixed_float(
    value: object,
    field: str,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if value is None:
        return None
    return _array(value, field=field, dtype=np.float32, shape=shape)


def _array_bytes(*values: np.ndarray | None) -> int:
    return sum(int(value.nbytes) for value in values if value is not None)


def _validate_rot6d(value: np.ndarray, field: str) -> None:
    rotation = value[3:9].astype(np.float64, copy=False)
    first = rotation[0:3]
    second = rotation[3:6]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= _ROT6D_NORM_EPSILON:
        raise FrameValidationError(f"{field} Rot6D first column must be nonzero")
    if second_norm <= _ROT6D_NORM_EPSILON:
        raise FrameValidationError(f"{field} Rot6D second column must be nonzero")
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    if abs(cosine) >= _ROT6D_MAX_ABS_COSINE:
        raise FrameValidationError(
            f"{field} Rot6D columns must not be collinear or near-collinear"
        )


def required_frame_count(history_len: object, current: object) -> int:
    if isinstance(history_len, bool) or not isinstance(history_len, Integral):
        raise ValueError("history_len must be an integer")
    if int(history_len) < 0:
        raise ValueError("history_len must be nonnegative")
    if not isinstance(current, (bool, np.bool_)):
        raise ValueError("current must be boolean")
    return int(history_len) + int(bool(current))


@dataclass(frozen=True, slots=True)
class CameraFrame:
    timestamp_ns: int
    encoding: str
    data: bytes
    valid: bool

    def __post_init__(self) -> None:
        timestamp = _timestamp_ns(self.timestamp_ns)
        if self.encoding != "jpeg":
            raise FrameValidationError("camera encoding must be 'jpeg'")
        if not isinstance(self.data, bytes):
            raise FrameValidationError("camera data must be bytes")
        valid = _boolean(self.valid, "camera valid")
        if valid and not self.data:
            raise FrameValidationError("valid camera frame has empty JPEG data")
        if not valid and self.data:
            raise FrameValidationError("invalid camera frame data must be empty")
        object.__setattr__(self, "timestamp_ns", timestamp)
        object.__setattr__(self, "data", bytes(self.data))
        object.__setattr__(self, "valid", valid)

    @property
    def byte_size(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class StateFrame:
    timestamp_ns: int
    left_joint: np.ndarray | None = None
    left_eef: np.ndarray | None = None
    left_eef_frame: str | None = None
    right_joint: np.ndarray | None = None
    right_eef: np.ndarray | None = None
    right_eef_frame: str | None = None
    left_hand_joint: np.ndarray | None = None
    right_hand_joint: np.ndarray | None = None
    valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _timestamp_ns(self.timestamp_ns))
        object.__setattr__(
            self,
            "left_joint",
            _optional_float_vector(self.left_joint, "state.left_joint"),
        )
        object.__setattr__(
            self,
            "right_joint",
            _optional_float_vector(self.right_joint, "state.right_joint"),
        )
        object.__setattr__(
            self,
            "left_eef",
            _optional_fixed_float(self.left_eef, "state.left_eef", (9,)),
        )
        object.__setattr__(
            self,
            "right_eef",
            _optional_fixed_float(self.right_eef, "state.right_eef", (9,)),
        )
        object.__setattr__(
            self,
            "left_hand_joint",
            _optional_fixed_float(
                self.left_hand_joint,
                "state.left_hand_joint",
                (22,),
            ),
        )
        object.__setattr__(
            self,
            "right_hand_joint",
            _optional_fixed_float(
                self.right_hand_joint,
                "state.right_hand_joint",
                (22,),
            ),
        )
        if self.left_eef is not None:
            _validate_rot6d(self.left_eef, "state.left_eef")
        if self.right_eef is not None:
            _validate_rot6d(self.right_eef, "state.right_eef")
        self._validate_eef_frame("left", self.left_eef, self.left_eef_frame)
        self._validate_eef_frame("right", self.right_eef, self.right_eef_frame)
        object.__setattr__(self, "valid", _boolean(self.valid, "state valid"))

    @staticmethod
    def _validate_eef_frame(
        side: str,
        eef: np.ndarray | None,
        frame: str | None,
    ) -> None:
        if eef is None and frame is not None:
            raise FrameValidationError(
                f"state.{side}_eef_frame must be None without EEF"
            )
        if eef is not None and frame != "robot_base":
            raise FrameValidationError(
                f"state.{side}_eef_frame must be 'robot_base'"
            )

    @property
    def byte_size(self) -> int:
        return _array_bytes(
            self.left_joint,
            self.left_eef,
            self.right_joint,
            self.right_eef,
            self.left_hand_joint,
            self.right_hand_joint,
        )


@dataclass(frozen=True, slots=True)
class TauFrame:
    timestamp_ns: int
    left: np.ndarray
    right: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _timestamp_ns(self.timestamp_ns))
        object.__setattr__(
            self,
            "left",
            _array(self.left, field="tau.left", dtype=np.float32, shape=(22,)),
        )
        object.__setattr__(
            self,
            "right",
            _array(self.right, field="tau.right", dtype=np.float32, shape=(22,)),
        )
        object.__setattr__(
            self,
            "left_valid",
            _array(
                self.left_valid,
                field="tau.left_valid",
                dtype=np.bool_,
                shape=(22,),
            ),
        )
        object.__setattr__(
            self,
            "right_valid",
            _array(
                self.right_valid,
                field="tau.right_valid",
                dtype=np.bool_,
                shape=(22,),
            ),
        )

    @property
    def byte_size(self) -> int:
        return _array_bytes(
            self.left,
            self.right,
            self.left_valid,
            self.right_valid,
        )


@dataclass(frozen=True, slots=True)
class WrenchFrame:
    timestamp_ns: int
    left: np.ndarray
    right: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _timestamp_ns(self.timestamp_ns))
        object.__setattr__(
            self,
            "left",
            _array(
                self.left,
                field="wrench.left",
                dtype=np.float32,
                shape=(5, 6),
            ),
        )
        object.__setattr__(
            self,
            "right",
            _array(
                self.right,
                field="wrench.right",
                dtype=np.float32,
                shape=(5, 6),
            ),
        )
        object.__setattr__(
            self,
            "left_valid",
            _array(
                self.left_valid,
                field="wrench.left_valid",
                dtype=np.bool_,
                shape=(5,),
            ),
        )
        object.__setattr__(
            self,
            "right_valid",
            _array(
                self.right_valid,
                field="wrench.right_valid",
                dtype=np.bool_,
                shape=(5,),
            ),
        )

    @property
    def byte_size(self) -> int:
        return _array_bytes(
            self.left,
            self.right,
            self.left_valid,
            self.right_valid,
        )


@dataclass(frozen=True, slots=True)
class DeformationFrame:
    timestamp_ns: int
    left: np.ndarray
    right: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _timestamp_ns(self.timestamp_ns))
        object.__setattr__(
            self,
            "left",
            _array(
                self.left,
                field="deformation.left",
                dtype=np.uint8,
                shape=(5, 240, 240),
            ),
        )
        object.__setattr__(
            self,
            "right",
            _array(
                self.right,
                field="deformation.right",
                dtype=np.uint8,
                shape=(5, 240, 240),
            ),
        )
        object.__setattr__(
            self,
            "left_valid",
            _array(
                self.left_valid,
                field="deformation.left_valid",
                dtype=np.bool_,
                shape=(5,),
            ),
        )
        object.__setattr__(
            self,
            "right_valid",
            _array(
                self.right_valid,
                field="deformation.right_valid",
                dtype=np.bool_,
                shape=(5,),
            ),
        )

    @property
    def byte_size(self) -> int:
        return _array_bytes(
            self.left,
            self.right,
            self.left_valid,
            self.right_valid,
        )


FrameT = TypeVar(
    "FrameT",
    CameraFrame,
    StateFrame,
    TauFrame,
    WrenchFrame,
    DeformationFrame,
)


class TimestampedBuffer(Generic[FrameT]):
    """A deduplicating buffer ordered by source timestamp, oldest first."""

    def __init__(
        self,
        frame_type: type[FrameT],
        *,
        frame_capacity: int,
        byte_capacity: int,
    ) -> None:
        if isinstance(frame_capacity, bool) or not isinstance(
            frame_capacity, Integral
        ):
            raise ValueError("frame_capacity must be an integer")
        if isinstance(byte_capacity, bool) or not isinstance(byte_capacity, Integral):
            raise ValueError("byte_capacity must be an integer")
        if int(frame_capacity) <= 0:
            raise ValueError("frame_capacity must be positive")
        if int(byte_capacity) <= 0:
            raise ValueError("byte_capacity must be positive")
        self._frame_type = frame_type
        self._frame_capacity = int(frame_capacity)
        self._byte_capacity = int(byte_capacity)
        self._timestamps: list[int] = []
        self._frames: list[FrameT] = []
        self._total_bytes = 0
        self._lock = threading.RLock()

    @property
    def frame_capacity(self) -> int:
        return self._frame_capacity

    @property
    def byte_capacity(self) -> int:
        return self._byte_capacity

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def timestamps(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._timestamps)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def append(self, frame: FrameT) -> bool:
        """Insert or replace by timestamp and return whether it remains stored."""

        if not isinstance(frame, self._frame_type):
            raise TypeError(
                f"expected {self._frame_type.__name__}, got {type(frame).__name__}"
            )
        frame_bytes = int(frame.byte_size)
        if frame_bytes > self._byte_capacity:
            raise BufferCapacityError(
                f"frame uses {frame_bytes} bytes, buffer limit is "
                f"{self._byte_capacity}"
            )
        with self._lock:
            index = bisect_left(self._timestamps, frame.timestamp_ns)
            if (
                index < len(self._timestamps)
                and self._timestamps[index] == frame.timestamp_ns
            ):
                self._total_bytes -= int(self._frames[index].byte_size)
                self._frames[index] = frame
                self._total_bytes += frame_bytes
            else:
                self._timestamps.insert(index, frame.timestamp_ns)
                self._frames.insert(index, frame)
                self._total_bytes += frame_bytes
            self._evict_oldest_locked()
            retained_index = bisect_left(self._timestamps, frame.timestamp_ns)
            return bool(
                retained_index < len(self._timestamps)
                and self._timestamps[retained_index] == frame.timestamp_ns
            )

    def clear(self) -> None:
        with self._lock:
            self._timestamps.clear()
            self._frames.clear()
            self._total_bytes = 0

    def frames(self) -> tuple[FrameT, ...]:
        with self._lock:
            return tuple(self._frames)

    def select(
        self,
        *,
        history_len: int,
        current: bool,
    ) -> tuple[tuple[FrameT, ...], FrameT | None]:
        count = required_frame_count(history_len, current)
        with self._lock:
            if len(self._frames) < count:
                raise BufferUnderflowError(
                    required=count,
                    available=len(self._frames),
                )
            if count == 0:
                return (), None
            selected = tuple(self._frames[-count:])
        history = selected[:-1] if current else selected
        current_frame = selected[-1] if current else None
        return history, current_frame

    def _evict_oldest_locked(self) -> None:
        while (
            len(self._frames) > self._frame_capacity
            or self._total_bytes > self._byte_capacity
        ):
            evicted = self._frames.pop(0)
            self._timestamps.pop(0)
            self._total_bytes -= int(evicted.byte_size)


class CameraBuffer(TimestampedBuffer[CameraFrame]):
    def __init__(self, *, frame_capacity: int, byte_capacity: int) -> None:
        super().__init__(
            CameraFrame,
            frame_capacity=frame_capacity,
            byte_capacity=byte_capacity,
        )


class StateBuffer(TimestampedBuffer[StateFrame]):
    def __init__(self, *, frame_capacity: int, byte_capacity: int) -> None:
        super().__init__(
            StateFrame,
            frame_capacity=frame_capacity,
            byte_capacity=byte_capacity,
        )


class TauBuffer(TimestampedBuffer[TauFrame]):
    def __init__(self, *, frame_capacity: int, byte_capacity: int) -> None:
        super().__init__(
            TauFrame,
            frame_capacity=frame_capacity,
            byte_capacity=byte_capacity,
        )


class WrenchBuffer(TimestampedBuffer[WrenchFrame]):
    def __init__(self, *, frame_capacity: int, byte_capacity: int) -> None:
        super().__init__(
            WrenchFrame,
            frame_capacity=frame_capacity,
            byte_capacity=byte_capacity,
        )


class DeformationBuffer(TimestampedBuffer[DeformationFrame]):
    def __init__(self, *, frame_capacity: int, byte_capacity: int) -> None:
        super().__init__(
            DeformationFrame,
            frame_capacity=frame_capacity,
            byte_capacity=byte_capacity,
        )


class ObservationBuffers:
    """Own all continuously collected v3 source buffers under one lock."""

    def __init__(
        self,
        *,
        camera_frame_capacity: int = 3,
        camera_byte_capacity: int = 64 * 1024 * 1024,
        state_frame_capacity: int = 19,
        state_byte_capacity: int = 16 * 1024 * 1024,
        tau_frame_capacity: int = 19,
        tau_byte_capacity: int = 16 * 1024 * 1024,
        wrench_frame_capacity: int = 19,
        wrench_byte_capacity: int = 16 * 1024 * 1024,
        deformation_frame_capacity: int = 3,
        deformation_byte_capacity: int = 64 * 1024 * 1024,
    ) -> None:
        self._lock = threading.RLock()
        self.cameras = {
            name: CameraBuffer(
                frame_capacity=camera_frame_capacity,
                byte_capacity=camera_byte_capacity,
            )
            for name in CAMERA_NAMES
        }
        self.state = StateBuffer(
            frame_capacity=state_frame_capacity,
            byte_capacity=state_byte_capacity,
        )
        self.tau = TauBuffer(
            frame_capacity=tau_frame_capacity,
            byte_capacity=tau_byte_capacity,
        )
        self.wrench = WrenchBuffer(
            frame_capacity=wrench_frame_capacity,
            byte_capacity=wrench_byte_capacity,
        )
        self.deformation = DeformationBuffer(
            frame_capacity=deformation_frame_capacity,
            byte_capacity=deformation_byte_capacity,
        )

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            yield

    def camera(self, name: str) -> CameraBuffer:
        try:
            return self.cameras[name]
        except KeyError as exc:
            raise ValueError(f"unknown camera: {name!r}") from exc

    def push_camera(self, name: str, frame: CameraFrame) -> bool:
        with self._lock:
            return self.camera(name).append(frame)

    def push_state(self, frame: StateFrame) -> bool:
        with self._lock:
            return self.state.append(frame)

    def push_tau(self, frame: TauFrame) -> bool:
        with self._lock:
            return self.tau.append(frame)

    def push_wrench(self, frame: WrenchFrame) -> bool:
        with self._lock:
            return self.wrench.append(frame)

    def push_deformation(self, frame: DeformationFrame) -> bool:
        with self._lock:
            return self.deformation.append(frame)

    def clear_source(self, source: str) -> None:
        with self._lock:
            if source in self.cameras:
                self.cameras[source].clear()
            elif source == "state":
                self.state.clear()
            elif source == "tau":
                self.tau.clear()
            elif source == "wrench":
                self.wrench.clear()
            elif source == "deformation":
                self.deformation.clear()
            else:
                raise ValueError(f"unknown observation source: {source!r}")

    def clear(self) -> None:
        with self._lock:
            for camera in self.cameras.values():
                camera.clear()
            self.state.clear()
            self.tau.clear()
            self.wrench.clear()
            self.deformation.clear()
