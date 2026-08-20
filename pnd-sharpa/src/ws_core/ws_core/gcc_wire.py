"""Build the GCC N1.7 request from selected 30 Hz workstation samples.

The workstation deliberately keeps this representation in the PND recording
order.  Joint reordering, delta-q construction, sensor normalization, wrist
calibration, and model-specific decoding belong to the GCC server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


GCC_REQUEST_SCHEMA = "gcc_n17_sharpa62_observation.v1"
GCC_HISTORY_FRAMES = 9
SHARPA_JOINTS = 44
TACTILE_FINGERS = 10
WRENCH_DIM = 6
DEFORMATION_SIZE = 64

FINGER_ORDER = (
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb",
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb",
)


@dataclass(frozen=True)
class GccHistoryFrame:
    """The only fields retained across `/ws/obs` ticks for GCC."""

    obs_seq: int
    obs_stamp_ns: int
    timestamp_unix_s: float
    q_exe: np.ndarray
    q_exe_valid: np.ndarray
    q_cmd: np.ndarray
    q_cmd_valid: np.ndarray
    tau: np.ndarray
    tau_valid: np.ndarray
    tactile_wrench: np.ndarray
    tactile_wrench_valid: np.ndarray
    joint_order: tuple[str, ...]
    joint_layout: str
    tactile_order: tuple[str, ...]
    tactile_layout: str


def extract_gcc_history_frame(
    observation: Any,
    *,
    obs_seq: int,
    obs_stamp_ns: int,
    timestamp_unix_s: float,
    joint_max_age_ms: float | None = None,
    wrench_max_age_ms: float | None = None,
) -> GccHistoryFrame:
    """Extract a selective GCC history frame from one current PolicyObs."""

    if not isinstance(observation, dict):
        raise ValueError("PolicyObs payload must be a JSON object")
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        raise ValueError("PolicyObs payload is missing robot_state")
    state = _robot_state_json(observation)
    sharpa = state.get("sharpa") if isinstance(state, dict) else None
    tactile = state.get("tactile") if isinstance(state, dict) else None
    if not isinstance(sharpa, dict):
        raise ValueError("PolicyObs robot_state is missing sharpa facts")
    if not isinstance(tactile, dict):
        raise ValueError("PolicyObs robot_state is missing tactile facts")

    q_exe = _float_array(sharpa.get("q_exe"), (SHARPA_JOINTS,), "sharpa.q_exe")
    q_exe_valid = _bool_array(
        sharpa.get("q_exe_valid"),
        (SHARPA_JOINTS,),
        "sharpa.q_exe_valid",
    )
    q_cmd = _float_array(sharpa.get("q_cmd"), (SHARPA_JOINTS,), "sharpa.q_cmd")
    q_cmd_valid = _bool_array(
        sharpa.get("q_cmd_valid"),
        (SHARPA_JOINTS,),
        "sharpa.q_cmd_valid",
    )
    tau = _float_array(sharpa.get("tau"), (SHARPA_JOINTS,), "sharpa.tau")
    tau_valid = _bool_array(
        sharpa.get("tau_valid"),
        (SHARPA_JOINTS,),
        "sharpa.tau_valid",
    )
    if not _combined_age_is_fresh(
        source_age_ms=sharpa.get("age_ms"),
        transport_age_ms=robot_state.get("age_ms"),
        max_age_ms=joint_max_age_ms,
        source_field="sharpa.age_ms",
        transport_field="robot_state.age_ms",
    ):
        q_exe_valid[:] = False
        q_cmd_valid[:] = False
        tau_valid[:] = False

    joint_order_raw = sharpa.get("joint_order", sharpa.get("name"))
    joint_order = _string_order(
        joint_order_raw,
        SHARPA_JOINTS,
        "sharpa.joint_order",
    )
    joint_layout = str(sharpa.get("joint_layout") or "")
    if not joint_layout:
        raise ValueError("sharpa.joint_layout is missing")

    tactile_order = _string_order(
        tactile.get("order"),
        TACTILE_FINGERS,
        "tactile.order",
    )
    tactile_layout = str(tactile.get("tactile_layout") or "")
    if not tactile_layout:
        raise ValueError("tactile.tactile_layout is missing")
    wrench_raw = tactile.get("wrench")
    if wrench_raw is None:
        force = _float_array(
            tactile.get("force"),
            (TACTILE_FINGERS, 3),
            "tactile.force",
        )
        torque = _float_array(
            tactile.get("torque"),
            (TACTILE_FINGERS, 3),
            "tactile.torque",
        )
        wrench_raw = np.concatenate((force, torque), axis=1)
    wrench = _float_array(
        wrench_raw,
        (TACTILE_FINGERS, WRENCH_DIM),
        "tactile.wrench",
    )
    wrench_valid = _bool_array(
        tactile.get("wrench_valid", tactile.get("force_valid")),
        (TACTILE_FINGERS,),
        "tactile.wrench_valid",
    )
    if not _combined_age_is_fresh(
        source_age_ms=tactile.get("force_age_ms"),
        transport_age_ms=robot_state.get("age_ms"),
        max_age_ms=wrench_max_age_ms,
        source_field="tactile.force_age_ms",
        transport_field="robot_state.age_ms",
    ):
        wrench_valid[:] = False
    wrench, wrench_valid = _reorder_tactile(
        wrench,
        wrench_valid,
        tactile_order,
    )

    return GccHistoryFrame(
        obs_seq=int(obs_seq),
        obs_stamp_ns=int(obs_stamp_ns),
        timestamp_unix_s=float(timestamp_unix_s),
        q_exe=q_exe,
        q_exe_valid=q_exe_valid,
        q_cmd=q_cmd,
        q_cmd_valid=q_cmd_valid,
        tau=tau,
        tau_valid=tau_valid,
        tactile_wrench=wrench,
        tactile_wrench_valid=wrench_valid,
        joint_order=joint_order,
        joint_layout=joint_layout,
        tactile_order=tuple(FINGER_ORDER),
        tactile_layout=tactile_layout,
    )


def build_gcc_request(
    history: Sequence[GccHistoryFrame],
    *,
    current_observation: Any,
    current_tactile_data: bytes,
    current_hand_pose_62d: np.ndarray,
    current_ego_view_jpeg: bytes,
    current_obs_seq: int,
    current_timestamp_unix_s: float,
    session_id: str,
    prompt: str = "",
    history_is_real: Any | None = None,
    require_full_real_history: bool = True,
    deformation_max_age_ms: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and stack one oldest-to-newest GCC request.

    Padding is representable for offline fixtures, but deployment callers use
    ``require_full_real_history=True`` and therefore wait for nine real ticks.
    Any padded position has all of its history validity masks forced false.
    """

    frames = tuple(history)
    if len(frames) != GCC_HISTORY_FRAMES:
        message = (
            f"GCC history has {len(frames)} frames, "
            f"expected {GCC_HISTORY_FRAMES}"
        )
        raise ValueError(message)
    _validate_history_order(frames)
    if frames[-1].obs_seq != int(current_obs_seq):
        raise ValueError(
            "GCC current observation does not match the newest history frame: "
            f"current={current_obs_seq}, history={frames[-1].obs_seq}"
        )
    _validate_history_layout(frames)

    if history_is_real is None:
        real = np.ones(GCC_HISTORY_FRAMES, dtype=bool)
    else:
        real = _bool_array(
            history_is_real,
            (GCC_HISTORY_FRAMES,),
            "history_is_real",
        )
    real_count = int(real.sum())
    if require_full_real_history and real_count != GCC_HISTORY_FRAMES:
        raise ValueError(
            f"GCC requires {GCC_HISTORY_FRAMES} real frames, got {real_count}"
        )

    hand_pose = _float_array(
        current_hand_pose_62d,
        (62,),
        "current_hand_pose_62d",
    )
    q_exe = np.stack([frame.q_exe for frame in frames]).astype(
        np.float32, copy=False
    )
    q_exe_valid = np.stack([frame.q_exe_valid for frame in frames])
    q_cmd = np.stack([frame.q_cmd for frame in frames]).astype(
        np.float32, copy=False
    )
    q_cmd_valid = np.stack([frame.q_cmd_valid for frame in frames])
    tau = np.stack([frame.tau for frame in frames]).astype(
        np.float32,
        copy=False,
    )
    tau_valid = np.stack([frame.tau_valid for frame in frames])
    wrench = np.stack([frame.tactile_wrench for frame in frames]).astype(
        np.float32, copy=False
    )
    wrench_valid = np.stack(
        [frame.tactile_wrench_valid for frame in frames]
    )

    # Repeated padding values may provide a fixed shape, but never valid data.
    q_exe_valid &= real[:, None]
    q_cmd_valid &= real[:, None]
    tau_valid &= real[:, None]
    wrench_valid &= real[:, None]

    deformation, deformation_valid = extract_current_deformation(
        current_observation,
        current_tactile_data,
        max_age_ms=deformation_max_age_ms,
    )
    obs_seqs = np.asarray([frame.obs_seq for frame in frames], dtype=np.int64)
    obs_stamps = np.asarray(
        [frame.obs_stamp_ns for frame in frames],
        dtype=np.int64,
    )
    timestamps = np.asarray(
        [frame.timestamp_unix_s for frame in frames],
        dtype=np.float64,
    )

    request: dict[str, Any] = {
        "schema": GCC_REQUEST_SCHEMA,
        "endpoint": "infer",
        "session_id": str(session_id),
        "observation/ego_view_jpeg": bytes(current_ego_view_jpeg),
        "observation/hand_pose_62d": hand_pose,
        "observation/timestamp_unix_s": float(current_timestamp_unix_s),
        "observation/q_exe_history_9x44": q_exe,
        "observation/q_exe_valid_history_9x44": q_exe_valid,
        "observation/q_cmd_history_9x44": q_cmd,
        "observation/q_cmd_valid_history_9x44": q_cmd_valid,
        "observation/tau_history_9x44": tau,
        "observation/tau_valid_history_9x44": tau_valid,
        "observation/tactile_wrench_history_9x10x6": wrench,
        "observation/tactile_wrench_valid_history_9x10": wrench_valid,
        "observation/tactile_deformation_10x64x64": deformation,
        "observation/tactile_deformation_valid_10": deformation_valid,
        "history_obs_seq_9": obs_seqs,
        "history_stamp_ns_9": obs_stamps,
        "history_timestamp_unix_s_9": timestamps,
        "history_is_real_9": real,
        "history_real_count": np.asarray(
            real_count,
            dtype=np.int64,
        ),
        "joint_order": list(frames[-1].joint_order),
        "joint_layout": frames[-1].joint_layout,
        "tactile_order": list(FINGER_ORDER),
        "tactile_layout": frames[-1].tactile_layout,
    }
    if prompt:
        request["prompt"] = str(prompt)

    info = {
        "schema": "ws.gcc_n17_sharpa62_request_info.v1",
        "request_schema": GCC_REQUEST_SCHEMA,
        "history_obs_seqs": obs_seqs.tolist(),
        "history_stamp_ns": obs_stamps.tolist(),
        "history_real_count": real_count,
        "q_exe_valid": int(q_exe_valid.sum()),
        "q_cmd_valid": int(q_cmd_valid.sum()),
        "tau_valid": int(tau_valid.sum()),
        "tactile_wrench_valid": int(wrench_valid.sum()),
        "tactile_deformation_valid": int(deformation_valid.sum()),
    }
    return request, info


def extract_current_deformation(
    observation: Any,
    tactile_data: bytes,
    *,
    max_age_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the current ten deformation images resized with INTER_AREA."""

    output = np.zeros(
        (TACTILE_FINGERS, DEFORMATION_SIZE, DEFORMATION_SIZE),
        dtype=np.uint8,
    )
    valid = np.zeros(TACTILE_FINGERS, dtype=bool)
    if not isinstance(observation, dict):
        return output, valid
    robot_tactile = observation.get("robot_tactile")
    if not isinstance(robot_tactile, dict) or not _age_is_fresh(
        robot_tactile.get("age_ms"),
        max_age_ms,
        "robot_tactile.age_ms",
    ):
        return output, valid
    metadata = (
        _unwrap_json(robot_tactile.get("metadata"))
        if isinstance(robot_tactile, dict)
        else None
    )
    entries = metadata.get("entries") if isinstance(metadata, dict) else None
    if not isinstance(entries, list) or not tactile_data:
        return output, valid

    raw = memoryview(tactile_data)
    destination = {name: index for index, name in enumerate(FINGER_ORDER)}
    for entry in entries:
        if not isinstance(entry, dict) or not bool(entry.get("valid")):
            continue
        name = f"{entry.get('side', '')}_{entry.get('finger', '')}"
        target_index = destination.get(name)
        if target_index is None:
            continue
        try:
            raw_offset = (
                entry["raw_offset"]
                if "raw_offset" in entry
                else entry["offset"]
            )
            raw_length = (
                entry["raw_length"]
                if "raw_length" in entry
                else entry["length"]
            )
            offset = int(raw_offset)
            length = int(raw_length)
            height = int(entry["height"])
            width = int(entry["width"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            offset < 0
            or length <= 0
            or height <= 0
            or width <= 0
            or length != height * width
            or offset + length > len(raw)
        ):
            continue
        image = np.frombuffer(
            raw[offset:offset + length],
            dtype=np.uint8,
        ).reshape(height, width)
        if image.shape != (DEFORMATION_SIZE, DEFORMATION_SIZE):
            import cv2

            image = cv2.resize(
                image,
                (DEFORMATION_SIZE, DEFORMATION_SIZE),
                interpolation=cv2.INTER_AREA,
            )
        output[target_index] = image
        valid[target_index] = True
    return output, valid


def _robot_state_json(observation: dict[str, Any]) -> dict[str, Any]:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        raise ValueError("PolicyObs payload is missing robot_state")
    state = _unwrap_json(robot_state.get("payload"))
    if not isinstance(state, dict):
        raise ValueError("PolicyObs robot_state payload is not valid JSON")
    return state


def _unwrap_json(value: Any) -> Any:
    if (
        isinstance(value, dict)
        and value.get("valid") is True
        and "json" in value
    ):
        return value["json"]
    return value


def _float_array(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if array.shape != shape:
        raise ValueError(f"{field} shape is {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} contains NaN or Inf")
    return array.copy()


def _bool_array(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{field} shape is {array.shape}, expected {shape}")
    if array.dtype.kind != "b":
        # JSON booleans become bool.  Reject numeric masks instead of silently
        # accepting a malformed or ambiguous contract.
        raise ValueError(f"{field} must contain booleans")
    return array.astype(bool, copy=True)


def _age_is_fresh(
    value: Any,
    max_age_ms: float | None,
    field: str,
) -> bool:
    if max_age_ms is None:
        return True
    try:
        age_ms = float(value)
        limit_ms = float(max_age_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not np.isfinite(limit_ms) or limit_ms <= 0.0:
        raise ValueError("freshness limit must be finite and positive")
    return bool(np.isfinite(age_ms) and 0.0 <= age_ms <= limit_ms)


def _combined_age_is_fresh(
    *,
    source_age_ms: Any,
    transport_age_ms: Any,
    max_age_ms: float | None,
    source_field: str,
    transport_field: str,
) -> bool:
    if max_age_ms is None:
        return True
    try:
        source_age = float(source_age_ms)
        transport_age = float(transport_age_ms)
        limit_ms = float(max_age_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_field} and {transport_field} must be numeric"
        ) from exc
    if not np.isfinite(limit_ms) or limit_ms <= 0.0:
        raise ValueError("freshness limit must be finite and positive")
    effective_age = source_age + transport_age
    return bool(
        np.isfinite(source_age)
        and source_age >= 0.0
        and np.isfinite(transport_age)
        and transport_age >= 0.0
        and effective_age <= limit_ms
    )


def _string_order(value: Any, size: int, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{field} must contain exactly {size} names")
    order = tuple(str(item) for item in value)
    if any(not item for item in order) or len(set(order)) != size:
        raise ValueError(f"{field} contains empty or duplicate names")
    return order


def _reorder_tactile(
    wrench: np.ndarray,
    valid: np.ndarray,
    source_order: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    source = {name: index for index, name in enumerate(source_order)}
    if set(source) != set(FINGER_ORDER):
        raise ValueError(
            "tactile.order does not match the required ten-finger layout"
        )
    indices = [source[name] for name in FINGER_ORDER]
    return wrench[indices].copy(), valid[indices].copy()


def _validate_history_order(frames: tuple[GccHistoryFrame, ...]) -> None:
    seqs = [frame.obs_seq for frame in frames]
    stamps = [frame.obs_stamp_ns for frame in frames]
    if any(current <= previous for previous, current in zip(seqs, seqs[1:])):
        raise ValueError("GCC history obs_seq is not strictly increasing")
    if any(
        current <= previous
        for previous, current in zip(stamps, stamps[1:])
    ):
        raise ValueError("GCC history stamp_ns is not strictly increasing")


def _validate_history_layout(frames: tuple[GccHistoryFrame, ...]) -> None:
    first = frames[0]
    for frame in frames[1:]:
        if frame.joint_order != first.joint_order:
            raise ValueError(
                "GCC history joint order changed inside the window"
            )
        if frame.joint_layout != first.joint_layout:
            raise ValueError(
                "GCC history joint layout changed inside the window"
            )
        if frame.tactile_order != first.tactile_order:
            raise ValueError(
                "GCC history tactile order changed inside the window"
            )
        if frame.tactile_layout != first.tactile_layout:
            raise ValueError(
                "GCC history tactile layout changed inside the window"
            )
