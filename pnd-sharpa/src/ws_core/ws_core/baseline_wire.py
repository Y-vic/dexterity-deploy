"""Build T-Rex and ViTacFormer requests from 30 Hz workstation samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ws_core.gcc_wire import FINGER_ORDER, extract_current_deformation


BASELINE_REQUEST_SCHEMA = "dreamzero_sharpa62_observation.v1"
BASELINE_HISTORY_FRAMES = 18
TREX_HISTORY_FRAMES = 16
VITACFORMER_HISTORY_FRAMES = 18
HAND_POSE_DIM = 62
TACTILE_FINGERS = 10
WRENCH_DIM = 6

TREX_PROVIDERS = {"trex", "t_rex"}
VITACFORMER_PROVIDERS = {"vitacformer", "vitac"}


@dataclass(frozen=True)
class BaselineHistoryFrame:
    obs_seq: int
    obs_stamp_ns: int
    timestamp_unix_s: float
    hand_pose_62d: np.ndarray
    tactile_wrench: np.ndarray
    tactile_wrench_valid: np.ndarray
    tactile_order: tuple[str, ...]
    tactile_layout: str


def extract_baseline_history_frame(
    observation: Any,
    *,
    hand_pose_62d: Any,
    obs_seq: int,
    obs_stamp_ns: int,
    timestamp_unix_s: float,
    wrench_max_age_ms: float | None = None,
) -> BaselineHistoryFrame:
    """Extract state and tactile wrench facts from one real `/ws/obs` tick."""

    if not isinstance(observation, dict):
        raise ValueError("PolicyObs payload must be a JSON object")
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        raise ValueError("PolicyObs payload is missing robot_state")
    state = _unwrap_json(robot_state.get("payload"))
    if not isinstance(state, dict):
        raise ValueError("PolicyObs robot_state payload is not valid JSON")
    tactile = state.get("tactile")
    if not isinstance(tactile, dict):
        raise ValueError("PolicyObs robot_state is missing tactile facts")

    hand_pose = _float_array(
        hand_pose_62d,
        (HAND_POSE_DIM,),
        "hand_pose_62d",
    )
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
    ):
        wrench_valid[:] = False

    wrench, wrench_valid = _reorder_tactile(
        wrench,
        wrench_valid,
        tactile_order,
    )
    wrench[~wrench_valid] = 0.0
    return BaselineHistoryFrame(
        obs_seq=int(obs_seq),
        obs_stamp_ns=int(obs_stamp_ns),
        timestamp_unix_s=float(timestamp_unix_s),
        hand_pose_62d=hand_pose,
        tactile_wrench=wrench,
        tactile_wrench_valid=wrench_valid,
        tactile_order=tuple(FINGER_ORDER),
        tactile_layout=tactile_layout,
    )


def build_baseline_request(
    history: Sequence[BaselineHistoryFrame],
    *,
    provider: str,
    current_observation: Any,
    current_tactile_data: bytes,
    current_hand_pose_62d: Any,
    current_ego_view_jpeg: bytes,
    current_obs_seq: int,
    current_timestamp_unix_s: float,
    session_id: str,
    prompt: str = "",
    deformation_max_age_ms: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one strict baseline request from oldest-to-newest real frames."""

    provider_key = str(provider).strip().lower()
    if provider_key in TREX_PROVIDERS:
        required_frames = TREX_HISTORY_FRAMES
        state_frames = 1
        policy_family = "trex"
    elif provider_key in VITACFORMER_PROVIDERS:
        required_frames = VITACFORMER_HISTORY_FRAMES
        state_frames = TREX_HISTORY_FRAMES
        policy_family = "vitacformer"
    else:
        raise ValueError(f"unsupported baseline provider: {provider!r}")

    frames = tuple(history)
    if len(frames) < required_frames:
        raise ValueError(
            f"{policy_family} history has {len(frames)} frames, "
            f"expected at least {required_frames}"
        )
    frames = frames[-required_frames:]
    _validate_history(frames, current_obs_seq=current_obs_seq)

    hand_pose = _float_array(
        current_hand_pose_62d,
        (HAND_POSE_DIM,),
        "current_hand_pose_62d",
    )
    wrench = np.stack([frame.tactile_wrench for frame in frames]).astype(
        np.float32,
        copy=False,
    )
    wrench_valid = np.stack([frame.tactile_wrench_valid for frame in frames])
    wrench[~wrench_valid] = 0.0
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
        "schema": BASELINE_REQUEST_SCHEMA,
        "endpoint": "infer",
        "session_id": str(session_id),
        "observation/ego_view_jpeg": bytes(current_ego_view_jpeg),
        "observation/timestamp_unix_s": float(current_timestamp_unix_s),
        "tactile_order": list(FINGER_ORDER),
        "tactile_layout": frames[-1].tactile_layout,
    }
    if prompt:
        request["prompt"] = str(prompt)

    wrench_key = (
        f"observation/tactile_wrench_history_"
        f"{required_frames}x{TACTILE_FINGERS}x{WRENCH_DIM}"
    )
    wrench_valid_key = (
        f"observation/tactile_wrench_valid_history_"
        f"{required_frames}x{TACTILE_FINGERS}"
    )
    request[wrench_key] = wrench
    request[wrench_valid_key] = wrench_valid

    if policy_family == "trex":
        request["observation/hand_pose_62d"] = hand_pose
        deformation, deformation_valid = extract_current_deformation(
            current_observation,
            current_tactile_data,
            max_age_ms=deformation_max_age_ms,
        )
        request["observation/tactile_deformation_10x64x64"] = deformation
        request["observation/tactile_deformation_valid_10"] = deformation_valid
    else:
        state_history = np.stack(
            [frame.hand_pose_62d for frame in frames[-state_frames:]]
        ).astype(np.float32, copy=False)
        request["observation/hand_pose_history_16x62"] = state_history

    info = {
        "schema": "ws.baseline_sharpa62_request_info.v1",
        "provider": policy_family,
        "history_obs_seqs": obs_seqs.tolist(),
        "history_stamp_ns": obs_stamps.tolist(),
        "history_timestamp_unix_s": timestamps.tolist(),
        "history_frame_count": len(frames),
        "state_history_frame_count": state_frames,
        "tactile_wrench_valid": int(wrench_valid.sum()),
    }
    if policy_family == "trex":
        info["tactile_deformation_valid"] = int(
            request["observation/tactile_deformation_valid_10"].sum()
        )
    return request, info


def _validate_history(
    frames: tuple[BaselineHistoryFrame, ...],
    *,
    current_obs_seq: int,
) -> None:
    seqs = [frame.obs_seq for frame in frames]
    stamps = [frame.obs_stamp_ns for frame in frames]
    if any(current <= previous for previous, current in zip(seqs, seqs[1:])):
        raise ValueError("baseline history obs_seq is not strictly increasing")
    if any(current <= previous for previous, current in zip(stamps, stamps[1:])):
        raise ValueError("baseline history stamp_ns is not strictly increasing")
    if seqs[-1] != int(current_obs_seq):
        raise ValueError(
            "current observation does not match the newest baseline history "
            f"frame: current={current_obs_seq}, history={seqs[-1]}"
        )
    layout = frames[0].tactile_layout
    for frame in frames:
        if frame.tactile_order != tuple(FINGER_ORDER):
            raise ValueError("baseline history has an unexpected tactile order")
        if frame.tactile_layout != layout:
            raise ValueError(
                "baseline history tactile layout changed inside the window"
            )


def _unwrap_json(value: Any) -> Any:
    if isinstance(value, dict) and value.get("valid") is True and "json" in value:
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
        raise ValueError(f"{field} must contain booleans")
    return array.astype(bool, copy=True)


def _string_order(value: Any, size: int, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{field} must contain exactly {size} names")
    order = tuple(str(item) for item in value)
    if any(not item for item in order) or len(set(order)) != size:
        raise ValueError(f"{field} contains empty or duplicate names")
    return order


def _combined_age_is_fresh(
    *,
    source_age_ms: Any,
    transport_age_ms: Any,
    max_age_ms: float | None,
) -> bool:
    if max_age_ms is None:
        return True
    try:
        source_age = float(source_age_ms)
        transport_age = float(transport_age_ms)
        limit_ms = float(max_age_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("tactile and robot-state ages must be numeric") from exc
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
