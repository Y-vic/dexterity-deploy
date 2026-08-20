"""SharpA v3 workstation wire adapter.

This module deliberately contains no ROS dependencies.  It converts the
workstation's fixed observation facts into the metadata-driven public policy
protocol and converts a validated server action back to the local 62D action
boundary used by ``action_ik``.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SERVER_SCHEMA = "sharpa_policy_server.v3"
OBSERVATION_SCHEMA = "sharpa_policy_observation.v3"
ACTION_SCHEMA = "sharpa_policy_action.v4"
ERROR_SCHEMA = "sharpa_policy_error.v1"
METADATA_FORMAT_SCHEMA = "sharpa_policy_metadata_format.v1"

HAND_JOINTS_PER_SIDE = 22
TACTILE_FINGERS_PER_SIDE = 5
DEFORMATION_SIZE = 240

HAND_JOINT_SUFFIXES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
HAND_JOINT_ORDER = tuple(
    f"{side}_{suffix}"
    for side in ("left", "right")
    for suffix in HAND_JOINT_SUFFIXES
)
FINGER_SUFFIXES = ("pinky", "ring", "middle", "index", "thumb")
TACTILE_ORDER = tuple(
    f"{side}_{finger}"
    for side in ("left", "right")
    for finger in FINGER_SUFFIXES
)
HISTORY_STREAM_NAMES = (
    "ego_cam",
    "left_wrist_cam",
    "right_wrist_cam",
    "state",
    "tau",
    "wrench",
    "deformation",
)
HISTORY_STREAM_MAX_CAPACITIES: dict[str, int] = {
    "ego_cam": 3,
    "left_wrist_cam": 3,
    "right_wrist_cam": 3,
    "state": 19,
    "tau": 19,
    "wrench": 19,
    "deformation": 3,
}


class SharpaV3ProtocolError(ValueError):
    """Raised when local facts or a server message violate the v3 contract."""


class SharpaV3ServerError(RuntimeError):
    """Structured error returned by the policy server over ``/infer``."""

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
class SharpaV3Frame:
    """One real workstation tick in semantic left/right order."""

    obs_seq: int
    timestamp_ns: int
    image_jpeg: bytes
    image_valid: bool
    left_eef: np.ndarray
    right_eef: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    state_valid: bool
    left_tau: np.ndarray
    right_tau: np.ndarray
    left_tau_valid: np.ndarray
    right_tau_valid: np.ndarray
    left_wrench: np.ndarray
    right_wrench: np.ndarray
    left_wrench_valid: np.ndarray
    right_wrench_valid: np.ndarray
    left_deformation: np.ndarray
    right_deformation: np.ndarray
    left_deformation_valid: np.ndarray
    right_deformation_valid: np.ndarray


@dataclass(frozen=True)
class _CameraHistoryFrame:
    obs_seq: int
    timestamp_ns: int
    jpeg: bytes
    valid: bool


@dataclass(frozen=True)
class _StateHistoryFrame:
    obs_seq: int
    timestamp_ns: int
    left_eef: np.ndarray
    right_eef: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    valid: bool


@dataclass(frozen=True)
class _SensorHistoryFrame:
    obs_seq: int
    timestamp_ns: int
    left: np.ndarray
    right: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray


@dataclass(frozen=True)
class SharpaV3HistorySnapshot:
    frames: tuple[SharpaV3Frame, ...]
    metadata_format: dict[str, Any]
    generation: int
    format_revision: int
    anchor_obs_seq: int
    anchor_timestamp_ns: int
    stream_lengths: dict[str, int]
    stream_capacities: dict[str, int]
    stream_required_lengths: dict[str, int]


class SharpaV3History:
    """Seven fixed-capacity buffers continuously sharing one observation clock."""

    def __init__(self) -> None:
        self._buffers: dict[str, deque[Any]] = {
            name: deque(maxlen=HISTORY_STREAM_MAX_CAPACITIES[name])
            for name in HISTORY_STREAM_NAMES
        }
        self._metadata_format: dict[str, Any] | None = None
        self._stream_required_lengths = {
            name: 0 for name in HISTORY_STREAM_NAMES
        }
        self._last_obs_seq: int | None = None
        self._last_timestamp_ns: int | None = None
        self._generation = 0
        self._format_revision = 0

    @property
    def configured(self) -> bool:
        return self._metadata_format is not None

    @property
    def last_obs_seq(self) -> int | None:
        return self._last_obs_seq

    @property
    def last_timestamp_ns(self) -> int | None:
        return self._last_timestamp_ns

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def format_revision(self) -> int:
        return self._format_revision

    @property
    def ready(self) -> bool:
        return self._ready_timeline() is not None

    def stream_lengths(self) -> dict[str, int]:
        return {name: len(buffer) for name, buffer in self._buffers.items()}

    def stream_capacities(self) -> dict[str, int]:
        return {
            name: int(buffer.maxlen or 0)
            for name, buffer in self._buffers.items()
        }

    def stream_required_lengths(self) -> dict[str, int]:
        return dict(self._stream_required_lengths)

    def configure(self, metadata_format: Mapping[str, Any]) -> dict[str, int]:
        validated = validate_metadata_format(metadata_format)
        required_lengths = required_stream_lengths(validated)
        capacities = self.stream_capacities()
        for name, required_length in required_lengths.items():
            capacity = capacities[name]
            if required_length > capacity:
                raise SharpaV3ProtocolError(
                    f"metadata_format stream {name} requires "
                    f"{required_length} buffered frames, workstation "
                    f"capacity is {capacity}"
                )
        self._metadata_format = deepcopy(validated)
        self._stream_required_lengths = required_lengths
        self._format_revision += 1
        return dict(required_lengths)

    def clear(self) -> None:
        for buffer in self._buffers.values():
            buffer.clear()
        self._last_obs_seq = None
        self._last_timestamp_ns = None
        self._generation += 1

    def append(self, frame: SharpaV3Frame) -> None:
        obs_seq = int(frame.obs_seq)
        timestamp_ns = int(frame.timestamp_ns)
        samples: dict[str, Any] = {
            "ego_cam": _CameraHistoryFrame(
                obs_seq,
                timestamp_ns,
                frame.image_jpeg,
                frame.image_valid,
            ),
            "left_wrist_cam": _CameraHistoryFrame(
                obs_seq,
                timestamp_ns,
                b"",
                False,
            ),
            "right_wrist_cam": _CameraHistoryFrame(
                obs_seq,
                timestamp_ns,
                b"",
                False,
            ),
            "state": _StateHistoryFrame(
                obs_seq,
                timestamp_ns,
                frame.left_eef,
                frame.right_eef,
                frame.left_hand,
                frame.right_hand,
                frame.state_valid,
            ),
            "tau": _SensorHistoryFrame(
                obs_seq,
                timestamp_ns,
                frame.left_tau,
                frame.right_tau,
                frame.left_tau_valid,
                frame.right_tau_valid,
            ),
            "wrench": _SensorHistoryFrame(
                obs_seq,
                timestamp_ns,
                frame.left_wrench,
                frame.right_wrench,
                frame.left_wrench_valid,
                frame.right_wrench_valid,
            ),
            "deformation": _SensorHistoryFrame(
                obs_seq,
                timestamp_ns,
                frame.left_deformation,
                frame.right_deformation,
                frame.left_deformation_valid,
                frame.right_deformation_valid,
            ),
        }
        for name, sample in samples.items():
            self._buffers[name].append(sample)
        self._last_obs_seq = obs_seq
        self._last_timestamp_ns = timestamp_ns

    def snapshot(self) -> SharpaV3HistorySnapshot | None:
        timeline = self._ready_timeline()
        if timeline is None or self._metadata_format is None:
            return None
        capacities = self.stream_capacities()
        required_lengths = self.stream_required_lengths()
        frames = _join_history_streams(
            self._buffers,
            timeline,
            required_lengths,
        )
        return SharpaV3HistorySnapshot(
            frames=frames,
            metadata_format=deepcopy(self._metadata_format),
            generation=self._generation,
            format_revision=self._format_revision,
            anchor_obs_seq=int(self._last_obs_seq),
            anchor_timestamp_ns=int(self._last_timestamp_ns),
            stream_lengths=self.stream_lengths(),
            stream_capacities=capacities,
            stream_required_lengths=required_lengths,
        )

    def is_current(self, snapshot: SharpaV3HistorySnapshot | None) -> bool:
        return bool(
            snapshot is not None
            and self.ready
            and snapshot.generation == self._generation
            and snapshot.format_revision == self._format_revision
            and snapshot.anchor_obs_seq == self._last_obs_seq
            and snapshot.anchor_timestamp_ns == self._last_timestamp_ns
        )

    def _ready_timeline(self) -> tuple[tuple[int, int], ...] | None:
        if (
            self._metadata_format is None
            or self._last_obs_seq is None
            or self._last_timestamp_ns is None
        ):
            return None
        required_lengths = self.stream_required_lengths()
        for name, required_length in required_lengths.items():
            buffer = self._buffers[name]
            if required_length and (
                len(buffer) < required_length
                or buffer[-1].obs_seq != self._last_obs_seq
                or buffer[-1].timestamp_ns != self._last_timestamp_ns
            ):
                return None

        maximum = max(required_lengths.values(), default=0)
        if maximum:
            reference_name = next(
                name
                for name, required_length in required_lengths.items()
                if required_length == maximum
            )
            timeline = tuple(
                (sample.obs_seq, sample.timestamp_ns)
                for sample in tuple(self._buffers[reference_name])[-maximum:]
            )
        else:
            timeline = ((self._last_obs_seq, self._last_timestamp_ns),)

        for name, required_length in required_lengths.items():
            if not required_length:
                continue
            expected = timeline[-required_length:]
            actual = tuple(
                (sample.obs_seq, sample.timestamp_ns)
                for sample in tuple(self._buffers[name])[-required_length:]
            )
            if actual != expected:
                return None
        return timeline


def validate_server_metadata(
    value: Any,
    *,
    configured_prompt: str = "",
    expected_policy_family: str | None = None,
) -> dict[str, Any]:
    metadata = _mapping(value, "server metadata")
    if metadata.get("schema") != SERVER_SCHEMA:
        raise SharpaV3ProtocolError(
            f"server metadata schema must be {SERVER_SCHEMA}"
        )
    expected_strings = {
        "transport": "websocket+binary_msgpack",
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "infer_path": "/infer",
        "metadata_path": "/metadata",
        "reset_path": "/reset",
    }
    for field, expected in expected_strings.items():
        if metadata.get(field) != expected:
            raise SharpaV3ProtocolError(
                f"server metadata {field} must be {expected!r}"
            )
    policy_family = _nonempty_string(
        metadata.get("policy_family"), "server metadata policy_family"
    )
    if (
        expected_policy_family is not None
        and policy_family.lower() != expected_policy_family.strip().lower()
    ):
        raise SharpaV3ProtocolError(
            "server policy_family does not match configured provider: "
            f"{policy_family!r} != {expected_policy_family!r}"
        )
    server_prompt = metadata.get("prompt")
    if not isinstance(server_prompt, str):
        raise SharpaV3ProtocolError("server metadata prompt must be a string")
    client_prompt = str(configured_prompt)
    if server_prompt and client_prompt and server_prompt != client_prompt:
        raise SharpaV3ProtocolError(
            "configured prompt disagrees with the server task prompt"
        )
    metadata_format = validate_metadata_format(metadata.get("metadata_format"))
    result = dict(metadata)
    result["metadata_format"] = metadata_format
    return result


def validate_metadata_format(value: Any) -> dict[str, Any]:
    metadata_format = _mapping(value, "metadata_format")
    if metadata_format.get("schema") != METADATA_FORMAT_SCHEMA:
        raise SharpaV3ProtocolError(
            f"metadata_format.schema must be {METADATA_FORMAT_SCHEMA}"
        )
    _nonempty_string(metadata_format.get("format_id"), "metadata_format.format_id")
    image = _mapping(metadata_format.get("image"), "metadata_format.image")
    for camera in ("ego_cam", "left_wrist_cam", "right_wrist_cam"):
        _temporal_requirement(image.get(camera), f"metadata_format.image.{camera}")
    state = _mapping(metadata_format.get("state"), "metadata_format.state")
    _temporal_requirement(state, "metadata_format.state")
    for wrist_name in ("left_wrist", "right_wrist"):
        wrist = _mapping(state.get(wrist_name), f"metadata_format.state.{wrist_name}")
        _boolean(wrist.get("joint"), f"metadata_format.state.{wrist_name}.joint")
        _boolean(wrist.get("eef"), f"metadata_format.state.{wrist_name}.eef")
    hand = _mapping(state.get("hand_joint"), "metadata_format.state.hand_joint")
    _boolean(hand.get("left"), "metadata_format.state.hand_joint.left")
    _boolean(hand.get("right"), "metadata_format.state.hand_joint.right")
    sensor = _mapping(metadata_format.get("sensor"), "metadata_format.sensor")
    for sensor_name in ("tau", "wrench", "deformation"):
        _temporal_requirement(
            sensor.get(sensor_name),
            f"metadata_format.sensor.{sensor_name}",
        )
    return dict(metadata_format)


def required_frame_count(metadata_format: Mapping[str, Any]) -> int:
    validated = validate_metadata_format(metadata_format)
    lengths: list[int] = []
    lengths.extend(
        int(validated["image"][camera]["history_len"])
        for camera in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
    )
    lengths.append(int(validated["state"]["history_len"]))
    lengths.extend(
        int(validated["sensor"][name]["history_len"])
        for name in ("tau", "wrench", "deformation")
    )
    return max(lengths, default=0) + 1


def required_stream_lengths(
    metadata_format: Mapping[str, Any],
) -> dict[str, int]:
    validated = validate_metadata_format(metadata_format)

    def required_length(requirement: Mapping[str, Any]) -> int:
        history_len = int(requirement["history_len"])
        if history_len:
            return history_len + 1
        return int(bool(requirement["current"]))

    return {
        "ego_cam": required_length(validated["image"]["ego_cam"]),
        "left_wrist_cam": required_length(
            validated["image"]["left_wrist_cam"]
        ),
        "right_wrist_cam": required_length(
            validated["image"]["right_wrist_cam"]
        ),
        "state": required_length(validated["state"]),
        "tau": required_length(validated["sensor"]["tau"]),
        "wrench": required_length(validated["sensor"]["wrench"]),
        "deformation": required_length(validated["sensor"]["deformation"]),
    }


def _join_history_streams(
    buffers: Mapping[str, deque[Any]],
    timeline: Sequence[tuple[int, int]],
    required_lengths: Mapping[str, int],
) -> tuple[SharpaV3Frame, ...]:
    indexed = {
        name: {
            (int(sample.obs_seq), int(sample.timestamp_ns)): sample
            for sample in (
                tuple(buffer)[-required_lengths[name]:]
                if required_lengths[name]
                else ()
            )
        }
        for name, buffer in buffers.items()
    }

    zero_eef = np.zeros(9, dtype=np.float32)
    zero_hand = np.zeros(HAND_JOINTS_PER_SIDE, dtype=np.float32)
    zero_tau = np.zeros(HAND_JOINTS_PER_SIDE, dtype=np.float32)
    zero_tau_valid = np.zeros(HAND_JOINTS_PER_SIDE, dtype=bool)
    zero_wrench = np.zeros((TACTILE_FINGERS_PER_SIDE, 6), dtype=np.float32)
    zero_wrench_valid = np.zeros(TACTILE_FINGERS_PER_SIDE, dtype=bool)
    zero_deformation = np.zeros(
        (TACTILE_FINGERS_PER_SIDE, DEFORMATION_SIZE, DEFORMATION_SIZE),
        dtype=np.uint8,
    )
    zero_deformation_valid = np.zeros(TACTILE_FINGERS_PER_SIDE, dtype=bool)

    frames: list[SharpaV3Frame] = []
    for obs_seq, timestamp_ns in timeline:
        key = (int(obs_seq), int(timestamp_ns))
        camera = indexed["ego_cam"].get(key)
        state = indexed["state"].get(key)
        tau = indexed["tau"].get(key)
        wrench = indexed["wrench"].get(key)
        deformation = indexed["deformation"].get(key)
        frames.append(
            SharpaV3Frame(
                obs_seq=key[0],
                timestamp_ns=key[1],
                image_jpeg=camera.jpeg if camera is not None else b"",
                image_valid=bool(camera.valid) if camera is not None else False,
                left_eef=state.left_eef if state is not None else zero_eef,
                right_eef=state.right_eef if state is not None else zero_eef,
                left_hand=state.left_hand if state is not None else zero_hand,
                right_hand=state.right_hand if state is not None else zero_hand,
                state_valid=bool(state.valid) if state is not None else False,
                left_tau=tau.left if tau is not None else zero_tau,
                right_tau=tau.right if tau is not None else zero_tau,
                left_tau_valid=(
                    tau.left_valid if tau is not None else zero_tau_valid
                ),
                right_tau_valid=(
                    tau.right_valid if tau is not None else zero_tau_valid
                ),
                left_wrench=wrench.left if wrench is not None else zero_wrench,
                right_wrench=wrench.right if wrench is not None else zero_wrench,
                left_wrench_valid=(
                    wrench.left_valid
                    if wrench is not None
                    else zero_wrench_valid
                ),
                right_wrench_valid=(
                    wrench.right_valid
                    if wrench is not None
                    else zero_wrench_valid
                ),
                left_deformation=(
                    deformation.left
                    if deformation is not None
                    else zero_deformation
                ),
                right_deformation=(
                    deformation.right
                    if deformation is not None
                    else zero_deformation
                ),
                left_deformation_valid=(
                    deformation.left_valid
                    if deformation is not None
                    else zero_deformation_valid
                ),
                right_deformation_valid=(
                    deformation.right_valid
                    if deformation is not None
                    else zero_deformation_valid
                ),
            )
        )
    return tuple(frames)


def extract_frame(
    observation: Any,
    *,
    image_jpeg: bytes,
    tactile_data: bytes,
    obs_seq: int,
    timestamp_ns: int,
    image_valid: bool,
    joint_max_age_ms: float | None = None,
    wrench_max_age_ms: float | None = None,
    deformation_max_age_ms: float | None = None,
) -> SharpaV3Frame:
    wrapper = _mapping(observation, "PolicyObs payload")
    timestamp = max(0, int(timestamp_ns))
    hand_pose = _hand_pose_62d(wrapper)
    state_valid = hand_pose is not None
    if hand_pose is None:
        hand_pose = np.zeros(62, dtype=np.float32)

    left_eef = hand_pose[:9].copy()
    right_eef = hand_pose[9:18].copy()
    left_hand = hand_pose[18:40].copy()
    right_hand = hand_pose[40:62].copy()

    left_tau = np.zeros(HAND_JOINTS_PER_SIDE, dtype=np.float32)
    right_tau = np.zeros(HAND_JOINTS_PER_SIDE, dtype=np.float32)
    left_tau_valid = np.zeros(HAND_JOINTS_PER_SIDE, dtype=bool)
    right_tau_valid = np.zeros(HAND_JOINTS_PER_SIDE, dtype=bool)
    left_wrench = np.zeros((TACTILE_FINGERS_PER_SIDE, 6), dtype=np.float32)
    right_wrench = np.zeros((TACTILE_FINGERS_PER_SIDE, 6), dtype=np.float32)
    left_wrench_valid = np.zeros(TACTILE_FINGERS_PER_SIDE, dtype=bool)
    right_wrench_valid = np.zeros(TACTILE_FINGERS_PER_SIDE, dtype=bool)

    robot_state = wrapper.get("robot_state")
    state_payload = None
    transport_age = None
    if isinstance(robot_state, Mapping):
        state_payload = _unwrap_json(robot_state.get("payload"))
        transport_age = robot_state.get("age_ms")
    state_valid = bool(
        state_valid
        and isinstance(robot_state, Mapping)
        and robot_state.get("valid") is True
        and _age_is_fresh(transport_age, joint_max_age_ms)
    )
    sharpa = state_payload.get("sharpa") if isinstance(state_payload, Mapping) else None
    if isinstance(sharpa, Mapping):
        tau, tau_valid = _ordered_joint_pair(
            sharpa.get("tau"),
            sharpa.get("tau_valid"),
            sharpa.get("joint_order", sharpa.get("name")),
        )
        if not _combined_age_is_fresh(
            sharpa.get("age_ms"), transport_age, joint_max_age_ms
        ):
            tau_valid[:] = False
        tau[~tau_valid] = 0.0
        left_tau = tau[:HAND_JOINTS_PER_SIDE].copy()
        right_tau = tau[HAND_JOINTS_PER_SIDE:].copy()
        left_tau_valid = tau_valid[:HAND_JOINTS_PER_SIDE].copy()
        right_tau_valid = tau_valid[HAND_JOINTS_PER_SIDE:].copy()

    tactile = state_payload.get("tactile") if isinstance(state_payload, Mapping) else None
    if isinstance(tactile, Mapping):
        wrench, wrench_valid = _ordered_wrench(tactile)
        if not _combined_age_is_fresh(
            tactile.get("force_age_ms"), transport_age, wrench_max_age_ms
        ):
            wrench_valid[:] = False
        wrench[~wrench_valid] = 0.0
        left_wrench = wrench[:TACTILE_FINGERS_PER_SIDE].copy()
        right_wrench = wrench[TACTILE_FINGERS_PER_SIDE:].copy()
        left_wrench_valid = wrench_valid[:TACTILE_FINGERS_PER_SIDE].copy()
        right_wrench_valid = wrench_valid[TACTILE_FINGERS_PER_SIDE:].copy()

    deformation, deformation_valid = _deformation(
        wrapper,
        tactile_data,
        max_age_ms=deformation_max_age_ms,
    )
    return SharpaV3Frame(
        obs_seq=int(obs_seq),
        timestamp_ns=timestamp,
        image_jpeg=bytes(image_jpeg) if image_valid else b"",
        image_valid=bool(image_valid and image_jpeg),
        left_eef=left_eef,
        right_eef=right_eef,
        left_hand=left_hand,
        right_hand=right_hand,
        state_valid=bool(state_valid),
        left_tau=left_tau,
        right_tau=right_tau,
        left_tau_valid=left_tau_valid,
        right_tau_valid=right_tau_valid,
        left_wrench=left_wrench,
        right_wrench=right_wrench,
        left_wrench_valid=left_wrench_valid,
        right_wrench_valid=right_wrench_valid,
        left_deformation=deformation[:TACTILE_FINGERS_PER_SIDE].copy(),
        right_deformation=deformation[TACTILE_FINGERS_PER_SIDE:].copy(),
        left_deformation_valid=deformation_valid[:TACTILE_FINGERS_PER_SIDE].copy(),
        right_deformation_valid=deformation_valid[TACTILE_FINGERS_PER_SIDE:].copy(),
    )


def build_observation(
    frames: Sequence[SharpaV3Frame],
    *,
    metadata_format: Mapping[str, Any],
    session_id: str,
    request_id: int,
    prompt: str,
    execution_feedback: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_format = validate_metadata_format(metadata_format)
    frame_window = tuple(frames)
    needed = required_frame_count(active_format)
    if len(frame_window) < needed:
        raise SharpaV3ProtocolError(
            f"v3 history has {len(frame_window)} real frames, requires {needed}"
        )
    frame_window = frame_window[-needed:]
    _validate_frame_order(frame_window)
    current = frame_window[-1]
    image_format = active_format["image"]
    state_format = active_format["state"]
    sensor_format = active_format["sensor"]

    if image_format["ego_cam"]["current"] and not current.image_valid:
        raise SharpaV3ProtocolError("current ego camera is required but invalid")
    if state_format["current"] and not current.state_valid:
        raise SharpaV3ProtocolError("current 62D FK state is required but invalid")

    effective_feedback = _execution_feedback(execution_feedback)
    result = {
        "schema": OBSERVATION_SCHEMA,
        "metadata_format_id": active_format["format_id"],
        "session_id": _nonempty_string(session_id, "session_id"),
        "request_id": _integer(request_id, "request_id", minimum=0),
        "timestamp_ns": int(current.timestamp_ns),
        "prompt": str(prompt),
        "image": {
            "ego_cam": _camera_stream(
                frame_window, image_format["ego_cam"], source="ego"
            ),
            "left_wrist_cam": _camera_stream(
                frame_window, image_format["left_wrist_cam"], source="missing"
            ),
            "right_wrist_cam": _camera_stream(
                frame_window, image_format["right_wrist_cam"], source="missing"
            ),
        },
        "state": _state_observation(frame_window, state_format),
        "sensor": {
            "tau": _sensor_observation(
                frame_window, sensor_format["tau"], sensor="tau"
            ),
            "wrench": _sensor_observation(
                frame_window, sensor_format["wrench"], sensor="wrench"
            ),
            "deformation": _sensor_observation(
                frame_window,
                sensor_format["deformation"],
                sensor="deformation",
            ),
        },
        "execution_feedback": effective_feedback,
    }
    info = {
        "schema": "ws.sharpa_v3_request_info.v1",
        "metadata_format_id": active_format["format_id"],
        "history_obs_seqs": [frame.obs_seq for frame in frame_window],
        "history_frame_count": len(frame_window),
        "execution_feedback": effective_feedback,
        "valid": {
            "image": current.image_valid,
            "state": current.state_valid,
            "tau": int(
                current.left_tau_valid.sum() + current.right_tau_valid.sum()
            ),
            "wrench": int(
                current.left_wrench_valid.sum()
                + current.right_wrench_valid.sum()
            ),
            "deformation": int(
                current.left_deformation_valid.sum()
                + current.right_deformation_valid.sum()
            ),
        },
    }
    return result, info


def action_to_policy_payload(
    value: Any,
    *,
    expected_session_id: str,
    expected_request_id: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    action_result = _mapping(value, "policy response")
    if action_result.get("schema") == ERROR_SCHEMA:
        request_id = action_result.get("request_id")
        if request_id is not None:
            request_id = _integer(request_id, "error.request_id", minimum=0)
            if request_id != int(expected_request_id):
                raise SharpaV3ProtocolError(
                    "error request_id does not match request"
                )
        error = _mapping(action_result.get("error"), "error.error")
        code = _nonempty_string(error.get("code"), "error.error.code")
        message = _nonempty_string(error.get("message"), "error.error.message")
        retryable = _boolean(error.get("retryable"), "error.error.retryable")
        raise SharpaV3ServerError(
            code,
            message,
            request_id=request_id,
            retryable=retryable,
        )
    if action_result.get("schema") != ACTION_SCHEMA:
        raise SharpaV3ProtocolError(f"response schema must be {ACTION_SCHEMA}")
    if action_result.get("session_id") != expected_session_id:
        raise SharpaV3ProtocolError("response session_id does not match request")
    response_request_id = _integer(
        action_result.get("request_id"), "response.request_id", minimum=0
    )
    if response_request_id != int(expected_request_id):
        raise SharpaV3ProtocolError("response request_id does not match request")
    action_id = _nonempty_string(action_result.get("action_id"), "response.action_id")
    revision = _integer(action_result.get("revision"), "response.revision", minimum=0)
    _integer(action_result.get("timestamp_ns"), "response.timestamp_ns", minimum=0)

    execution = _mapping(action_result.get("execution"), "response.execution")
    frequency_hz = _positive_float(
        execution.get("frequency_hz"), "response.execution.frequency_hz"
    )
    action_length = _integer(
        execution.get("action_length"),
        "response.execution.action_length",
        minimum=1,
    )
    execute_start = _integer(
        execution.get("execute_start"),
        "response.execution.execute_start",
        minimum=0,
    )
    execute_length = _integer(
        execution.get("execute_length"),
        "response.execution.execute_length",
        minimum=1,
    )
    execute_stop = execute_start + execute_length
    if execute_stop > action_length:
        raise SharpaV3ProtocolError("response execution slice exceeds action_length")

    action = _required_mapping(
        action_result.get("action"),
        "response.action",
        ("left_wrist", "right_wrist", "hand_joint"),
    )

    def absolute_eef(side: str) -> np.ndarray:
        wrist = _required_mapping(
            action[side],
            f"response.action.{side}",
            ("joint", "eef", "eef_def"),
        )
        eef_def = "absolute" if wrist["eef_def"] is None else wrist["eef_def"]
        if eef_def != "absolute":
            raise SharpaV3ProtocolError(
                f"response.action.{side}.eef_def must be absolute"
            )
        return _array(
            wrist["eef"],
            f"response.action.{side}.eef",
            dtype=np.float32,
            shape=(action_length, 9),
        )

    left_wrist = absolute_eef("left_wrist")
    right_wrist = absolute_eef("right_wrist")
    hands = _required_mapping(
        action["hand_joint"],
        "response.action.hand_joint",
        ("left", "right"),
    )
    left_hand = _array(
        hands["left"],
        "response.action.hand_joint.left",
        dtype=np.float32,
        shape=(action_length, 22),
    )
    right_hand = _array(
        hands["right"],
        "response.action.hand_joint.right",
        dtype=np.float32,
        shape=(action_length, 22),
    )
    hand_joint = np.concatenate((left_hand, right_hand), axis=1)

    auxiliary = _required_mapping(
        action_result.get("auxiliary"),
        "response.auxiliary",
        ("video", "tactile"),
    )
    auxiliary_video = _required_mapping(
        auxiliary["video"],
        "response.auxiliary.video",
        ("ego", "left_wrist", "right_wrist"),
    )
    _required_mapping(
        auxiliary["tactile"],
        "response.auxiliary.tactile",
        ("deformation", "wrench", "hand_tau"),
    )
    diagnostics = _mapping(action_result.get("diagnostics"), "response.diagnostics")
    for field in ("policy_family", "checkpoint_id", "checkpoint_path"):
        _nonempty_string(diagnostics.get(field), f"response.diagnostics.{field}")
    latency_ms = diagnostics.get("inference_latency_ms")
    if _nonnegative_float(latency_ms, "response.diagnostics.inference_latency_ms") < 0:
        raise AssertionError("unreachable")

    debug_value = action_result.get("debug")
    debug = dict(debug_value) if isinstance(debug_value, Mapping) else {}
    normalized_diagnostics = dict(diagnostics)
    video_path = ""
    for candidate in (
        auxiliary_video.get("ego"),
        action_result.get("server_video_pred_path"),
        action_result.get("video_path"),
        action_result.get("video"),
        action.get("video_path"),
        action.get("video"),
        debug.get("server_video_pred_path"),
        debug.get("video_path"),
        debug.get("video"),
        normalized_diagnostics.get("server_video_pred_path"),
        normalized_diagnostics.get("video_path"),
        normalized_diagnostics.get("video"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            video_path = candidate.strip()
            break
    if video_path:
        debug["server_video_pred_path"] = video_path
        normalized_diagnostics["server_video_pred_path"] = video_path

    next_format_raw = action_result.get("next_metadata_format")
    next_format = (
        validate_metadata_format(next_format_raw)
        if next_format_raw is not None
        else None
    )
    action_62d = np.concatenate(
        (left_wrist, right_wrist, hand_joint), axis=1
    )[execute_start:execute_stop].copy()
    server_execution = {
        "action_id": action_id,
        "revision": revision,
        "frequency_hz": frequency_hz,
        "action_length": action_length,
        "execute_start": execute_start,
        "execute_length": execute_length,
        "server_driven_execution": True,
    }
    payload = {
        "schema": ACTION_SCHEMA,
        "session_id": expected_session_id,
        "request_id": response_request_id,
        "action_id": action_id,
        "revision": revision,
        "timestamp_ns": int(action_result["timestamp_ns"]),
        "action_horizon": execute_length,
        "action_hz": frequency_hz,
        "action_hand_pose_62d": action_62d,
        "action_space": "sharpa_dexretarget_position_62d",
        "eef_def": "absolute",
        "layout": "left_wrist9,right_wrist9,sharpa_q44",
        "diagnostics": normalized_diagnostics,
        "auxiliary": dict(auxiliary),
        "_ws_sharpa_v4": server_execution,
    }
    if debug:
        payload["debug"] = debug
    if video_path:
        payload["server_video_pred_path"] = video_path
        payload["video"] = video_path
    return payload, next_format


def _state_observation(
    frames: tuple[SharpaV3Frame, ...], requirement: Mapping[str, Any]
) -> dict[str, Any]:
    history_len = int(requirement["history_len"])
    history_frames = _history_frames(frames, history_len)
    history = None
    if history_len:
        history = {
            "timestamp_ns": np.asarray(
                [frame.timestamp_ns for frame in history_frames], dtype=np.int64
            ),
            "left_wrist": _wrist_state(
                history_frames,
                requirement["left_wrist"],
                side="left",
                history=True,
            ),
            "right_wrist": _wrist_state(
                history_frames,
                requirement["right_wrist"],
                side="right",
                history=True,
            ),
            "hand_joint": _hand_state(
                history_frames,
                requirement["hand_joint"],
                history=True,
            ),
            "valid": np.asarray(
                [frame.state_valid for frame in history_frames], dtype=bool
            ),
        }
    current = None
    if requirement["current"]:
        frame = frames[-1]
        current = {
            "timestamp_ns": int(frame.timestamp_ns),
            "left_wrist": _wrist_state(
                (frame,), requirement["left_wrist"], side="left", history=False
            ),
            "right_wrist": _wrist_state(
                (frame,), requirement["right_wrist"], side="right", history=False
            ),
            "hand_joint": _hand_state(
                (frame,), requirement["hand_joint"], history=False
            ),
            "valid": bool(frame.state_valid),
        }
    return {"history": history, "current": current}


def _wrist_state(
    frames: tuple[SharpaV3Frame, ...],
    requirement: Mapping[str, Any],
    *,
    side: str,
    history: bool,
) -> dict[str, Any]:
    joint = None
    if requirement["joint"]:
        raise SharpaV3ProtocolError(
            "workstation has no arm wrist-joint vector for v3 state"
        )
    eef = None
    eef_def = None
    if requirement["eef"]:
        values = [getattr(frame, f"{side}_eef") for frame in frames]
        eef = np.stack(values).astype(np.float32, copy=False) if history else values[0]
        eef_def = "absolute"
    return {"joint": joint, "eef": eef, "eef_def": eef_def}


def _hand_state(
    frames: tuple[SharpaV3Frame, ...],
    requirement: Mapping[str, Any],
    *,
    history: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        if requirement[side]:
            values = [getattr(frame, f"{side}_hand") for frame in frames]
            result[side] = (
                np.stack(values).astype(np.float32, copy=False)
                if history
                else values[0]
            )
        else:
            result[side] = None
    return result


def _sensor_observation(
    frames: tuple[SharpaV3Frame, ...],
    requirement: Mapping[str, Any],
    *,
    sensor: str,
) -> dict[str, Any]:
    history_len = int(requirement["history_len"])
    history_frames = _history_frames(frames, history_len)
    history = None
    if history_len:
        history = {
            "left": np.stack(
                [getattr(frame, f"left_{sensor}") for frame in history_frames]
            ),
            "right": np.stack(
                [getattr(frame, f"right_{sensor}") for frame in history_frames]
            ),
            "timestamp_ns": np.asarray(
                [frame.timestamp_ns for frame in history_frames], dtype=np.int64
            ),
            "valid": {
                "left": np.stack(
                    [
                        getattr(frame, f"left_{sensor}_valid")
                        for frame in history_frames
                    ]
                ),
                "right": np.stack(
                    [
                        getattr(frame, f"right_{sensor}_valid")
                        for frame in history_frames
                    ]
                ),
            },
        }
    current = None
    if requirement["current"]:
        frame = frames[-1]
        current = {
            "left": getattr(frame, f"left_{sensor}"),
            "right": getattr(frame, f"right_{sensor}"),
            "timestamp_ns": int(frame.timestamp_ns),
            "valid": {
                "left": getattr(frame, f"left_{sensor}_valid"),
                "right": getattr(frame, f"right_{sensor}_valid"),
            },
        }
    return {"history": history, "current": current}


def _camera_stream(
    frames: tuple[SharpaV3Frame, ...],
    requirement: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    history_frames = _history_frames(frames, int(requirement["history_len"]))
    history = [_camera_frame(frame, source=source) for frame in history_frames]
    current = (
        _camera_frame(frames[-1], source=source)
        if requirement["current"]
        else None
    )
    return {"history": history, "current": current}


def _camera_frame(frame: SharpaV3Frame, *, source: str) -> dict[str, Any]:
    valid = bool(source == "ego" and frame.image_valid)
    return {
        "encoding": "jpeg",
        "data": frame.image_jpeg if valid else b"",
        "timestamp_ns": int(frame.timestamp_ns),
        "valid": valid,
    }


def _history_frames(
    frames: tuple[SharpaV3Frame, ...], history_len: int
) -> tuple[SharpaV3Frame, ...]:
    if history_len <= 0:
        return ()
    return frames[-(history_len + 1):-1]


def _execution_feedback(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {"last_action_id": None, "executed_steps": 0, "success": True}
    feedback = _mapping(value, "execution_feedback")
    action_id = feedback.get("last_action_id")
    if action_id is not None:
        action_id = _nonempty_string(action_id, "execution_feedback.last_action_id")
    return {
        "last_action_id": action_id,
        "executed_steps": _integer(
            feedback.get("executed_steps"),
            "execution_feedback.executed_steps",
            minimum=0,
        ),
        "success": _boolean(
            feedback.get("success"), "execution_feedback.success"
        ),
    }


def _hand_pose_62d(wrapper: Mapping[str, Any]) -> np.ndarray | None:
    candidates: list[Any] = [
        wrapper.get("observation/hand_pose_62d"),
        wrapper.get("hand_pose_62d"),
    ]
    for field in ("policy_input", "dreamzero", "converted_state"):
        value = wrapper.get(field)
        if not isinstance(value, Mapping):
            continue
        if field == "policy_input" and value.get("valid") is False:
            continue
        candidates.extend(
            (value.get("hand_pose_62d"), value.get("observation/hand_pose_62d"))
        )
    for value in candidates:
        if value is None:
            continue
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if array.shape == (1, 62):
            array = array[0]
        if array.shape == (62,) and np.all(np.isfinite(array)):
            return array.copy()
    return None


def _ordered_joint_pair(
    values: Any, validity: Any, order: Any
) -> tuple[np.ndarray, np.ndarray]:
    try:
        array = np.asarray(values, dtype=np.float32)
        valid = np.asarray(validity)
    except (TypeError, ValueError):
        return np.zeros(44, dtype=np.float32), np.zeros(44, dtype=bool)
    if array.shape != (44,) or valid.shape != (44,) or valid.dtype.kind != "b":
        return np.zeros(44, dtype=np.float32), np.zeros(44, dtype=bool)
    if not np.all(np.isfinite(array)):
        valid = valid.astype(bool, copy=True)
        valid[~np.isfinite(array)] = False
        array = np.nan_to_num(array, copy=True)
    if not isinstance(order, (list, tuple)) or len(order) != 44:
        return np.zeros(44, dtype=np.float32), np.zeros(44, dtype=bool)
    source = {str(name): index for index, name in enumerate(order)}
    if set(source) != set(HAND_JOINT_ORDER):
        return np.zeros(44, dtype=np.float32), np.zeros(44, dtype=bool)
    indices = [source[name] for name in HAND_JOINT_ORDER]
    array = array[indices]
    valid = valid[indices]
    return array.astype(np.float32, copy=True), valid.astype(bool, copy=True)


def _ordered_wrench(tactile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = tactile.get("wrench")
    if raw is None:
        try:
            force = np.asarray(tactile.get("force"), dtype=np.float32)
            torque = np.asarray(tactile.get("torque"), dtype=np.float32)
            raw = np.concatenate((force, torque), axis=1)
        except (TypeError, ValueError):
            raw = None
    try:
        array = np.asarray(raw, dtype=np.float32)
        valid = np.asarray(
            tactile.get("wrench_valid", tactile.get("force_valid"))
        )
    except (TypeError, ValueError):
        return np.zeros((10, 6), dtype=np.float32), np.zeros(10, dtype=bool)
    order = tactile.get("order")
    if (
        array.shape != (10, 6)
        or valid.shape != (10,)
        or valid.dtype.kind != "b"
        or not isinstance(order, (list, tuple))
        or len(order) != 10
    ):
        return np.zeros((10, 6), dtype=np.float32), np.zeros(10, dtype=bool)
    source = {str(name): index for index, name in enumerate(order)}
    if set(source) != set(TACTILE_ORDER):
        return np.zeros((10, 6), dtype=np.float32), np.zeros(10, dtype=bool)
    indices = [source[name] for name in TACTILE_ORDER]
    array = array[indices]
    valid = valid[indices].astype(bool, copy=True)
    finite = np.all(np.isfinite(array), axis=1)
    valid &= finite
    array = np.nan_to_num(array, copy=True).astype(np.float32, copy=False)
    return array, valid


def _deformation(
    wrapper: Mapping[str, Any],
    tactile_data: bytes,
    *,
    max_age_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((10, DEFORMATION_SIZE, DEFORMATION_SIZE), dtype=np.uint8)
    valid = np.zeros(10, dtype=bool)
    robot_tactile = wrapper.get("robot_tactile")
    if not isinstance(robot_tactile, Mapping):
        return output, valid
    if not _age_is_fresh(robot_tactile.get("age_ms"), max_age_ms):
        return output, valid
    metadata = _unwrap_json(robot_tactile.get("metadata"))
    entries = metadata.get("entries") if isinstance(metadata, Mapping) else None
    if not isinstance(entries, list) or not tactile_data:
        return output, valid
    destination = {name: index for index, name in enumerate(TACTILE_ORDER)}
    raw = memoryview(tactile_data)
    for entry in entries:
        if not isinstance(entry, Mapping) or not bool(entry.get("valid")):
            continue
        index = destination.get(f"{entry.get('side', '')}_{entry.get('finger', '')}")
        if index is None:
            continue
        try:
            raw_offset = (
                entry["raw_offset"] if "raw_offset" in entry else entry["offset"]
            )
            raw_length = (
                entry["raw_length"] if "raw_length" in entry else entry["length"]
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
            raw[offset:offset + length], dtype=np.uint8
        ).reshape(height, width)
        if image.shape != (DEFORMATION_SIZE, DEFORMATION_SIZE):
            from PIL import Image

            image = np.asarray(
                Image.fromarray(image, mode="L").resize(
                    (DEFORMATION_SIZE, DEFORMATION_SIZE),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
        output[index] = image
        valid[index] = True
    return output, valid


def _validate_frame_order(frames: tuple[SharpaV3Frame, ...]) -> None:
    seqs = [frame.obs_seq for frame in frames]
    timestamps = [frame.timestamp_ns for frame in frames]
    if any(current <= previous for previous, current in zip(seqs, seqs[1:])):
        raise SharpaV3ProtocolError("v3 history obs_seq must be strictly increasing")
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        raise SharpaV3ProtocolError("v3 history timestamps must be chronological")


def _combined_age_is_fresh(
    source_age: Any, transport_age: Any, limit: float | None
) -> bool:
    if limit is None:
        return True
    try:
        source = float(source_age)
        transport = float(transport_age)
        maximum = float(limit)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(source)
        and source >= 0.0
        and np.isfinite(transport)
        and transport >= 0.0
        and np.isfinite(maximum)
        and maximum > 0.0
        and source + transport <= maximum
    )


def _age_is_fresh(value: Any, limit: float | None) -> bool:
    if limit is None:
        return True
    try:
        age = float(value)
        maximum = float(limit)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(age)
        and age >= 0.0
        and np.isfinite(maximum)
        and maximum > 0.0
        and age <= maximum
    )


def _unwrap_json(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get("valid") is True:
        return value.get("json")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SharpaV3ProtocolError(f"{label} must be an object")
    return value


def _required_mapping(
    value: Any, label: str, fields: tuple[str, ...]
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    missing = [field for field in fields if field not in result]
    if missing:
        raise SharpaV3ProtocolError(
            f"{label} missing fields: {', '.join(missing)}"
        )
    return result


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SharpaV3ProtocolError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise SharpaV3ProtocolError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise SharpaV3ProtocolError(f"{label} must be >= {minimum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise SharpaV3ProtocolError(f"{label} must be a boolean")
    return bool(value)


def _positive_float(value: Any, label: str) -> float:
    result = _nonnegative_float(value, label)
    if result <= 0.0:
        raise SharpaV3ProtocolError(f"{label} must be positive")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SharpaV3ProtocolError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise SharpaV3ProtocolError(f"{label} must be finite and nonnegative")
    return result


def _array(
    value: Any,
    label: str,
    *,
    dtype: type[np.generic],
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise SharpaV3ProtocolError(f"{label} must be a numpy.ndarray")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise SharpaV3ProtocolError(
            f"{label} dtype is {value.dtype}, expected {expected_dtype}"
        )
    if value.shape != shape:
        raise SharpaV3ProtocolError(
            f"{label} shape is {value.shape}, expected {shape}"
        )
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise SharpaV3ProtocolError(f"{label} contains NaN or Inf")
    return value


def _temporal_requirement(value: Any, label: str) -> None:
    requirement = _mapping(value, label)
    _integer(requirement.get("history_len"), f"{label}.history_len", minimum=0)
    _boolean(requirement.get("current"), f"{label}.current")
