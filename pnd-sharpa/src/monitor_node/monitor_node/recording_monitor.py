#!/usr/bin/env python3
"""Record local monitor samples gated by /teleop/status_json t_record.

The on-disk format is columnar per sample: one schema file describes static
names/orders, while dynamic telemetry is written as NPZ arrays. Large tactile
deform images are streamed to a raw file during capture and zstd-compressed
after the t_record sample closes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

try:
    from teleop_interfaces.msg import (
        SharpaJointState,
        TactileContactPointsArray,
        TactileDeformImageArray,
        TactileForce6DArray,
    )

    TELEOP_INTERFACES_IMPORT_ERROR = ""
except Exception as exc:
    SharpaJointState = None
    TactileContactPointsArray = None
    TactileDeformImageArray = None
    TactileForce6DArray = None
    TELEOP_INTERFACES_IMPORT_ERROR = str(exc)


ADAM_JOINT_COUNT = 31
ADAM_UPPER_BODY_START = 12
ADAM_UPPER_BODY_COUNT = 19
ADAM_UPPER_BODY_INDICES = list(
    range(ADAM_UPPER_BODY_START, ADAM_UPPER_BODY_START + ADAM_UPPER_BODY_COUNT)
)
SHARPA_JOINT_COUNT = 44
TACTILE_DEFORM_SHAPE = (10, 240, 240)
TACTILE_DEFORM_BYTES = (
    TACTILE_DEFORM_SHAPE[0] * TACTILE_DEFORM_SHAPE[1] * TACTILE_DEFORM_SHAPE[2]
)
# Must match sharpa_node.TACTILE_AGGREGATE_ORDER and the raw tensor plane order:
# [right_pinky, right_ring, right_middle, right_index, right_thumb,
#  left_pinky, left_ring, left_middle, left_index, left_thumb].
TACTILE_PLANES = (
    ("right", "pinky"),
    ("right", "ring"),
    ("right", "middle"),
    ("right", "index"),
    ("right", "thumb"),
    ("left", "pinky"),
    ("left", "ring"),
    ("left", "middle"),
    ("left", "index"),
    ("left", "thumb"),
)
SOURCE_NAMES = (
    "status_json",
    "adam",
    "sharpa",
    "tactile_force6d",
    "tactile_contact",
    "tactile_deform",
    "zed_status",
)
SOURCE_MAX_AGE_MS = {
    "status_json": 1500.0,
    "adam": 500.0,
    "sharpa": 500.0,
    "tactile_force6d": 500.0,
    "tactile_contact": 500.0,
    "tactile_deform": 500.0,
    "zed_status": 2500.0,
}
TRUE_VALUES = {"1", "true", "yes", "on", "active", "record", "t_record"}
FALSE_VALUES = {"0", "false", "no", "off", "inactive", "stop", "stopped", ""}


@dataclass
class Latest:
    payload: dict[str, Any] | None = None
    stamp_mono: float | None = None
    count: int = 0
    valid: bool = False
    last_error: str = ""


@dataclass
class RecordingSession:
    index: int
    sample_name: str
    partial_dir: Path
    final_dir: Path
    started_unix_ns: int
    started_mono: float
    start_status: dict[str, Any]
    raw_handle: BinaryIO
    zed_video_process: Any | None = None
    zed_video_log_handle: BinaryIO | None = None
    zed_video_started: bool = False
    zed_video_error: str = ""
    timeline_rows: int = 0
    deform_frames: int = 0
    deform_dropped: int = 0
    ended_unix_ns: int | None = None
    ended_mono: float | None = None
    stop_status: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""

    time_unix_ns: list[int] = field(default_factory=list)
    elapsed_ns: list[int] = field(default_factory=list)
    source_valid: list[list[bool]] = field(default_factory=list)
    source_age_ms: list[list[float]] = field(default_factory=list)
    source_count: list[list[int]] = field(default_factory=list)

    adam_q: list[list[float]] = field(default_factory=list)
    adam_dq: list[list[float]] = field(default_factory=list)
    adam_tau: list[list[float]] = field(default_factory=list)
    adam_valid: list[bool] = field(default_factory=list)
    adam_age_ms: list[float] = field(default_factory=list)
    adam_source_count: list[int] = field(default_factory=list)
    adam_header_stamp_ns: list[int] = field(default_factory=list)

    sharpa_q: list[list[float]] = field(default_factory=list)
    sharpa_dq: list[list[float]] = field(default_factory=list)
    sharpa_tau: list[list[float]] = field(default_factory=list)
    sharpa_q_cmd: list[list[float]] = field(default_factory=list)
    sharpa_q_cmd_valid: list[bool] = field(default_factory=list)
    sharpa_valid: list[bool] = field(default_factory=list)
    sharpa_age_ms: list[float] = field(default_factory=list)
    sharpa_source_count: list[int] = field(default_factory=list)
    sharpa_header_stamp_ns: list[int] = field(default_factory=list)

    tactile_force: list[list[list[float]]] = field(default_factory=list)
    tactile_torque: list[list[list[float]]] = field(default_factory=list)
    tactile_force_frame_id: list[list[int]] = field(default_factory=list)
    tactile_force_sensor_time: list[list[float]] = field(default_factory=list)
    tactile_force_valid: list[list[bool]] = field(default_factory=list)

    tactile_contact_points_xyz: list[list[float]] = field(default_factory=list)
    tactile_contact_offset: list[list[int]] = field(default_factory=list)
    tactile_contact_count: list[list[int]] = field(default_factory=list)
    tactile_contact_frame_id: list[list[int]] = field(default_factory=list)
    tactile_contact_sensor_time: list[list[float]] = field(default_factory=list)
    tactile_contact_valid: list[list[bool]] = field(default_factory=list)

    tactile_deform_time_unix_ns: list[int] = field(default_factory=list)
    tactile_deform_elapsed_ns: list[int] = field(default_factory=list)
    tactile_deform_timeline_row: list[int] = field(default_factory=list)
    tactile_deform_raw_frame_index: list[int] = field(default_factory=list)
    tactile_deform_frame_id: list[list[int]] = field(default_factory=list)
    tactile_deform_sensor_time: list[list[float]] = field(default_factory=list)
    tactile_deform_valid: list[list[bool]] = field(default_factory=list)

    status_events: list[dict[str, Any]] = field(default_factory=list)
    quest_webvr_events: list[dict[str, Any]] = field(default_factory=list)
    quest_retarget_events: list[dict[str, Any]] = field(default_factory=list)


def _now_unix_ns() -> int:
    return time.time_ns()


def _elapsed_ns(started_mono: float, now: float) -> int:
    return int(round((now - started_mono) * 1_000_000_000))


def _age_ms(stamp: float | None, now: float) -> float | None:
    if stamp is None:
        return None
    value = (now - stamp) * 1000.0
    return round(value, 3) if math.isfinite(value) else None


def _age_ms_value(stamp: float | None, now: float) -> float:
    value = _age_ms(stamp, now)
    return math.nan if value is None else value


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _float_or_nan(value: Any) -> float:
    number = _safe_float(value)
    return math.nan if number is None else number


def _tactile_metadata_valid(frame_id: int, sensor_time: float | None) -> bool:
    return frame_id > 0 and sensor_time is not None and sensor_time > 0.0


def _fixed_float_vector(values: Any, length: int) -> tuple[list[float | None], bool]:
    output: list[float | None] = []
    valid = True
    try:
        count = len(values)
    except TypeError:
        count = 0
        valid = False
    for idx in range(length):
        value = _safe_float(values[idx]) if idx < count else None
        output.append(value)
        if value is None:
            valid = False
    return output, valid


def _nan_vector(values: Any, length: int) -> list[float]:
    output = [math.nan] * length
    try:
        count = len(values)
    except TypeError:
        return output
    for idx in range(min(length, count)):
        output[idx] = _float_or_nan(values[idx])
    return output


def _stamp_payload(stamp: Any) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def _stamp_payload_ns(stamp: dict[str, Any] | None) -> int:
    if not isinstance(stamp, dict):
        return 0
    try:
        sec = int(stamp.get("sec", 0))
        nanosec = int(stamp.get("nanosec", 0))
    except (TypeError, ValueError):
        return 0
    return sec * 1_000_000_000 + nanosec


def _header_stamp_ns(payload: dict[str, Any]) -> int:
    header = payload.get("header")
    if not isinstance(header, dict):
        return 0
    return _stamp_payload_ns(header.get("stamp"))


def _header_payload(msg: Any) -> dict[str, Any]:
    header = getattr(msg, "header", None)
    if header is None:
        return {}
    return {
        "stamp": _stamp_payload(header.stamp),
        "frame_id": str(header.frame_id),
    }


def _json_load_dict(raw: str) -> dict[str, Any]:
    data = json.loads(raw.strip() or "{}")
    if not isinstance(data, dict):
        raise ValueError("status JSON payload must be an object")
    return data


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
    return None


def _param_bool(value: Any, default: bool = False) -> bool:
    parsed = _coerce_bool(value)
    return default if parsed is None else parsed


def _json_sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def _containing_mount(path: Path) -> Path:
    """Return the deepest mounted filesystem containing path."""
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    for item in (candidate, *candidate.parents):
        if item.is_mount():
            return item
    return Path("/")


def _expected_mount_for_recording_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    parts = resolved.parts
    if len(parts) >= 3 and parts[1] == "mnt":
        return Path(*parts[:3])
    if len(parts) >= 4 and parts[1] == "media":
        return Path(*parts[:4])
    return resolved


def _array_rows(rows: list[Any], tail_shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    if rows:
        return np.asarray(rows, dtype=dtype)
    return np.empty((0, *tail_shape), dtype=dtype)


def _array_1d(rows: list[Any], dtype: Any) -> np.ndarray:
    return np.asarray(rows, dtype=dtype) if rows else np.empty((0,), dtype=dtype)


class RecordingMonitor(Node):
    """Local file recorder for columnar robot samples."""

    def __init__(self) -> None:
        super().__init__("monitor")

        self.declare_parameter("recording_root", "/mnt/t9/recordings")
        self.declare_parameter("require_recording_root_mount", True)
        self.declare_parameter("sample_rate_hz", 30.0)
        self.declare_parameter("status_json_topic", "/teleop/status_json")
        self.declare_parameter("quest_webvr_status_topic", "/quest/webvr_status")
        self.declare_parameter("quest_retarget_status_topic", "/quest/retarget_status")
        self.declare_parameter("adam_topic", "/adam_physical_joint_states")
        self.declare_parameter("sharpa_joint_topic", "/sharpa_physical_joint_states")
        self.declare_parameter("tactile_deform_topic", "/sharpa_physical_tactile/deform_images")
        self.declare_parameter("tactile_force6d_topic", "/sharpa_physical_tactile/force6d")
        self.declare_parameter("tactile_contact_topic", "/sharpa_physical_tactile/contact_points")
        self.declare_parameter("zed_status_topic", "/zed/status")
        self.declare_parameter("record_zed_video", True)
        self.declare_parameter("zed_video_rtp_port", 5600)
        self.declare_parameter("zed_video_ffmpeg", "ffmpeg")
        self.declare_parameter("zed_video_stop_timeout_s", 5.0)
        self.declare_parameter("zstd_level", 1)
        self.declare_parameter("keep_raw_deform", False)
        self.declare_parameter("require_tactile_fresh_on_start", True)
        self.declare_parameter("block_recording_on_tactile_error", False)
        self.declare_parameter("tactile_start_max_age_ms", 500.0)
        self.declare_parameter("tactile_error_log_period_s", 1.0)

        self.recording_root = Path(str(self.get_parameter("recording_root").value)).expanduser()
        self.sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self.status_json_topic = str(self.get_parameter("status_json_topic").value)
        self.quest_webvr_status_topic = str(
            self.get_parameter("quest_webvr_status_topic").value
        )
        self.quest_retarget_status_topic = str(
            self.get_parameter("quest_retarget_status_topic").value
        )
        self.adam_topic = str(self.get_parameter("adam_topic").value)
        self.sharpa_joint_topic = str(self.get_parameter("sharpa_joint_topic").value)
        self.tactile_deform_topic = str(self.get_parameter("tactile_deform_topic").value)
        self.tactile_force6d_topic = str(self.get_parameter("tactile_force6d_topic").value)
        self.tactile_contact_topic = str(self.get_parameter("tactile_contact_topic").value)
        self.zed_status_topic = str(self.get_parameter("zed_status_topic").value)
        self.record_zed_video = _param_bool(
            self.get_parameter("record_zed_video").value,
            default=True,
        )
        self.zed_video_rtp_port = int(self.get_parameter("zed_video_rtp_port").value)
        self.zed_video_ffmpeg = str(self.get_parameter("zed_video_ffmpeg").value).strip()
        self.zed_video_stop_timeout_s = float(
            self.get_parameter("zed_video_stop_timeout_s").value
        )
        self.zstd_level = int(self.get_parameter("zstd_level").value)
        self.keep_raw_deform = _param_bool(self.get_parameter("keep_raw_deform").value)
        self.require_tactile_fresh_on_start = _param_bool(
            self.get_parameter("require_tactile_fresh_on_start").value,
            default=True,
        )
        self.block_recording_on_tactile_error = _param_bool(
            self.get_parameter("block_recording_on_tactile_error").value,
            default=False,
        )
        self.tactile_start_max_age_ms = float(
            self.get_parameter("tactile_start_max_age_ms").value
        )
        self.tactile_error_log_period_s = float(
            self.get_parameter("tactile_error_log_period_s").value
        )
        self.require_recording_root_mount = _param_bool(
            self.get_parameter("require_recording_root_mount").value,
            default=True,
        )

        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.zstd_level < 1 or self.zstd_level > 19:
            raise ValueError("zstd_level must be in [1, 19]")
        if self.record_zed_video and not self.zed_video_ffmpeg:
            raise ValueError("zed_video_ffmpeg must be non-empty")
        if self.zed_video_rtp_port <= 0 or self.zed_video_rtp_port > 65535:
            raise ValueError("zed_video_rtp_port must be in [1, 65535]")
        if self.zed_video_stop_timeout_s <= 0.0:
            raise ValueError("zed_video_stop_timeout_s must be positive")
        if self.tactile_start_max_age_ms <= 0.0:
            raise ValueError("tactile_start_max_age_ms must be positive")
        if self.tactile_error_log_period_s <= 0.0:
            raise ValueError("tactile_error_log_period_s must be positive")
        self._validate_recording_root_mount()

        self._lock = threading.RLock()
        self._active: RecordingSession | None = None
        self._finalizers: list[threading.Thread] = []
        self._last_t_record = False
        self._status = Latest()
        self._quest_webvr_status = Latest()
        self._quest_retarget_status = Latest()
        self._adam = Latest()
        self._sharpa = Latest()
        self._tactile_deform = Latest()
        self._tactile_force6d = Latest()
        self._tactile_contact = Latest()
        self._zed_status = Latest()
        self._last_recording_tactile_error = ""
        self._last_recording_tactile_error_mono = 0.0

        default_qos = 10
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(String, self.status_json_topic, self._on_status_json, default_qos)
        self.create_subscription(
            String,
            self.quest_webvr_status_topic,
            self._on_quest_webvr_status,
            default_qos,
        )
        self.create_subscription(
            String,
            self.quest_retarget_status_topic,
            self._on_quest_retarget_status,
            default_qos,
        )
        self.create_subscription(JointState, self.adam_topic, self._on_adam, default_qos)
        if SharpaJointState is None:
            self.get_logger().warning(
                "teleop_interfaces.msg.SharpaJointState is unavailable; "
                "Sharpa physical joint samples will not be recorded. Build and "
                "source teleop_interfaces first."
            )
        else:
            self.create_subscription(
                SharpaJointState,
                self.sharpa_joint_topic,
                self._on_sharpa,
                default_qos,
            )
        self.create_subscription(String, self.zed_status_topic, self._on_zed_status, default_qos)
        self._create_tactile_subscriptions(sensor_qos)
        self.create_timer(1.0 / self.sample_rate_hz, self._sample_tick)

        self.get_logger().info(
            "monitor ready; root="
            f"{self.recording_root}; gate={self.status_json_topic}.t_record; "
            f"rate={self.sample_rate_hz:g} Hz; format=npz_columnar_v2"
        )

    def _validate_recording_root_mount(self) -> None:
        if not self.require_recording_root_mount:
            return
        expected_mount = _expected_mount_for_recording_root(self.recording_root)
        mount = _containing_mount(self.recording_root)
        if mount == expected_mount:
            return
        resolved = self.recording_root.expanduser().resolve(strict=False)
        raise RuntimeError(
            f"recording_root={self.recording_root} resolves to {resolved}, "
            f"but expected mount point {expected_mount} is not mounted "
            f"(containing mount is {mount}). Mount the dataset disk first or set "
            "require_recording_root_mount:=false for intentional local testing."
        )

    def _create_tactile_subscriptions(self, qos: QoSProfile) -> None:
        if TELEOP_INTERFACES_IMPORT_ERROR:
            self.get_logger().warning(
                "teleop_interfaces.msg tactile array types are unavailable; "
                "tactile deform/force/contact topics will not be recorded. "
                f"Build and source teleop_interfaces to enable them. Import error: "
                f"{TELEOP_INTERFACES_IMPORT_ERROR}"
            )
            return
        assert TactileDeformImageArray is not None
        assert TactileForce6DArray is not None
        assert TactileContactPointsArray is not None
        self.create_subscription(
            TactileDeformImageArray,
            self.tactile_deform_topic,
            self._on_tactile_deform,
            qos,
        )
        self.create_subscription(
            TactileForce6DArray,
            self.tactile_force6d_topic,
            self._on_tactile_force6d,
            qos,
        )
        self.create_subscription(
            TactileContactPointsArray,
            self.tactile_contact_topic,
            self._on_tactile_contact,
            qos,
        )

    def _on_status_json(self, msg: String) -> None:
        now = time.monotonic()
        try:
            status = _json_load_dict(msg.data)
            t_record = _coerce_bool(status.get("t_record"))
            if t_record is None:
                raise ValueError("status JSON has no boolean-like t_record field")
            payload = _json_sanitize(status)
            with self._lock:
                self._status = Latest(
                    payload=payload,
                    stamp_mono=now,
                    count=self._status.count + 1,
                    valid=True,
                )
                previous = self._last_t_record
                self._last_t_record = t_record
                active = self._active
                if active is not None and t_record and previous:
                    self._append_status_event_locked(active, payload, now, "status")
        except Exception as exc:
            with self._lock:
                self._status = Latest(
                    payload=None,
                    stamp_mono=now,
                    count=self._status.count + 1,
                    valid=False,
                    last_error=str(exc),
                )
            self.get_logger().warning(f"ignored malformed status JSON: {exc}")
            return

        if t_record and not previous:
            if not self._start_recording(payload):
                with self._lock:
                    self._last_t_record = False
        elif previous and not t_record:
            self._stop_recording(payload, "t_record_false")

    def _on_adam(self, msg: JointState) -> None:
        self._set_latest(self._adam, self._joint_payload(msg, ADAM_JOINT_COUNT))

    def _on_quest_webvr_status(self, msg: String) -> None:
        self._on_quest_status(msg, "webvr")

    def _on_quest_retarget_status(self, msg: String) -> None:
        self._on_quest_status(msg, "retarget")

    def _on_quest_status(self, msg: String, source: str) -> None:
        now = time.monotonic()
        try:
            payload = _json_sanitize(_json_load_dict(msg.data))
            valid = True
            error = ""
        except Exception as exc:
            payload = {"raw": msg.data}
            valid = False
            error = str(exc)
        with self._lock:
            latest = (
                self._quest_webvr_status
                if source == "webvr"
                else self._quest_retarget_status
            )
            updated = Latest(
                payload=payload,
                stamp_mono=now,
                count=latest.count + 1,
                valid=valid,
                last_error=error,
            )
            if source == "webvr":
                self._quest_webvr_status = updated
            else:
                self._quest_retarget_status = updated
            if self._active is not None:
                self._append_quest_event_locked(
                    self._active,
                    source,
                    updated,
                    now,
                    "status",
                )

    def _on_sharpa(self, msg: Any) -> None:
        payload = self._joint_payload(msg.joint_state, SHARPA_JOINT_COUNT)
        q_cmd, q_cmd_complete = _fixed_float_vector(msg.q_cmd, SHARPA_JOINT_COUNT)
        payload["q_cmd"] = q_cmd
        payload["q_cmd_complete"] = q_cmd_complete
        payload["q_cmd_valid"] = bool(msg.q_cmd_valid) and q_cmd_complete
        self._set_latest(self._sharpa, payload)

    def _on_zed_status(self, msg: String) -> None:
        try:
            payload = _json_sanitize(_json_load_dict(msg.data))
            self._set_latest(self._zed_status, payload, valid=True)
        except Exception as exc:
            self._set_latest(
                self._zed_status,
                {"raw": msg.data},
                valid=False,
                error=str(exc),
            )

    def _on_tactile_force6d(self, msg: Any) -> None:
        items = []
        valid = True
        for item in msg.forces:
            force = [_safe_float(value) for value in item.force]
            torque = [_safe_float(value) for value in item.torque]
            frame_id = int(item.frame_id)
            sensor_time = _safe_float(item.sensor_time)
            item_valid = (
                _tactile_metadata_valid(frame_id, sensor_time)
                and len(force) == 3
                and len(torque) == 3
                and all(
                    value is not None for value in [*force, *torque]
                )
            )
            valid = valid and item_valid
            items.append(
                {
                    "side": str(item.side).strip().lower(),
                    "finger": str(item.finger).strip().lower(),
                    "channel": int(item.channel),
                    "frame_id": frame_id,
                    "sensor_time": sensor_time,
                    "force": force,
                    "torque": torque,
                    "valid": item_valid,
                }
            )
        payload = {
            "header": _header_payload(msg),
            "count": len(items),
            "expected_count": len(TACTILE_PLANES),
            "complete": len(items) == len(TACTILE_PLANES),
            "items": items,
        }
        self._set_latest(self._tactile_force6d, payload, valid=valid and bool(items))

    def _on_tactile_contact(self, msg: Any) -> None:
        items = []
        valid = True
        for item in msg.contacts:
            points = [_safe_float(value) for value in item.points]
            frame_id = int(item.frame_id)
            sensor_time = _safe_float(item.sensor_time)
            item_valid = (
                _tactile_metadata_valid(frame_id, sensor_time)
                and all(value is not None for value in points)
                and len(points) % 3 == 0
            )
            valid = valid and item_valid
            items.append(
                {
                    "side": str(item.side).strip().lower(),
                    "finger": str(item.finger).strip().lower(),
                    "channel": int(item.channel),
                    "frame_id": frame_id,
                    "sensor_time": sensor_time,
                    "value_count": len(points),
                    "point_count_xyz": len(points) // 3 if len(points) % 3 == 0 else 0,
                    "points": points,
                    "valid": item_valid,
                }
            )
        payload = {
            "header": _header_payload(msg),
            "count": len(items),
            "expected_count": len(TACTILE_PLANES),
            "complete": len(items) == len(TACTILE_PLANES),
            "items": items,
        }
        self._set_latest(self._tactile_contact, payload, valid=valid and bool(items))

    def _on_tactile_deform(self, msg: Any) -> None:
        frame_bytes, metadata, error = self._deform_frame_bytes(msg)
        now = time.monotonic()
        now_ns = _now_unix_ns()
        wrote_frame = False
        raw_frame_index: int | None = None
        with self._lock:
            session = self._active
            if frame_bytes is not None and session is not None:
                raw_frame_index = session.deform_frames
                try:
                    session.raw_handle.write(frame_bytes)
                    session.deform_frames += 1
                    wrote_frame = True
                    frame_id, sensor_time, valid = self._deform_meta_vectors(metadata)
                    session.tactile_deform_time_unix_ns.append(now_ns)
                    session.tactile_deform_elapsed_ns.append(
                        _elapsed_ns(session.started_mono, now)
                    )
                    session.tactile_deform_timeline_row.append(
                        max(0, session.timeline_rows - 1)
                    )
                    session.tactile_deform_raw_frame_index.append(raw_frame_index)
                    session.tactile_deform_frame_id.append(frame_id)
                    session.tactile_deform_sensor_time.append(sensor_time)
                    session.tactile_deform_valid.append(valid)
                except Exception as exc:
                    error = str(exc)
                    session.deform_dropped += 1
            elif frame_bytes is None and session is not None:
                session.deform_dropped += 1
            payload = {
                "header": _header_payload(msg),
                "image_count": len(msg.images),
                "shape": list(TACTILE_DEFORM_SHAPE),
                "dtype": "uint8",
                "raw_frame_index": raw_frame_index,
                "wrote_frame": wrote_frame,
                "metadata": metadata,
            }
            self._tactile_deform = Latest(
                payload=payload,
                stamp_mono=now,
                count=self._tactile_deform.count + 1,
                valid=frame_bytes is not None,
                last_error=error,
            )

    def _set_latest(
        self,
        latest: Latest,
        payload: dict[str, Any],
        *,
        valid: bool | None = None,
        error: str = "",
    ) -> None:
        now = time.monotonic()
        with self._lock:
            latest.payload = payload
            latest.stamp_mono = now
            latest.count += 1
            latest.valid = bool(payload.get("valid", True)) if valid is None else valid
            latest.last_error = error

    def _joint_payload(self, msg: JointState, expected_count: int) -> dict[str, Any]:
        position, position_valid = _fixed_float_vector(msg.position, expected_count)
        velocity, velocity_valid = _fixed_float_vector(msg.velocity, expected_count)
        torque, torque_valid = _fixed_float_vector(msg.effort, expected_count)
        names = [str(name) for name in msg.name[:expected_count]]
        valid = position_valid and velocity_valid and torque_valid
        return {
            "header": _header_payload(msg),
            "name": names,
            "name_count": len(msg.name),
            "expected_count": expected_count,
            "position": position,
            "velocity": velocity,
            "torque": torque,
            "valid": valid,
            "complete": len(msg.name) >= expected_count,
        }

    def _deform_frame_bytes(self, msg: Any) -> tuple[bytes | None, list[dict[str, Any]], str]:
        metadata = []
        by_key: dict[tuple[str, str], Any] = {}
        error = ""
        for idx, image in enumerate(msg.images):
            side = str(image.side).strip().lower()
            finger = str(image.finger).strip().lower()
            key = (side, finger)
            data_len = len(image.data)
            frame_id = int(image.frame_id)
            sensor_time = _safe_float(image.sensor_time)
            image_valid = _tactile_metadata_valid(frame_id, sensor_time)
            image_meta = {
                "index": idx,
                "side": side,
                "finger": finger,
                "channel": int(image.channel),
                "frame_id": frame_id,
                "sensor_time": sensor_time,
                "height": int(image.height),
                "width": int(image.width),
                "byte_count": data_len,
                "valid": image_valid,
            }
            metadata.append(image_meta)
            if key in by_key:
                error = f"duplicate tactile deform plane {key}"
                continue
            by_key[key] = image

        missing = [key for key in TACTILE_PLANES if key not in by_key]
        if missing:
            return None, metadata, f"missing tactile deform planes: {missing}"
        invalid = [
            (str(item.get("side", "")), str(item.get("finger", "")))
            for item in metadata
            if not bool(item.get("valid", False))
        ]
        if invalid:
            return None, metadata, f"invalid tactile deform planes: {invalid}"

        chunks = []
        for key in TACTILE_PLANES:
            image = by_key[key]
            if int(image.height) != TACTILE_DEFORM_SHAPE[1] or int(image.width) != TACTILE_DEFORM_SHAPE[2]:
                return None, metadata, (
                    f"bad tactile deform shape for {key}: "
                    f"{int(image.height)}x{int(image.width)}"
                )
            data = bytes(image.data)
            expected_bytes = TACTILE_DEFORM_SHAPE[1] * TACTILE_DEFORM_SHAPE[2]
            if len(data) != expected_bytes:
                return None, metadata, (
                    f"bad tactile deform byte count for {key}: "
                    f"{len(data)} != {expected_bytes}"
                )
            chunks.append(data)
        frame = b"".join(chunks)
        if len(frame) != TACTILE_DEFORM_BYTES:
            return None, metadata, (
                f"bad tactile deform frame byte count: "
                f"{len(frame)} != {TACTILE_DEFORM_BYTES}"
            )
        return frame, metadata, error

    def _deform_meta_vectors(
        self,
        metadata: list[dict[str, Any]],
    ) -> tuple[list[int], list[float], list[bool]]:
        by_key = {
            (str(item.get("side", "")), str(item.get("finger", ""))): item
            for item in metadata
        }
        frame_id = []
        sensor_time = []
        valid = []
        for key in TACTILE_PLANES:
            item = by_key.get(key)
            if item is None:
                frame_id.append(0)
                sensor_time.append(math.nan)
                valid.append(False)
                continue
            frame_id.append(int(item.get("frame_id") or 0))
            sensor_time.append(_float_or_nan(item.get("sensor_time")))
            valid.append(
                bool(item.get("valid", False))
                and _tactile_metadata_valid(
                    int(item.get("frame_id") or 0),
                    _safe_float(item.get("sensor_time")),
                )
                and int(item.get("height") or 0) == TACTILE_DEFORM_SHAPE[1]
                and int(item.get("width") or 0) == TACTILE_DEFORM_SHAPE[2]
                and int(item.get("byte_count") or 0)
                == TACTILE_DEFORM_SHAPE[1] * TACTILE_DEFORM_SHAPE[2]
            )
        return frame_id, sensor_time, valid

    def _start_recording(self, status: dict[str, Any]) -> bool:
        with self._lock:
            if self._active is not None:
                return True
            blocked_reason = self._recording_preflight_block_reason(time.monotonic())
            if blocked_reason:
                if self.block_recording_on_tactile_error:
                    self.get_logger().error(
                        f"recording blocked before start: {blocked_reason}"
                    )
                    return False
                self.get_logger().error(
                    "recording started with tactile error; "
                    f"sample will be marked error if unresolved: {blocked_reason}"
                )
            self.recording_root.mkdir(parents=True, exist_ok=True)
            index = self._next_sample_index()
            sample_name = f"sample_{index:04d}"
            partial_dir = self.recording_root / f"{sample_name}.partial"
            final_dir = self.recording_root / sample_name
            if partial_dir.exists() or final_dir.exists():
                raise RuntimeError(f"sample directory already exists for {sample_name}")
            (partial_dir / "timeline").mkdir(parents=True)
            (partial_dir / "adam").mkdir(parents=True)
            (partial_dir / "sharpa" / "tactile").mkdir(parents=True)
            (partial_dir / "zed").mkdir(parents=True)
            (partial_dir / "status").mkdir(parents=True)
            (partial_dir / "quest").mkdir(parents=True)
            raw_handle = (partial_dir / "sharpa" / "tactile" / "deform_u8.raw.tmp").open("wb")
            self._active = RecordingSession(
                index=index,
                sample_name=sample_name,
                partial_dir=partial_dir,
                final_dir=final_dir,
                started_unix_ns=_now_unix_ns(),
                started_mono=time.monotonic(),
                start_status=status,
                raw_handle=raw_handle,
            )
            session = self._active
            self._append_status_event_locked(session, status, session.started_mono, "start")
            self._append_quest_event_locked(
                session,
                "webvr",
                self._quest_webvr_status,
                session.started_mono,
                "start_snapshot",
            )
            self._append_quest_event_locked(
                session,
                "retarget",
                self._quest_retarget_status,
                session.started_mono,
                "start_snapshot",
            )
        self._start_zed_video_capture(session)
        self.get_logger().info(f"recording started: {partial_dir}")
        return True

    def _recording_preflight_block_reason(self, now: float) -> str:
        if not self.require_tactile_fresh_on_start:
            return ""
        force_reason = self._latest_tactile_block_reason(
            self._tactile_force6d,
            "force6d",
            now,
        )
        deform_reason = self._latest_tactile_block_reason(
            self._tactile_deform,
            "deform",
            now,
        )
        contact_reason = self._latest_tactile_block_reason(
            self._tactile_contact,
            "contact_points",
            now,
        )
        reasons = [
            reason
            for reason in (force_reason, deform_reason, contact_reason)
            if reason
        ]
        return "; ".join(reasons)

    def _latest_tactile_block_reason(
        self,
        latest: Latest,
        label: str,
        now: float,
    ) -> str:
        if latest.count <= 0 or latest.payload is None:
            return f"{label} missing"
        age_ms = _age_ms_value(latest.stamp_mono, now)
        if not math.isfinite(age_ms) or age_ms > self.tactile_start_max_age_ms:
            return f"{label} stale latest age_ms={age_ms:.1f}"
        if not latest.valid:
            error = latest.last_error or "latest invalid"
            missing = self._invalid_tactile_labels(latest.payload)
            if missing:
                error = f"{error}; invalid={missing}"
            return f"{label} invalid: {error}"
        if not bool(latest.payload.get("complete", True)):
            return f"{label} incomplete"
        missing = self._invalid_tactile_labels(latest.payload)
        if missing:
            return f"{label} invalid channels: {missing}"
        return ""

    def _log_active_recording_tactile_problem(
        self,
        session: RecordingSession,
        now: float,
    ) -> None:
        reason = self._recording_preflight_block_reason(now)
        if not reason:
            if self._last_recording_tactile_error:
                self.get_logger().info(
                    f"recording tactile recovered: {session.sample_name}"
                )
            self._last_recording_tactile_error = ""
            return
        if (
            reason != self._last_recording_tactile_error
            or now - self._last_recording_tactile_error_mono
            >= self.tactile_error_log_period_s
        ):
            self.get_logger().error(
                f"recording tactile error in {session.sample_name}: {reason}"
            )
            self._last_recording_tactile_error = reason
            self._last_recording_tactile_error_mono = now

    @staticmethod
    def _invalid_tactile_labels(payload: dict[str, Any]) -> list[str]:
        items = payload.get("items")
        if not isinstance(items, list):
            metadata = payload.get("metadata")
            if isinstance(metadata, list):
                items = metadata
            else:
                return []
        invalid: list[str] = []
        by_key = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("side", "")).strip().lower(),
                str(item.get("finger", "")).strip().lower(),
            )
            by_key[key] = item
        for side, finger in TACTILE_PLANES:
            item = by_key.get((side, finger))
            if item is None:
                invalid.append(f"{side}/{finger}:missing")
                continue
            valid = bool(item.get("valid", False))
            frame_id = int(item.get("frame_id") or 0)
            sensor_time = _safe_float(item.get("sensor_time"))
            if not valid or not _tactile_metadata_valid(frame_id, sensor_time):
                invalid.append(f"{side}/{finger}:frame_id={frame_id}")
        return invalid

    def _stop_recording(self, status: dict[str, Any], reason: str) -> None:
        with self._lock:
            session = self._active
            if session is None:
                return
            self._active = None
            session.ended_unix_ns = _now_unix_ns()
            session.ended_mono = time.monotonic()
            session.stop_status = status
            session.stop_reason = reason
            self._append_status_event_locked(session, status, session.ended_mono, "stop")
            session.raw_handle.flush()
            session.raw_handle.close()

        self._stop_zed_video_capture(session)
        thread = threading.Thread(target=self._finalize_session, args=(session,), daemon=True)
        with self._lock:
            self._finalizers.append(thread)
        thread.start()
        self.get_logger().info(f"recording stopped: {session.partial_dir}; finalizing in background")

    def _append_status_event_locked(
        self,
        session: RecordingSession,
        status: dict[str, Any],
        now: float,
        event: str,
    ) -> None:
        session.status_events.append(
            {
                "event": event,
                "time_unix_ns": _now_unix_ns(),
                "elapsed_ns": _elapsed_ns(session.started_mono, now),
                "status": status,
            }
        )

    def _append_quest_event_locked(
        self,
        session: RecordingSession,
        source: str,
        latest: Latest,
        now: float,
        event: str,
    ) -> None:
        row = {
            "event": event,
            "time_unix_ns": _now_unix_ns(),
            "elapsed_ns": _elapsed_ns(session.started_mono, now),
            "valid": latest.valid,
            "source_count": latest.count,
            "status": latest.payload,
            "error": latest.last_error,
        }
        if source == "webvr":
            session.quest_webvr_events.append(row)
        elif source == "retarget":
            session.quest_retarget_events.append(row)
        else:
            raise ValueError(f"unknown Quest status source: {source}")

    def _next_sample_index(self) -> int:
        # Only normal sample directories participate in the normal counter.
        # Error directories use sample_XXXX_error_N and must not advance it.
        pattern = re.compile(r"^sample_(\d+)(?:\.partial)?$")
        highest = 0
        for child in self.recording_root.iterdir():
            match = pattern.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _next_error_index(self) -> int:
        pattern = re.compile(r"^sample_\d+_error_(\d+)$")
        highest = -1
        for child in self.recording_root.iterdir():
            match = pattern.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _error_final_dir(self, sample_name: str) -> Path:
        while True:
            index = self._next_error_index()
            candidate = self.recording_root / f"{sample_name}_error_{index}"
            if not candidate.exists():
                return candidate

    def _sample_tick(self) -> None:
        now = time.monotonic()
        now_ns = _now_unix_ns()
        with self._lock:
            session = self._active
            if session is None:
                return
            self._log_active_recording_tactile_problem(session, now)
            session.time_unix_ns.append(now_ns)
            session.elapsed_ns.append(_elapsed_ns(session.started_mono, now))
            self._append_source_quality(session, now)
            self._append_joint_sample(session, self._adam, ADAM_JOINT_COUNT, "adam", now)
            self._append_joint_sample(
                session,
                self._sharpa,
                SHARPA_JOINT_COUNT,
                "sharpa",
                now,
            )
            self._append_tactile_force_sample(session)
            self._append_tactile_contact_sample(session)
            session.timeline_rows += 1

    def _append_source_quality(self, session: RecordingSession, now: float) -> None:
        latest = [
            self._status,
            self._adam,
            self._sharpa,
            self._tactile_force6d,
            self._tactile_contact,
            self._tactile_deform,
            self._zed_status,
        ]
        session.source_valid.append([item.valid for item in latest])
        session.source_age_ms.append([_age_ms_value(item.stamp_mono, now) for item in latest])
        session.source_count.append([item.count for item in latest])

    def _append_joint_sample(
        self,
        session: RecordingSession,
        latest: Latest,
        expected_count: int,
        prefix: str,
        now: float,
    ) -> None:
        payload = latest.payload or {}
        q = _nan_vector(payload.get("position"), expected_count)
        dq = _nan_vector(payload.get("velocity"), expected_count)
        tau = _nan_vector(payload.get("torque"), expected_count)
        q_cmd = _nan_vector(payload.get("q_cmd"), expected_count)
        q_cmd_valid = bool(payload.get("q_cmd_valid", False))
        valid = bool(latest.valid and payload.get("complete", False))
        age = _age_ms_value(latest.stamp_mono, now)
        header_stamp_ns = _header_stamp_ns(payload)
        if prefix == "adam":
            session.adam_q.append(q)
            session.adam_dq.append(dq)
            session.adam_tau.append(tau)
            session.adam_valid.append(valid)
            session.adam_age_ms.append(age)
            session.adam_source_count.append(latest.count)
            session.adam_header_stamp_ns.append(header_stamp_ns)
        elif prefix == "sharpa":
            session.sharpa_q.append(q)
            session.sharpa_dq.append(dq)
            session.sharpa_tau.append(tau)
            session.sharpa_q_cmd.append(q_cmd)
            session.sharpa_q_cmd_valid.append(q_cmd_valid)
            session.sharpa_valid.append(valid)
            session.sharpa_age_ms.append(age)
            session.sharpa_source_count.append(latest.count)
            session.sharpa_header_stamp_ns.append(header_stamp_ns)
        else:
            raise ValueError(f"unknown joint sample prefix: {prefix}")

    def _append_tactile_force_sample(self, session: RecordingSession) -> None:
        payload = self._tactile_force6d.payload or {}
        by_key = self._items_by_tactile_key(payload.get("items", []))
        force_row: list[list[float]] = []
        torque_row: list[list[float]] = []
        frame_id_row: list[int] = []
        sensor_time_row: list[float] = []
        valid_row: list[bool] = []
        for key in TACTILE_PLANES:
            item = by_key.get(key)
            if item is None:
                force_row.append([math.nan, math.nan, math.nan])
                torque_row.append([math.nan, math.nan, math.nan])
                frame_id_row.append(0)
                sensor_time_row.append(math.nan)
                valid_row.append(False)
                continue
            force_row.append(_nan_vector(item.get("force"), 3))
            torque_row.append(_nan_vector(item.get("torque"), 3))
            frame_id_row.append(int(item.get("frame_id") or 0))
            sensor_time_row.append(_float_or_nan(item.get("sensor_time")))
            valid_row.append(bool(item.get("valid", False)))
        session.tactile_force.append(force_row)
        session.tactile_torque.append(torque_row)
        session.tactile_force_frame_id.append(frame_id_row)
        session.tactile_force_sensor_time.append(sensor_time_row)
        session.tactile_force_valid.append(valid_row)

    def _append_tactile_contact_sample(self, session: RecordingSession) -> None:
        payload = self._tactile_contact.payload or {}
        by_key = self._items_by_tactile_key(payload.get("items", []))
        offset_row: list[int] = []
        count_row: list[int] = []
        frame_id_row: list[int] = []
        sensor_time_row: list[float] = []
        valid_row: list[bool] = []
        for key in TACTILE_PLANES:
            item = by_key.get(key)
            if item is None:
                offset_row.append(len(session.tactile_contact_points_xyz))
                count_row.append(0)
                frame_id_row.append(0)
                sensor_time_row.append(math.nan)
                valid_row.append(False)
                continue
            points = item.get("points", [])
            flat = _nan_vector(points, len(points) if hasattr(points, "__len__") else 0)
            item_valid = (
                bool(item.get("valid", False))
                and len(flat) % 3 == 0
                and all(math.isfinite(value) for value in flat)
            )
            offset = len(session.tactile_contact_points_xyz)
            point_count = 0
            if item_valid:
                for idx in range(0, len(flat), 3):
                    session.tactile_contact_points_xyz.append(
                        [flat[idx], flat[idx + 1], flat[idx + 2]]
                    )
                point_count = len(flat) // 3
            offset_row.append(offset)
            count_row.append(point_count)
            frame_id_row.append(int(item.get("frame_id") or 0))
            sensor_time_row.append(_float_or_nan(item.get("sensor_time")))
            valid_row.append(item_valid)
        session.tactile_contact_offset.append(offset_row)
        session.tactile_contact_count.append(count_row)
        session.tactile_contact_frame_id.append(frame_id_row)
        session.tactile_contact_sensor_time.append(sensor_time_row)
        session.tactile_contact_valid.append(valid_row)

    @staticmethod
    def _items_by_tactile_key(items: Any) -> dict[tuple[str, str], dict[str, Any]]:
        output: dict[tuple[str, str], dict[str, Any]] = {}
        if not isinstance(items, list):
            return output
        for item in items:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("side", "")).strip().lower(),
                str(item.get("finger", "")).strip().lower(),
            )
            output[key] = item
        return output

    def _zed_status_monitor_stream(self) -> dict[str, Any]:
        payload = self._zed_status.payload or {}
        if not isinstance(payload, dict):
            return {}
        video = payload.get("video", {})
        if isinstance(video, dict):
            stream = video.get("monitor_stream")
            if isinstance(stream, dict):
                return stream
        return {}

    def _zed_video_source(self) -> dict[str, Any]:
        stream = self._zed_status_monitor_stream()
        port = self.zed_video_rtp_port
        if stream.get("port") is not None:
            parsed_port = _safe_float(stream.get("port"))
            if parsed_port is not None:
                port = int(parsed_port)
        return {
            "transport": "rtp/udp",
            "bind_host": "0.0.0.0",
            "port": port,
            "payload_type": int(stream.get("payload_type") or 96),
            "clock_rate": int(stream.get("clock_rate") or 90000),
            "zed_status_enabled": stream.get("enabled"),
            "zed_status_host": stream.get("host"),
        }

    def _write_zed_video_sdp(self, session: RecordingSession, source: dict[str, Any]) -> Path:
        port = int(source["port"])
        payload_type = int(source["payload_type"])
        clock_rate = int(source["clock_rate"])
        sdp = (
            "v=0\n"
            "o=- 0 0 IN IP4 127.0.0.1\n"
            "s=ZED RGB H264\n"
            "c=IN IP4 0.0.0.0\n"
            "t=0 0\n"
            f"m=video {port} RTP/AVP {payload_type}\n"
            f"a=rtpmap:{payload_type} H264/{clock_rate}\n"
            f"a=fmtp:{payload_type} packetization-mode=1\n"
        )
        path = session.partial_dir / "zed" / "rtp.sdp"
        path.write_text(sdp, encoding="ascii")
        return path

    def _start_zed_video_capture(self, session: RecordingSession) -> None:
        if not self.record_zed_video:
            session.zed_video_error = "record_zed_video_false"
            return
        source = self._zed_video_source()
        if source.get("zed_status_enabled") is False:
            session.zed_video_error = "zed_monitor_stream_disabled"
            self.get_logger().warning("ZED video capture skipped: monitor RTP stream is disabled")
            return
        if shutil.which(self.zed_video_ffmpeg) is None:
            session.zed_video_error = f"ffmpeg not found: {self.zed_video_ffmpeg}"
            self.get_logger().error(session.zed_video_error)
            return

        sdp_path = self._write_zed_video_sdp(session, source)
        output_tmp = session.partial_dir / "zed" / "rgb.mp4.tmp"
        log_path = session.partial_dir / "zed" / "ffmpeg.log.tmp"
        command = [
            self.zed_video_ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-protocol_whitelist",
            "file,udp,rtp",
            "-fflags",
            "+genpts",
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            str(sdp_path),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(output_tmp),
        ]
        try:
            log_handle = log_path.open("wb")
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            session.zed_video_process = process
            session.zed_video_log_handle = log_handle
            time.sleep(0.2)
            if process.poll() is not None:
                session.zed_video_error = f"ffmpeg exited during startup with code {process.returncode}"
                log_handle.close()
                session.zed_video_log_handle = None
                session.zed_video_process = None
                self.get_logger().error(session.zed_video_error)
                return
            session.zed_video_started = True
            self.get_logger().info(
                "ZED video capture started: "
                f"rtp://0.0.0.0:{int(source['port'])} -> {output_tmp}"
            )
        except Exception as exc:
            session.zed_video_error = str(exc)
            if session.zed_video_log_handle is not None:
                session.zed_video_log_handle.close()
                session.zed_video_log_handle = None
            self.get_logger().error(f"failed to start ZED video capture: {exc}")

    def _stop_zed_video_capture(self, session: RecordingSession) -> None:
        process = session.zed_video_process
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        if process.stdin is not None:
                            process.stdin.write(b"q\n")
                            process.stdin.flush()
                    except Exception:
                        pass
                    try:
                        process.wait(timeout=self.zed_video_stop_timeout_s)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except Exception:
                            process.terminate()
                        try:
                            process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except Exception:
                                process.kill()
                if process.returncode not in (0, 255) and not session.zed_video_error:
                    session.zed_video_error = f"ffmpeg exited with code {process.returncode}"
            finally:
                if session.zed_video_log_handle is not None:
                    session.zed_video_log_handle.close()
                    session.zed_video_log_handle = None
                session.zed_video_process = None

        tmp_path = session.partial_dir / "zed" / "rgb.mp4.tmp"
        final_path = session.partial_dir / "zed" / "rgb.mp4"
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.rename(final_path)
        elif session.zed_video_started and not session.zed_video_error:
            session.zed_video_error = "rgb.mp4.tmp missing or empty after ffmpeg stop"

    def _finalize_session(self, session: RecordingSession) -> None:
        try:
            final_dir = self._finalize_session_inner(session)
            self.get_logger().info(f"recording complete: {final_dir}")
        except Exception as exc:
            try:
                (session.partial_dir / "ERROR").write_text(str(exc) + "\n", encoding="utf-8")
            except Exception:
                pass
            self.get_logger().error(f"failed to finalize {session.partial_dir}: {exc}")

    def _finalize_session_inner(self, session: RecordingSession) -> Path:
        raw_tmp = session.partial_dir / "sharpa" / "tactile" / "deform_u8.raw.tmp"
        raw_path = session.partial_dir / "sharpa" / "tactile" / "deform_u8.raw"
        zst_path = session.partial_dir / "sharpa" / "tactile" / "deform_u8.zst"
        self._compress_deform_raw(raw_tmp, zst_path)
        raw_deleted = False
        raw_kept = False
        if self.keep_raw_deform:
            raw_tmp.rename(raw_path)
            raw_kept = True
        else:
            raw_tmp.unlink()
            raw_deleted = True

        video_log_tmp = session.partial_dir / "zed" / "ffmpeg.log.tmp"
        video_log_path = session.partial_dir / "zed" / "ffmpeg.log"
        if video_log_tmp.exists():
            video_log_tmp.rename(video_log_path)

        self._write_schema(session)
        self._write_timeline_arrays(session)
        self._write_adam_arrays(session)
        self._write_sharpa_arrays(session)
        self._write_tactile_arrays(session)
        self._write_status_events(session)
        self._write_quest_events(session)
        self._write_zed_meta(session)

        quality_report = self._session_quality_report(session)
        final_dir = session.final_dir
        if not quality_report["ok"]:
            with self._lock:
                final_dir = self._error_final_dir(session.sample_name)
            session.final_dir = final_dir
            quality_report["final_directory"] = final_dir.name
            self._write_quality_error_files(session, quality_report)
            self.get_logger().error(
                f"recording quality error: {session.sample_name} -> "
                f"{final_dir.name}; errors={len(quality_report['errors'])}"
            )

        manifest = self._manifest(
            session,
            raw_deleted=raw_deleted,
            raw_kept=raw_kept,
            quality_report=quality_report,
        )
        _write_json(session.partial_dir / "manifest.json", manifest)
        self._write_checksums(session.partial_dir)
        (session.partial_dir / "COMPLETE").write_text(
            json.dumps(
                {
                    "completed_unix_ns": _now_unix_ns(),
                    "sample": session.sample_name,
                    "state": manifest["state"],
                    "quality_ok": quality_report["ok"],
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        with self._lock:
            session.partial_dir.rename(final_dir)
        return final_dir

    def _session_quality_report(self, session: RecordingSession) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        ended_mono = session.ended_mono if session.ended_mono is not None else time.monotonic()
        duration_s = max(0.0, ended_mono - session.started_mono)
        timeline_rows = session.timeline_rows

        def add_error(code: str, message: str, **details: Any) -> None:
            errors.append(
                {
                    "code": code,
                    "message": message,
                    "details": _json_sanitize(details),
                }
            )

        def add_warning(code: str, message: str, **details: Any) -> None:
            warnings.append(
                {
                    "code": code,
                    "message": message,
                    "details": _json_sanitize(details),
                }
            )

        def check_row_count(name: str, rows: Any, expected: int) -> None:
            actual = len(rows) if hasattr(rows, "__len__") else 0
            if actual != expected:
                add_error(
                    "row_count_mismatch",
                    f"{name} row count mismatch",
                    item=name,
                    expected=expected,
                    actual=actual,
                )

        if timeline_rows <= 0:
            add_error("timeline_empty", "timeline has no rows")
        if session.ended_unix_ns is None or session.ended_mono is None:
            add_error("missing_stop_time", "recording stop timestamp is missing")

        for name, rows in (
            ("source_valid", session.source_valid),
            ("source_age_ms", session.source_age_ms),
            ("source_count", session.source_count),
            ("adam_valid", session.adam_valid),
            ("adam_q", session.adam_q),
            ("adam_dq", session.adam_dq),
            ("adam_tau", session.adam_tau),
            ("sharpa_valid", session.sharpa_valid),
            ("sharpa_q", session.sharpa_q),
            ("sharpa_dq", session.sharpa_dq),
            ("sharpa_tau", session.sharpa_tau),
            ("sharpa_q_cmd", session.sharpa_q_cmd),
            ("sharpa_q_cmd_valid", session.sharpa_q_cmd_valid),
            ("tactile_force_valid", session.tactile_force_valid),
            ("tactile_contact_valid", session.tactile_contact_valid),
        ):
            check_row_count(name, rows, timeline_rows)

        self._append_source_quality_errors(session, errors)
        self._append_joint_quality_errors(
            session,
            errors,
            prefix="adam",
            q_rows=session.adam_q,
            dq_rows=session.adam_dq,
            tau_rows=session.adam_tau,
            valid_rows=session.adam_valid,
            age_rows=session.adam_age_ms,
            expected_count=ADAM_JOINT_COUNT,
            max_age_ms=SOURCE_MAX_AGE_MS["adam"],
        )
        self._append_joint_quality_errors(
            session,
            errors,
            prefix="sharpa",
            q_rows=session.sharpa_q,
            dq_rows=session.sharpa_dq,
            tau_rows=session.sharpa_tau,
            valid_rows=session.sharpa_valid,
            age_rows=session.sharpa_age_ms,
            expected_count=SHARPA_JOINT_COUNT,
            max_age_ms=SOURCE_MAX_AGE_MS["sharpa"],
        )
        self._append_q_cmd_quality_errors(
            errors,
            q_cmd_rows=session.sharpa_q_cmd,
            valid_rows=session.sharpa_q_cmd_valid,
            expected_count=SHARPA_JOINT_COUNT,
        )
        self._append_tactile_quality_errors(
            errors,
            label="tactile_force6d",
            frame_id_rows=session.tactile_force_frame_id,
            sensor_time_rows=session.tactile_force_sensor_time,
            valid_rows=session.tactile_force_valid,
            expected_min_rows=timeline_rows,
            duration_s=duration_s,
        )
        self._append_tactile_quality_errors(
            errors,
            label="tactile_contact_points",
            frame_id_rows=session.tactile_contact_frame_id,
            sensor_time_rows=session.tactile_contact_sensor_time,
            valid_rows=session.tactile_contact_valid,
            expected_min_rows=timeline_rows,
            duration_s=duration_s,
        )
        deform_expected_min = timeline_rows
        if duration_s >= 1.0:
            deform_expected_min = max(1, int(math.floor(timeline_rows * 0.5)))
        self._append_tactile_quality_errors(
            errors,
            label="tactile_deform",
            frame_id_rows=session.tactile_deform_frame_id,
            sensor_time_rows=session.tactile_deform_sensor_time,
            valid_rows=session.tactile_deform_valid,
            expected_min_rows=deform_expected_min,
            duration_s=duration_s,
        )
        if session.deform_dropped:
            add_error(
                "tactile_deform_dropped",
                "tactile deform frames were dropped while recording",
                dropped=session.deform_dropped,
                written=session.deform_frames,
            )
        if session.deform_frames <= 0:
            add_error("tactile_deform_empty", "no tactile deform frames were written")

        self._append_quest_quality_errors(session, errors, warnings)
        self._append_zed_quality_errors(session, errors, warnings)
        self._append_file_quality_errors(session, errors)

        event_names = [
            str(item.get("event", ""))
            for item in session.status_events
            if isinstance(item, dict)
        ]
        if "start" not in event_names:
            add_error("status_event_missing", "recording start status event is missing")
        if "stop" not in event_names:
            add_error("status_event_missing", "recording stop status event is missing")

        report = {
            "schema": "pnd.recording_quality.v1",
            "sample": session.sample_name,
            "final_directory": session.final_dir.name,
            "ok": not errors,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "duration_s": round(duration_s, 6),
            "timeline_rows": timeline_rows,
            "errors": errors,
            "warnings": warnings,
        }
        return _json_sanitize(report)

    @staticmethod
    def _append_quest_quality_errors(
        session: RecordingSession,
        errors: list[dict[str, Any]],
        warnings: list[dict[str, Any]],
    ) -> None:
        webvr_statuses = [
            row.get("status")
            for row in session.quest_webvr_events
            if row.get("valid") and isinstance(row.get("status"), dict)
        ]
        active_statuses = [
            status for status in webvr_statuses if status.get("connected") is True
        ]
        if not active_statuses:
            return

        lost_states: set[str] = set()
        held_states: set[str] = set()
        receive_gap_ms = 0.0
        sequence_gaps: list[int] = []
        for status in active_statuses:
            positions = status.get("hand_position_tracking")
            if isinstance(positions, dict):
                for hand, state in positions.items():
                    if str(state).casefold() not in {"known", "inferred"}:
                        lost_states.add(f"{hand}:{state}")
            gates = status.get("hand_gates")
            if isinstance(gates, dict):
                for hand, gate in gates.items():
                    if isinstance(gate, dict) and gate.get("state") != "Normal":
                        held_states.add(f"{hand}:{gate.get('state')}")
            age = _safe_float(status.get("receive_age_ms"))
            if age is not None:
                receive_gap_ms = max(receive_gap_ms, age)
            gaps = _safe_float(status.get("source_sequence_gaps"))
            if gaps is not None:
                sequence_gaps.append(int(gaps))

        if lost_states:
            errors.append(
                {
                    "code": "quest_openxr_position_lost",
                    "message": "Quest controller position tracking became unavailable",
                    "details": {"states": sorted(lost_states)},
                }
            )
        if held_states:
            errors.append(
                {
                    "code": "quest_hand_gate_held",
                    "message": "Quest hand execution gate left Normal state",
                    "details": {"states": sorted(held_states)},
                }
            )
        sequence_gap_delta = (
            max(sequence_gaps) - min(sequence_gaps) if sequence_gaps else 0
        )
        if receive_gap_ms > 200.0 or sequence_gap_delta > 0:
            errors.append(
                {
                    "code": "quest_transport_frame_gap",
                    "message": "Quest tracking transport exceeded the collection limit",
                    "details": {
                        "max_receive_age_ms": receive_gap_ms,
                        "source_sequence_gap_delta": sequence_gap_delta,
                    },
                }
            )

        residuals: list[float] = []
        for row in session.quest_retarget_events:
            status = row.get("status")
            if not row.get("valid") or not isinstance(status, dict):
                continue
            values = status.get("wrist_position_residual_mm")
            if not isinstance(values, dict):
                continue
            for item in values.values():
                value = _safe_float(item)
                if value is not None:
                    residuals.append(value)
        max_residual_mm = max(residuals, default=0.0)
        if max_residual_mm > 100.0:
            warnings.append(
                {
                    "code": "quest_retarget_residual_high",
                    "message": "Quest wrist IK residual exceeded 100 mm",
                    "details": {"max_wrist_position_residual_mm": max_residual_mm},
                }
            )

    def _append_source_quality_errors(
        self,
        session: RecordingSession,
        errors: list[dict[str, Any]],
    ) -> None:
        rows = len(session.source_valid)
        if rows <= 0:
            errors.append(
                {
                    "code": "source_quality_empty",
                    "message": "source quality timeline is empty",
                    "details": {},
                }
            )
            return
        for source_idx, source_name in enumerate(SOURCE_NAMES):
            if source_name == "zed_status" and not self.record_zed_video:
                continue
            invalid_rows: list[int] = []
            stale_rows: list[int] = []
            max_age_ms = SOURCE_MAX_AGE_MS.get(source_name, 500.0)
            for row_idx in range(rows):
                valid_row = session.source_valid[row_idx]
                age_row = (
                    session.source_age_ms[row_idx]
                    if row_idx < len(session.source_age_ms)
                    else []
                )
                valid = source_idx < len(valid_row) and bool(valid_row[source_idx])
                age = (
                    float(age_row[source_idx])
                    if source_idx < len(age_row)
                    else math.nan
                )
                if not valid:
                    invalid_rows.append(row_idx)
                if not math.isfinite(age) or age > max_age_ms:
                    stale_rows.append(row_idx)
            if invalid_rows:
                errors.append(
                    {
                        "code": "source_invalid",
                        "message": f"{source_name} has invalid rows",
                        "details": {
                            "source": source_name,
                            "invalid_count": len(invalid_rows),
                            "first_rows": invalid_rows[:10],
                        },
                    }
                )
            if stale_rows:
                errors.append(
                    {
                        "code": "source_stale",
                        "message": f"{source_name} has stale or missing rows",
                        "details": {
                            "source": source_name,
                            "max_age_ms": max_age_ms,
                            "stale_count": len(stale_rows),
                            "first_rows": stale_rows[:10],
                        },
                    }
                )

    @staticmethod
    def _append_joint_quality_errors(
        session: RecordingSession,
        errors: list[dict[str, Any]],
        *,
        prefix: str,
        q_rows: list[list[float]],
        dq_rows: list[list[float]],
        tau_rows: list[list[float]],
        valid_rows: list[bool],
        age_rows: list[float],
        expected_count: int,
        max_age_ms: float,
    ) -> None:
        invalid_rows = [
            idx
            for idx, value in enumerate(valid_rows)
            if not bool(value)
        ]
        stale_rows = [
            idx
            for idx, age in enumerate(age_rows)
            if not math.isfinite(float(age)) or float(age) > max_age_ms
        ]
        bad_shape_rows: list[int] = []
        bad_value_rows: list[int] = []
        row_count = max(len(q_rows), len(dq_rows), len(tau_rows))
        for idx in range(row_count):
            vectors = []
            for rows in (q_rows, dq_rows, tau_rows):
                vectors.append(rows[idx] if idx < len(rows) else [])
            if any(len(vector) != expected_count for vector in vectors):
                bad_shape_rows.append(idx)
                continue
            if any(
                not math.isfinite(float(value))
                for vector in vectors
                for value in vector
            ):
                bad_value_rows.append(idx)
        if invalid_rows:
            errors.append(
                {
                    "code": "joint_invalid",
                    "message": f"{prefix} joint rows are invalid",
                    "details": {
                        "source": prefix,
                        "invalid_count": len(invalid_rows),
                        "first_rows": invalid_rows[:10],
                    },
                }
            )
        if stale_rows:
            errors.append(
                {
                    "code": "joint_stale",
                    "message": f"{prefix} joint rows are stale",
                    "details": {
                        "source": prefix,
                        "max_age_ms": max_age_ms,
                        "stale_count": len(stale_rows),
                        "first_rows": stale_rows[:10],
                    },
                }
            )
        if bad_shape_rows:
            errors.append(
                {
                    "code": "joint_shape",
                    "message": f"{prefix} joint vectors have the wrong shape",
                    "details": {
                        "source": prefix,
                        "expected_count": expected_count,
                        "bad_count": len(bad_shape_rows),
                        "first_rows": bad_shape_rows[:10],
                    },
                }
            )
        if bad_value_rows:
            errors.append(
                {
                    "code": "joint_nonfinite",
                    "message": f"{prefix} joint vectors contain NaN or inf",
                    "details": {
                        "source": prefix,
                        "bad_count": len(bad_value_rows),
                        "first_rows": bad_value_rows[:10],
                    },
                }
            )

    @staticmethod
    def _append_q_cmd_quality_errors(
        errors: list[dict[str, Any]],
        *,
        q_cmd_rows: list[list[float]],
        valid_rows: list[bool],
        expected_count: int,
    ) -> None:
        bad_shape_rows: list[int] = []
        bad_value_rows: list[int] = []
        for idx, valid in enumerate(valid_rows):
            if not valid:
                continue
            vector = q_cmd_rows[idx] if idx < len(q_cmd_rows) else []
            if len(vector) != expected_count:
                bad_shape_rows.append(idx)
            elif any(not math.isfinite(float(value)) for value in vector):
                bad_value_rows.append(idx)
        if bad_shape_rows:
            errors.append(
                {
                    "code": "sharpa_q_cmd_shape",
                    "message": "Sharpa q_cmd vectors have the wrong shape",
                    "details": {
                        "expected_count": expected_count,
                        "bad_count": len(bad_shape_rows),
                        "first_rows": bad_shape_rows[:10],
                    },
                }
            )
        if bad_value_rows:
            errors.append(
                {
                    "code": "sharpa_q_cmd_nonfinite",
                    "message": "Valid Sharpa q_cmd vectors contain NaN or inf",
                    "details": {
                        "bad_count": len(bad_value_rows),
                        "first_rows": bad_value_rows[:10],
                    },
                }
            )

    @staticmethod
    def _append_tactile_quality_errors(
        errors: list[dict[str, Any]],
        *,
        label: str,
        frame_id_rows: list[list[int]],
        sensor_time_rows: list[list[float]],
        valid_rows: list[list[bool]],
        expected_min_rows: int,
        duration_s: float,
    ) -> None:
        row_count = max(len(frame_id_rows), len(sensor_time_rows), len(valid_rows))
        if row_count < expected_min_rows:
            errors.append(
                {
                    "code": "tactile_row_count",
                    "message": f"{label} has too few rows",
                    "details": {
                        "source": label,
                        "expected_min_rows": expected_min_rows,
                        "actual_rows": row_count,
                    },
                }
            )
        if row_count <= 0:
            return
        min_span_s = duration_s * 0.5 if duration_s >= 1.0 else 0.0
        for plane_idx, (side, finger) in enumerate(TACTILE_PLANES):
            invalid_rows: list[int] = []
            frames: list[int] = []
            sensor_times: list[float] = []
            for row_idx in range(row_count):
                valid = (
                    row_idx < len(valid_rows)
                    and plane_idx < len(valid_rows[row_idx])
                    and bool(valid_rows[row_idx][plane_idx])
                )
                frame_id = (
                    int(frame_id_rows[row_idx][plane_idx])
                    if row_idx < len(frame_id_rows)
                    and plane_idx < len(frame_id_rows[row_idx])
                    else 0
                )
                sensor_time = (
                    float(sensor_time_rows[row_idx][plane_idx])
                    if row_idx < len(sensor_time_rows)
                    and plane_idx < len(sensor_time_rows[row_idx])
                    else math.nan
                )
                metadata_valid = _tactile_metadata_valid(frame_id, sensor_time)
                if not valid or not metadata_valid:
                    invalid_rows.append(row_idx)
                    continue
                frames.append(frame_id)
                sensor_times.append(sensor_time)
            plane = f"{side}/{finger}"
            if invalid_rows:
                errors.append(
                    {
                        "code": "tactile_invalid",
                        "message": f"{label} {plane} has invalid rows",
                        "details": {
                            "source": label,
                            "plane": plane,
                            "invalid_count": len(invalid_rows),
                            "first_rows": invalid_rows[:10],
                        },
                    }
                )
                continue
            if row_count >= 3 and len(set(frames)) <= 1:
                errors.append(
                    {
                        "code": "tactile_frozen_frame_id",
                        "message": f"{label} {plane} frame_id did not change",
                        "details": {
                            "source": label,
                            "plane": plane,
                            "rows": row_count,
                            "unique_frame_id": len(set(frames)),
                            "frame_id": frames[0] if frames else 0,
                        },
                    }
                )
            if sensor_times and min_span_s > 0.0:
                span_s = max(sensor_times) - min(sensor_times)
                if span_s < min_span_s:
                    errors.append(
                        {
                            "code": "tactile_sensor_time_span",
                            "message": f"{label} {plane} sensor_time span is too short",
                            "details": {
                                "source": label,
                                "plane": plane,
                                "duration_s": round(duration_s, 6),
                                "expected_min_span_s": round(min_span_s, 6),
                                "actual_span_s": round(span_s, 6),
                            },
                        }
                    )

    def _append_zed_quality_errors(
        self,
        session: RecordingSession,
        errors: list[dict[str, Any]],
        warnings: list[dict[str, Any]],
    ) -> None:
        if not self.record_zed_video:
            warnings.append(
                {
                    "code": "zed_video_disabled",
                    "message": "ZED video recording is disabled",
                    "details": {},
                }
            )
            return
        video_path = session.partial_dir / "zed" / "rgb.mp4"
        if not session.zed_video_started:
            errors.append(
                {
                    "code": "zed_video_not_started",
                    "message": "ZED video capture did not start",
                    "details": {"error": session.zed_video_error},
                }
            )
        if session.zed_video_error:
            errors.append(
                {
                    "code": "zed_video_error",
                    "message": "ZED video capture reported an error",
                    "details": {"error": session.zed_video_error},
                }
            )
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            errors.append(
                {
                    "code": "zed_video_missing",
                    "message": "ZED rgb.mp4 is missing or empty",
                    "details": {
                        "path": "zed/rgb.mp4",
                        "exists": video_path.is_file(),
                        "bytes": video_path.stat().st_size if video_path.is_file() else 0,
                    },
                }
            )

    def _append_file_quality_errors(
        self,
        session: RecordingSession,
        errors: list[dict[str, Any]],
    ) -> None:
        expected_files = [
            "schema.json",
            "timeline/clock.npz",
            "timeline/source_quality.npz",
            "adam/physical_31.npz",
            "sharpa/joints_44.npz",
            "sharpa/tactile/deform_index.npz",
            "sharpa/tactile/deform_u8.zst",
            "sharpa/tactile/force6d.npz",
            "sharpa/tactile/contact_points.npz",
            "status/events.jsonl",
            "quest/webvr_status.jsonl",
            "quest/retarget_status.jsonl",
            "zed/rgb_meta.json",
        ]
        if self.record_zed_video:
            expected_files.append("zed/rgb.mp4")
        for relative in expected_files:
            path = session.partial_dir / relative
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(
                    {
                        "code": "file_missing_or_empty",
                        "message": f"{relative} is missing or empty",
                        "details": {
                            "path": relative,
                            "exists": path.is_file(),
                            "bytes": path.stat().st_size if path.is_file() else 0,
                        },
                    }
                )

    def _write_quality_error_files(
        self,
        session: RecordingSession,
        quality_report: dict[str, Any],
    ) -> None:
        _write_json(session.partial_dir / "RECORDING_ERRORS.json", quality_report)
        lines = [
            f"sample: {session.sample_name}",
            f"final_directory: {quality_report.get('final_directory', session.final_dir.name)}",
            f"error_count: {quality_report.get('error_count', 0)}",
            "",
        ]
        for idx, error in enumerate(quality_report.get("errors", [])):
            if not isinstance(error, dict):
                continue
            lines.append(
                f"{idx}. [{error.get('code', '')}] {error.get('message', '')}"
            )
            details = error.get("details")
            if details:
                lines.append(
                    "   details: "
                    + json.dumps(details, ensure_ascii=True, sort_keys=True)
                )
        lines.append("")
        (session.partial_dir / "RECORDING_ERRORS.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        (session.partial_dir / "ERROR").write_text(
            "recording quality error; see RECORDING_ERRORS.txt and RECORDING_ERRORS.json\n",
            encoding="utf-8",
        )

    def _compress_deform_raw(self, raw_tmp: Path, zst_path: Path) -> None:
        zstd = shutil.which("zstd")
        if zstd is None:
            raise RuntimeError("zstd executable not found; cannot create sharpa/tactile/deform_u8.zst")
        subprocess.run(
            [zstd, f"-{self.zstd_level}", "-f", str(raw_tmp), "-o", str(zst_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _write_schema(self, session: RecordingSession) -> None:
        adam_names = self._latest_joint_names(self._adam, ADAM_JOINT_COUNT)
        sharpa_names = self._latest_joint_names(self._sharpa, SHARPA_JOINT_COUNT)
        schema = {
            "schema_version": 2,
            "format": "pnd_local_monitor_columnar",
            "sample": session.sample_name,
            "timeline": {
                "rate_hz": self.sample_rate_hz,
                "row_count": session.timeline_rows,
                "clock_file": "timeline/clock.npz",
                "source_quality_file": "timeline/source_quality.npz",
                "source_names": list(SOURCE_NAMES),
            },
            "adam": {
                "physical_31": {
                    "topic": self.adam_topic,
                    "message_type": "sensor_msgs/msg/JointState",
                    "file": "adam/physical_31.npz",
                    "joint_count": ADAM_JOINT_COUNT,
                    "joint_names": adam_names,
                    "fields": {
                        "q": "JointState.position",
                        "dq": "JointState.velocity",
                        "tau": "JointState.effort, Adam tau_est",
                    },
                    "groups": {
                        "upper_body_19_indices": ADAM_UPPER_BODY_INDICES,
                        "upper_body_19_names": [
                            adam_names[idx]
                            for idx in ADAM_UPPER_BODY_INDICES
                            if idx < len(adam_names)
                        ],
                    },
                }
            },
            "sharpa": {
                "joints_44": {
                    "topic": self.sharpa_joint_topic,
                    "message_type": "teleop_interfaces/msg/SharpaJointState",
                    "file": "sharpa/joints_44.npz",
                    "joint_count": SHARPA_JOINT_COUNT,
                    "joint_names": sharpa_names,
                    "fields": {
                        "q": "SharpaJointState.joint_state.position",
                        "dq": "SharpaJointState.joint_state.velocity",
                        "tau": "SharpaJointState.joint_state.effort",
                        "q_cmd": "SharpaJointState.q_cmd",
                        "q_cmd_valid": "SharpaJointState.q_cmd_valid",
                    },
                },
                "tactile": {
                    "plane_order": [
                        {"index": idx, "side": side, "finger": finger}
                        for idx, (side, finger) in enumerate(TACTILE_PLANES)
                    ],
                    "deform": {
                        "topic": self.tactile_deform_topic,
                        "message_type": "teleop_interfaces/msg/TactileDeformImageArray",
                        "data_file": "sharpa/tactile/deform_u8.zst",
                        "index_file": "sharpa/tactile/deform_index.npz",
                        "raw_logical_shape": [-1, *TACTILE_DEFORM_SHAPE],
                        "dtype": "uint8",
                        "compression": "zstd",
                    },
                    "force6d": {
                        "topic": self.tactile_force6d_topic,
                        "message_type": "teleop_interfaces/msg/TactileForce6DArray",
                        "file": "sharpa/tactile/force6d.npz",
                        "force_shape": [-1, len(TACTILE_PLANES), 3],
                        "torque_shape": [-1, len(TACTILE_PLANES), 3],
                        "dtype": "float32",
                    },
                    "contact_points": {
                        "topic": self.tactile_contact_topic,
                        "message_type": "teleop_interfaces/msg/TactileContactPointsArray",
                        "file": "sharpa/tactile/contact_points.npz",
                        "points_layout": "points_xyz[offset[row,channel]:offset[row,channel]+count[row,channel]]",
                        "dtype": "float32",
                    },
                },
            },
            "zed": {
                "rgb_file": "zed/rgb.mp4",
                "meta_file": "zed/rgb_meta.json",
                "source_status_topic": self.zed_status_topic,
            },
            "quest": {
                "webvr_status": {
                    "topic": self.quest_webvr_status_topic,
                    "message_type": "std_msgs/msg/String JSON",
                    "file": "quest/webvr_status.jsonl",
                },
                "retarget_status": {
                    "topic": self.quest_retarget_status_topic,
                    "message_type": "std_msgs/msg/String JSON",
                    "file": "quest/retarget_status.jsonl",
                },
            },
        }
        _write_json(session.partial_dir / "schema.json", schema)

    def _write_timeline_arrays(self, session: RecordingSession) -> None:
        np.savez_compressed(
            session.partial_dir / "timeline" / "clock.npz",
            time_unix_ns=_array_1d(session.time_unix_ns, np.int64),
            elapsed_ns=_array_1d(session.elapsed_ns, np.int64),
            sample_rate_hz=np.asarray(self.sample_rate_hz, dtype=np.float32),
        )
        np.savez_compressed(
            session.partial_dir / "timeline" / "source_quality.npz",
            source_names=np.asarray(SOURCE_NAMES),
            valid=_array_rows(session.source_valid, (len(SOURCE_NAMES),), np.bool_),
            age_ms=_array_rows(session.source_age_ms, (len(SOURCE_NAMES),), np.float32),
            count=_array_rows(session.source_count, (len(SOURCE_NAMES),), np.int64),
        )

    def _write_adam_arrays(self, session: RecordingSession) -> None:
        np.savez_compressed(
            session.partial_dir / "adam" / "physical_31.npz",
            q=_array_rows(session.adam_q, (ADAM_JOINT_COUNT,), np.float32),
            dq=_array_rows(session.adam_dq, (ADAM_JOINT_COUNT,), np.float32),
            tau=_array_rows(session.adam_tau, (ADAM_JOINT_COUNT,), np.float32),
            valid=_array_1d(session.adam_valid, np.bool_),
            age_ms=_array_1d(session.adam_age_ms, np.float32),
            source_count=_array_1d(session.adam_source_count, np.int64),
            header_stamp_ns=_array_1d(session.adam_header_stamp_ns, np.int64),
        )

    def _write_sharpa_arrays(self, session: RecordingSession) -> None:
        np.savez_compressed(
            session.partial_dir / "sharpa" / "joints_44.npz",
            q=_array_rows(session.sharpa_q, (SHARPA_JOINT_COUNT,), np.float32),
            dq=_array_rows(session.sharpa_dq, (SHARPA_JOINT_COUNT,), np.float32),
            tau=_array_rows(session.sharpa_tau, (SHARPA_JOINT_COUNT,), np.float32),
            q_cmd=_array_rows(session.sharpa_q_cmd, (SHARPA_JOINT_COUNT,), np.float32),
            q_cmd_valid=_array_1d(session.sharpa_q_cmd_valid, np.bool_),
            valid=_array_1d(session.sharpa_valid, np.bool_),
            age_ms=_array_1d(session.sharpa_age_ms, np.float32),
            source_count=_array_1d(session.sharpa_source_count, np.int64),
            header_stamp_ns=_array_1d(session.sharpa_header_stamp_ns, np.int64),
        )

    def _write_tactile_arrays(self, session: RecordingSession) -> None:
        tactile_dir = session.partial_dir / "sharpa" / "tactile"
        np.savez_compressed(
            tactile_dir / "force6d.npz",
            force=_array_rows(session.tactile_force, (len(TACTILE_PLANES), 3), np.float32),
            torque=_array_rows(session.tactile_torque, (len(TACTILE_PLANES), 3), np.float32),
            frame_id=_array_rows(session.tactile_force_frame_id, (len(TACTILE_PLANES),), np.uint32),
            sensor_time=_array_rows(session.tactile_force_sensor_time, (len(TACTILE_PLANES),), np.float64),
            valid=_array_rows(session.tactile_force_valid, (len(TACTILE_PLANES),), np.bool_),
        )
        np.savez_compressed(
            tactile_dir / "contact_points.npz",
            points_xyz=_array_rows(session.tactile_contact_points_xyz, (3,), np.float32),
            offset=_array_rows(session.tactile_contact_offset, (len(TACTILE_PLANES),), np.int64),
            count=_array_rows(session.tactile_contact_count, (len(TACTILE_PLANES),), np.uint16),
            frame_id=_array_rows(session.tactile_contact_frame_id, (len(TACTILE_PLANES),), np.uint32),
            sensor_time=_array_rows(session.tactile_contact_sensor_time, (len(TACTILE_PLANES),), np.float64),
            valid=_array_rows(session.tactile_contact_valid, (len(TACTILE_PLANES),), np.bool_),
        )
        np.savez_compressed(
            tactile_dir / "deform_index.npz",
            time_unix_ns=_array_1d(session.tactile_deform_time_unix_ns, np.int64),
            elapsed_ns=_array_1d(session.tactile_deform_elapsed_ns, np.int64),
            timeline_row=_array_1d(session.tactile_deform_timeline_row, np.int32),
            raw_frame_index=_array_1d(session.tactile_deform_raw_frame_index, np.int32),
            frame_id=_array_rows(session.tactile_deform_frame_id, (len(TACTILE_PLANES),), np.uint32),
            sensor_time=_array_rows(session.tactile_deform_sensor_time, (len(TACTILE_PLANES),), np.float64),
            valid=_array_rows(session.tactile_deform_valid, (len(TACTILE_PLANES),), np.bool_),
        )

    def _write_status_events(self, session: RecordingSession) -> None:
        _write_jsonl(session.partial_dir / "status" / "events.jsonl", session.status_events)

    def _write_quest_events(self, session: RecordingSession) -> None:
        _write_jsonl(
            session.partial_dir / "quest" / "webvr_status.jsonl",
            session.quest_webvr_events,
        )
        _write_jsonl(
            session.partial_dir / "quest" / "retarget_status.jsonl",
            session.quest_retarget_events,
        )

    def _write_zed_meta(self, session: RecordingSession) -> None:
        video_path = session.partial_dir / "zed" / "rgb.mp4"
        payload = {
            "owner": "monitor",
            "zed_node_scope": "source_only",
            "record_zed_video": self.record_zed_video,
            "source": self._zed_video_source(),
            "ffmpeg": self.zed_video_ffmpeg,
            "started": session.zed_video_started,
            "ok": video_path.is_file() and not session.zed_video_error,
            "error": session.zed_video_error,
            "path": "rgb.mp4",
            "sdp_path": "rtp.sdp",
            "log_path": "ffmpeg.log",
            "bytes": video_path.stat().st_size if video_path.is_file() else 0,
        }
        _write_json(session.partial_dir / "zed" / "rgb_meta.json", payload)

    def _manifest(
        self,
        session: RecordingSession,
        *,
        raw_deleted: bool,
        raw_kept: bool,
        quality_report: dict[str, Any],
    ) -> dict[str, Any]:
        ended_mono = session.ended_mono if session.ended_mono is not None else time.monotonic()
        return {
            "schema_version": 2,
            "format": "pnd_local_monitor_columnar",
            "sample": session.sample_name,
            "final_directory": session.final_dir.name,
            "state": "complete" if quality_report.get("ok") else "complete_error",
            "quality": quality_report,
            "started_unix_ns": session.started_unix_ns,
            "ended_unix_ns": session.ended_unix_ns,
            "duration_s": round(ended_mono - session.started_mono, 6),
            "sample_rate_hz": self.sample_rate_hz,
            "topics": {
                "status_json": self.status_json_topic,
                "adam_physical": self.adam_topic,
                "sharpa_joint": self.sharpa_joint_topic,
                "tactile_deform": self.tactile_deform_topic,
                "tactile_force6d": self.tactile_force6d_topic,
                "tactile_contact": self.tactile_contact_topic,
                "zed_status": self.zed_status_topic,
                "quest_webvr_status": self.quest_webvr_status_topic,
                "quest_retarget_status": self.quest_retarget_status_topic,
            },
            "counts": {
                "timeline_rows": session.timeline_rows,
                "status_events": len(session.status_events),
                "quest_webvr_events": len(session.quest_webvr_events),
                "quest_retarget_events": len(session.quest_retarget_events),
                "tactile_deform_frames": session.deform_frames,
                "tactile_deform_dropped": session.deform_dropped,
                "contact_points_xyz": len(session.tactile_contact_points_xyz),
            },
            "tactile_deform": {
                "dtype": "uint8",
                "shape": list(TACTILE_DEFORM_SHAPE),
                "plane_order": [
                    {"index": idx, "side": side, "finger": finger}
                    for idx, (side, finger) in enumerate(TACTILE_PLANES)
                ],
                "raw_tmp_path": "sharpa/tactile/deform_u8.raw.tmp",
                "zstd_path": "sharpa/tactile/deform_u8.zst",
                "zstd_level": self.zstd_level,
                "raw_deleted": raw_deleted,
                "raw_kept": raw_kept,
            },
            "tactile_quality": self._tactile_quality_summary(session),
            "teleop_interfaces_available": not bool(TELEOP_INTERFACES_IMPORT_ERROR),
            "teleop_interfaces_import_error": TELEOP_INTERFACES_IMPORT_ERROR,
            "zed_video": {
                "owner": "monitor",
                "zed_node_scope": "source_only",
                "record_zed_video": self.record_zed_video,
                "source": self._zed_video_source(),
                "ffmpeg": self.zed_video_ffmpeg,
                "started": session.zed_video_started,
                "ok": (session.partial_dir / "zed" / "rgb.mp4").is_file()
                and not session.zed_video_error,
                "error": session.zed_video_error,
                "path": "zed/rgb.mp4",
                "meta_path": "zed/rgb_meta.json",
            },
            "start_status": session.start_status,
            "stop_status": session.stop_status,
            "stop_reason": session.stop_reason,
            "files": {
                "schema": "schema.json",
                "timeline_clock": "timeline/clock.npz",
                "timeline_source_quality": "timeline/source_quality.npz",
                "adam_physical_31": "adam/physical_31.npz",
                "sharpa_joints_44": "sharpa/joints_44.npz",
                "tactile_force6d": "sharpa/tactile/force6d.npz",
                "tactile_contact_points": "sharpa/tactile/contact_points.npz",
                "tactile_deform_index": "sharpa/tactile/deform_index.npz",
                "tactile_deform_zstd": "sharpa/tactile/deform_u8.zst",
                "rgb_video": "zed/rgb.mp4",
                "rgb_video_meta": "zed/rgb_meta.json",
                "status_events": "status/events.jsonl",
                "quest_webvr_status": "quest/webvr_status.jsonl",
                "quest_retarget_status": "quest/retarget_status.jsonl",
                "manifest": "manifest.json",
                "checksums": "checksums.sha256",
                "complete": "COMPLETE",
                "error_marker": "ERROR" if not quality_report.get("ok") else "",
                "recording_errors_json": "RECORDING_ERRORS.json"
                if not quality_report.get("ok")
                else "",
                "recording_errors_txt": "RECORDING_ERRORS.txt"
                if not quality_report.get("ok")
                else "",
            },
        }

    def _tactile_quality_summary(self, session: RecordingSession) -> dict[str, Any]:
        return {
            "plane_order": [
                {"index": idx, "side": side, "finger": finger}
                for idx, (side, finger) in enumerate(TACTILE_PLANES)
            ],
            "force6d": self._tactile_series_quality(
                session.tactile_force_frame_id,
                session.tactile_force_sensor_time,
                session.tactile_force_valid,
            ),
            "contact_points": self._tactile_series_quality(
                session.tactile_contact_frame_id,
                session.tactile_contact_sensor_time,
                session.tactile_contact_valid,
            ),
            "deform": self._tactile_series_quality(
                session.tactile_deform_frame_id,
                session.tactile_deform_sensor_time,
                session.tactile_deform_valid,
            ),
        }

    @staticmethod
    def _tactile_series_quality(
        frame_id_rows: list[list[int]],
        sensor_time_rows: list[list[float]],
        valid_rows: list[list[bool]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        row_count = max(len(frame_id_rows), len(sensor_time_rows), len(valid_rows))
        for idx, (side, finger) in enumerate(TACTILE_PLANES):
            frames: list[int] = []
            sensor_times: list[float] = []
            valid_count = 0
            for row_idx in range(row_count):
                valid = (
                    row_idx < len(valid_rows)
                    and idx < len(valid_rows[row_idx])
                    and bool(valid_rows[row_idx][idx])
                )
                frame_id = (
                    int(frame_id_rows[row_idx][idx])
                    if row_idx < len(frame_id_rows)
                    and idx < len(frame_id_rows[row_idx])
                    else 0
                )
                sensor_time = (
                    float(sensor_time_rows[row_idx][idx])
                    if row_idx < len(sensor_time_rows)
                    and idx < len(sensor_time_rows[row_idx])
                    else math.nan
                )
                if valid:
                    valid_count += 1
                    if frame_id > 0:
                        frames.append(frame_id)
                    if math.isfinite(sensor_time) and sensor_time > 0.0:
                        sensor_times.append(sensor_time)
            unique_frame_id = len(set(frames))
            sensor_time_span_s = (
                max(sensor_times) - min(sensor_times)
                if len(sensor_times) > 1
                else 0.0
            )
            output.append(
                {
                    "index": idx,
                    "side": side,
                    "finger": finger,
                    "rows": row_count,
                    "valid_count": valid_count,
                    "valid_ratio": round(valid_count / row_count, 6)
                    if row_count
                    else 0.0,
                    "unique_frame_id": unique_frame_id,
                    "sensor_time_span_s": round(sensor_time_span_s, 6),
                    "first_frame_id": min(frames) if frames else 0,
                    "last_frame_id": max(frames) if frames else 0,
                    "first_sensor_time": min(sensor_times) if sensor_times else None,
                    "last_sensor_time": max(sensor_times) if sensor_times else None,
                }
            )
        return output

    @staticmethod
    def _latest_joint_names(latest: Latest, expected_count: int) -> list[str]:
        payload = latest.payload or {}
        names = payload.get("name", [])
        if not isinstance(names, list):
            return []
        return [str(name) for name in names[:expected_count]]

    def _write_checksums(self, root: Path) -> None:
        checksum_path = root / "checksums.sha256"
        rows = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name in {"checksums.sha256", "COMPLETE"}:
                continue
            digest = self._sha256_file(path)
            rows.append(f"{digest}  {path.relative_to(root).as_posix()}\n")
        checksum_path.write_text("".join(rows), encoding="utf-8")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def destroy_node(self) -> bool:
        with self._lock:
            active = self._active
        if active is not None:
            self._stop_recording(self._status.payload or {}, "node_shutdown")
        for thread in list(self._finalizers):
            thread.join()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RecordingMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
