"""Episode recorder ROS node for the dual UR5 + SharpA contract."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .episode_writer import EpisodeWriterCompat

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from ur_common.camera_transport import SharedCameraReader
    from ur_msgs.msg import (
        CameraSet,
        DeformationFrame,
        HandStateFrame,
        TauFrame,
        UrStateFrame,
        WrenchFrame,
    )

    ROS_AVAILABLE = True
except Exception:  # pragma: no cover
    ROS_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]


FINGERS = ("pinky", "ring", "middle", "index", "thumb")
CAMERA_NAMES = ("ego", "left_wrist", "right_wrist")
MODALITY_NAMES = ("camera", "ur", "hand", "tau", "wrench", "deform")
TELEOP_HAND_SOURCE = 2  # HandCommand.SOURCE_VIVE; Manus teleop source id.


def _fk(q44: Any) -> tuple[list[float], list[float]]:
    from ur_common.kinematics import UrCommonKinematics

    if not hasattr(_fk, "kin"):
        _fk.kin = UrCommonKinematics()  # type: ignore[attr-defined]
    positions = _fk.kin.hand_fingertip_positions(q44)  # type: ignore[attr-defined]
    return (
        positions["left"].reshape(-1).tolist(),
        positions["right"].reshape(-1).tolist(),
    )


def _vec(value: Any, size: int) -> list[float]:
    vals = list(value) if value is not None else []
    if len(vals) != size:
        raise ValueError(f"expected vector length {size}, got {len(vals)}")
    result = [float(x) for x in vals]
    if np is not None and not np.isfinite(result).all():
        raise ValueError("vector contains non-finite values")
    return result


def _stamp_ns(msg: Any, field: str = "timestamp_ns") -> int:
    return int(getattr(msg, field, 0))


@dataclass
class _Latest:
    msg: Any = None
    received: float = 0.0


class RecorderCore:
    """ROS-independent synchronizer; callbacks only update latest samples."""

    def __init__(
        self,
        root: str | Path,
        frequency: float = 30.0,
        camera_max_age_s: float = 0.25,
        video_ffmpeg: str = "ffmpeg",
    ) -> None:
        self.writer = EpisodeWriterCompat(root, frequency, ffmpeg=video_ffmpeg)
        self.frequency = float(frequency)
        self.camera_max_age_s = float(camera_max_age_s)
        self.recording = False
        self.started = time.monotonic()
        self.stopped = 0.0
        self.validation: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self.latest = {key: _Latest() for key in MODALITY_NAMES}
        self._reset_episode_counters()

    def _reset_episode_counters(self) -> None:
        self.frames = 0
        self.attempts = 0
        self.dropped = 0
        self.deformation_dropped = 0
        self.camera_dropped = {name: 0 for name in CAMERA_NAMES}
        self.last_error = ""
        self.quality_errors: list[str] = []
        self._last_video_timestamps = {name: 0 for name in CAMERA_NAMES}

    def update(self, key: str, msg: Any, received: float | None = None) -> Any:
        if key not in self.latest:
            raise KeyError(f"unknown recording modality {key!r}")
        if key == "camera":
            return self.update_camera(msg, received)
        if key == "deform":
            return self.update_deformation(msg, received)
        timestamp = time.time() if received is None else float(received)
        with self._lock:
            self.latest[key] = _Latest(msg, timestamp)
        return None

    def update_camera(self, msg: Any, received: float | None = None) -> bool:
        received = time.time() if received is None else float(received)
        with self._lock:
            self.latest["camera"] = _Latest(msg, received)
            if not self.recording:
                return True
            return self._record_camera_set(msg, received)

    def _record_camera_set(self, msg: Any, received: float) -> bool:
        accepted = True
        timeline_row = max(0, self.attempts - 1)
        for name in CAMERA_NAMES:
            valid = bool(getattr(msg, f"{name}_valid", False))
            timestamp = int(getattr(msg, f"{name}_timestamp_ns", 0))
            payload = getattr(msg, name, b"")
            previous = self._last_video_timestamps[name]
            if not valid or timestamp <= 0 or timestamp == previous:
                continue
            if previous and timestamp < previous:
                self.camera_dropped[name] += 1
                self._note_quality_error(f"{name} camera timestamp moved backwards")
                accepted = False
                continue
            if not self.writer.add_camera_frame(
                name,
                payload,
                timestamp,
                timeline_row,
                int(received * 1e9),
            ):
                self.camera_dropped[name] += 1
                self._note_quality_error(f"{name} camera writer queue full")
                accepted = False
                continue
            self._last_video_timestamps[name] = timestamp
        return accepted

    def update_deformation(self, msg: Any, received: float | None = None) -> bool:
        received = time.time() if received is None else float(received)
        with self._lock:
            self.latest["deform"] = _Latest(msg, received)
            if not self.recording:
                return True
            timeline_row = max(0, self.attempts - 1)
            accepted = self.writer.add_deformation(
                msg,
                timeline_row,
                int(received * 1e9),
            )
            if not accepted:
                self.deformation_dropped += 1
                self._note_quality_error("deformation writer queue full")
            return accepted

    def start(self) -> bool:
        with self._lock:
            if self.recording:
                return False
            self.validation = None
            self.writer.start()
            self.recording = True
            self.started = time.monotonic()
            self.stopped = 0.0
            self._reset_episode_counters()
            camera = self.latest["camera"]
            if camera.msg is not None:
                self._record_camera_set(camera.msg, time.time())
            return True

    def stop(self, save: bool = True) -> Path | None:
        with self._lock:
            self.recording = False
            self.stopped = time.monotonic()
        result = self.writer.stop(
            save=save,
            dropped_frames=self.dropped,
            quality_errors=self.quality_errors,
        )
        with self._lock:
            self.validation = self.writer.last_validation if save else None
        return result
    def classify(self, category: str) -> Path:
        with self._lock:
            if self.recording:
                raise RuntimeError("cannot classify while recording")
            destination = self.writer.classify(category)
            self.validation = self.writer.last_validation
            self.last_error = ""
            return destination

    def drop_last_saved(self) -> Path:
        with self._lock:
            if self.recording:
                raise RuntimeError("cannot drop a sample while recording")
            removed = self.writer.delete_last_saved()
            self.validation = None
            self.last_error = ""
            return removed

    def _fresh(self, latest: _Latest, now: float) -> bool:
        return (
            latest.msg is not None
            and now - latest.received <= self.camera_max_age_s
        )

    def _note_quality_error(self, message: str) -> None:
        self.last_error = str(message)
        if self.last_error not in self.quality_errors:
            self.quality_errors.append(self.last_error)

    @staticmethod
    def _finger_valid(msg: Any, side: str) -> list[bool]:
        values = list(getattr(msg, f"{side}_valid", [])) if msg is not None else []
        return [bool(value) for value in values] if len(values) == 5 else [False] * 5

    def _tactile_quality_errors(self) -> list[str]:
        errors: list[str] = []
        for key, label in (("wrench", "wrench"), ("deform", "deformation")):
            msg = self.latest[key].msg
            missing: list[str] = []
            for side in ("left", "right"):
                valid = self._finger_valid(msg, side)
                missing.extend(
                    f"{side}_{finger}"
                    for finger, ok in zip(FINGERS, valid)
                    if not ok
                )
            if missing:
                errors.append(f"SharpA {label} invalid: {', '.join(missing)}")
        return errors

    def _drop_sample(self, reason: str) -> bool:
        self.dropped += 1
        self.last_error = str(reason)
        self.quality_errors.append(self.last_error)
        return False

    def sample(self, now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        with self._lock:
            if not self.recording:
                return False
            missing = [
                key
                for key in MODALITY_NAMES
                if not self._fresh(self.latest[key], now)
            ]
            self.writer.add_raw_row(self._build_raw_row(now))
            timeline_index = self.attempts
            self.attempts += 1
            if missing:
                return self._drop_sample(
                    "missing/stale modalities: " + ", ".join(missing)
                )
            for error in self._tactile_quality_errors():
                self._note_quality_error(error)
            try:
                item = self._build_item(now)
            except Exception as exc:
                return self._drop_sample(str(exc))
            item["metadata"]["timeline_index"] = timeline_index
            if not self.writer.add_item(item):
                return self._drop_sample("writer queue full")
            self.frames += 1
            return True

    @staticmethod
    def _raw_values(msg: Any, field: str, size: int, valid: bool) -> list[float]:
        if not valid or msg is None:
            return [float("nan")] * size
        try:
            return _vec(getattr(msg, field), size)
        except Exception:
            return [float("nan")] * size

    def _build_raw_row(self, now: float) -> dict[str, Any]:
        source_fresh = {
            key: self._fresh(self.latest[key], now)
            for key in MODALITY_NAMES
        }
        source_valid = dict(source_fresh)
        cam = self.latest["camera"].msg
        ur = self.latest["ur"].msg
        hand = self.latest["hand"].msg
        tau = self.latest["tau"].msg
        wrench = self.latest["wrench"].msg
        deform = self.latest["deform"].msg

        if source_valid["camera"]:
            source_valid["camera"] = bool(
                cam.ego_valid and cam.left_wrist_valid and cam.right_wrist_valid
            )
        if source_valid["ur"]:
            source_valid["ur"] = bool(getattr(ur, "valid", True))
        if source_valid["hand"]:
            source_valid["hand"] = bool(hand.left_valid and hand.right_valid)
        if source_valid["tau"]:
            source_valid["tau"] = bool(tau.left_valid and tau.right_valid)

        wrench_valid = [
            self._finger_valid(wrench, side)
            if source_fresh["wrench"] else [False] * 5
            for side in ("left", "right")
        ]
        deformation_valid = [
            self._finger_valid(deform, side)
            if source_fresh["deform"] else [False] * 5
            for side in ("left", "right")
        ]
        source_valid["wrench"] = bool(
            source_fresh["wrench"]
            and all(value for side in wrench_valid for value in side)
        )
        source_valid["deform"] = bool(
            source_fresh["deform"]
            and all(value for side in deformation_valid for value in side)
            and len(deform.left) == 5 * 240 * 240
            and len(deform.right) == 5 * 240 * 240
        )
        ages = [
            float("inf")
            if self.latest[key].msg is None
            else max(0.0, (now - self.latest[key].received) * 1000.0)
            for key in MODALITY_NAMES
        ]
        values = self._raw_values
        return {
            "time_unix_ns": int(now * 1e9),
            "source_names": MODALITY_NAMES,
            "source_valid": [source_valid[key] for key in MODALITY_NAMES],
            "source_age_ms": ages,
            "source_timestamp_ns": [
                max(
                    int(getattr(cam, f"{name}_timestamp_ns", 0))
                    for name in CAMERA_NAMES
                ) if cam is not None else 0,
                *[
                    0 if msg is None else _stamp_ns(msg)
                    for msg in (ur, hand, tau, wrench, deform)
                ],
            ],
            "ur_q": [
                *values(ur, "left_joint", 6, source_valid["ur"]),
                *values(ur, "right_joint", 6, source_valid["ur"]),
            ],
            "ur_tcp": [
                values(
                    ur,
                    "left_tcp_pose6",
                    6,
                    source_valid["ur"] and hasattr(ur, "left_tcp_pose6"),
                ),
                values(
                    ur,
                    "right_tcp_pose6",
                    6,
                    source_valid["ur"] and hasattr(ur, "right_tcp_pose6"),
                ),
            ],
            "ur_wrist": [
                values(ur, "left_eef", 9, source_valid["ur"]),
                values(ur, "right_eef", 9, source_valid["ur"]),
            ],
            "hand_q": values(hand, "q", 44, source_valid["hand"]),
            "hand_q_cmd": values(hand, "q_cmd", 44, source_valid["hand"]),
            "hand_q_cmd_valid": bool(
                source_valid["hand"]
                and getattr(hand, "q_cmd_valid", False)
                and int(getattr(hand, "q_cmd_source", 0)) == TELEOP_HAND_SOURCE
            ),
            "hand_q_cmd_timestamp_ns": (
                int(getattr(hand, "q_cmd_timestamp_ns", 0))
                if hand is not None else 0
            ),
            "hand_q_cmd_source": (
                int(getattr(hand, "q_cmd_source", 0))
                if hand is not None else 0
            ),
            "hand_tau": [
                *values(tau, "left", 22, source_valid["tau"]),
                *values(tau, "right", 22, source_valid["tau"]),
            ],
            "wrench": [
                [
                    values(wrench, side, 30, source_fresh["wrench"])[
                        index * 6:(index + 1) * 6
                    ]
                    for index in range(5)
                ]
                for side in ("left", "right")
            ],
            "wrench_valid": wrench_valid,
            "deformation_valid": deformation_valid,
            "camera_timestamp_ns": [
                int(getattr(cam, f"{name}_timestamp_ns", 0))
                if source_valid["camera"] else 0
                for name in CAMERA_NAMES
            ],
        }

    def _build_item(self, now: float) -> dict[str, Any]:
        cam = self.latest["camera"].msg
        ur = self.latest["ur"].msg
        hand = self.latest["hand"].msg
        tau = self.latest["tau"].msg
        wrench = self.latest["wrench"].msg
        deform = self.latest["deform"].msg

        if str(getattr(cam, "encoding", "jpeg")) != "jpeg":
            raise ValueError("camera encoding must be jpeg")
        if not bool(getattr(ur, "valid", True)):
            raise ValueError("UR state invalid")
        if (
            not bool(getattr(hand, "left_valid", True))
            or not bool(getattr(hand, "right_valid", True))
        ):
            raise ValueError("SharpA hand state invalid")
        if not bool(getattr(hand, "q_cmd_valid", False)):
            raise ValueError("SharpA hand action has no complete successful SDK command")
        if int(getattr(hand, "q_cmd_source", 0)) != TELEOP_HAND_SOURCE:
            raise ValueError("SharpA hand action is not from Manus teleop")
        q_cmd_timestamp_ns = int(getattr(hand, "q_cmd_timestamp_ns", 0))
        if q_cmd_timestamp_ns <= 0:
            raise ValueError("SharpA hand action timestamp is invalid")
        hand_state_timestamp_ns = int(getattr(hand, "timestamp_ns", 0))
        if q_cmd_timestamp_ns > hand_state_timestamp_ns:
            raise ValueError("SharpA hand action is newer than its paired hand state")
        if (
            not bool(getattr(tau, "left_valid", True))
            or not bool(getattr(tau, "right_valid", True))
        ):
            raise ValueError("SharpA tau invalid")
        if (
            len(getattr(deform, "left", [])) != 5 * 240 * 240
            or len(getattr(deform, "right", [])) != 5 * 240 * 240
        ):
            raise ValueError("SharpA deformation shape invalid")
        if not (
            cam.ego_valid
            and cam.left_wrist_valid
            and cam.right_wrist_valid
        ):
            raise ValueError("camera view flagged invalid")

        q44 = _vec(hand.q, 44)
        q_cmd44 = _vec(hand.q_cmd, 44)
        left_tip, right_tip = _fk(q44)
        colors = {
            "color_0": "cameras/ego/rgb.mp4",
            "color_1": "cameras/left_wrist/rgb.mp4",
            "color_2": "cameras/right_wrist/rgb.mp4",
        }
        states = {
            "left_arm": {"qpos": _vec(ur.left_joint, 6)},
            "right_arm": {"qpos": _vec(ur.right_joint, 6)},
            "left_arm_pose": {"pose": _vec(ur.left_eef, 9)},
            "right_arm_pose": {"pose": _vec(ur.right_eef, 9)},
            "left_hand": {"qpos": q44[:22]},
            "right_hand": {"qpos": q44[22:]},
            "left_hand_positions": {"positions": left_tip},
            "right_hand_positions": {"positions": right_tip},
            "left_hand_torques": {"torques": _vec(tau.left, 22)},
            "right_hand_torques": {"torques": _vec(tau.right, 22)},
            "tactile": {
                "left_tactile": self._wrench_dict(wrench.left),
                "right_tactile": self._wrench_dict(wrench.right),
            },
        }
        if hasattr(ur, "left_tcp_pose6") and hasattr(ur, "right_tcp_pose6"):
            states["left_arm"]["tcp_pose6"] = _vec(ur.left_tcp_pose6, 6)
            states["right_arm"]["tcp_pose6"] = _vec(ur.right_tcp_pose6, 6)
        # q and q_cmd come from one HandStateFrame: measured feedback plus the
        # most recent complete Manus target successfully handed to the SDK.
        # Vive arm commands are intentionally not recorded.
        actions = {
            "left_hand": {"qpos": q_cmd44[:22]},
            "right_hand": {"qpos": q_cmd44[22:]},
        }
        stamps = {
            "frame_start": now,
            "wall_clock": now,
            "wrist": now,
            "head_image": int(cam.ego_timestamp_ns) / 1e9,
            "left_image": int(cam.left_wrist_timestamp_ns) / 1e9,
            "right_image": int(cam.right_wrist_timestamp_ns) / 1e9,
            "arm": int(getattr(ur, "timestamp_ns", 0)) / 1e9,
            "hand_state": hand_state_timestamp_ns / 1e9,
            "hand_action": q_cmd_timestamp_ns / 1e9,
            "tactile": int(getattr(tau, "timestamp_ns", 0)) / 1e9,
            "frame_end": time.time(),
        }
        validity = {
            kind: {
                side: self._finger_valid(msg, side)
                for side in ("left", "right")
            }
            for kind, msg in (("wrench", wrench), ("deformation", deform))
        }
        return {
            "colors": colors,
            "depths": {},
            "states": states,
            "actions": actions,
            "tactiles": None,
            "audios": None,
            "tactile_images": None,
            "tactile_images_raw": {},
            "tactile_images_deform": {},
            "metadata": {
                "frame_count": self.frames,
                "stamps": stamps,
                "hand_action_source": int(hand.q_cmd_source),
                "tactile_valid": validity,
            },
        }

    @staticmethod
    def _wrench_dict(values: Any) -> dict[str, list[float]]:
        vector = _vec(values, 30)
        return {
            finger: vector[index * 6:(index + 1) * 6]
            for index, finger in enumerate(FINGERS)
        }

    def status(self) -> dict[str, Any]:
        end = (
            time.monotonic()
            if self.recording or not self.stopped
            else self.stopped
        )
        elapsed = max(0.0, end - self.started)
        return {
            "recording": self.recording,
            "episode_id": self.writer.episode_id,
            "frame_count": self.frames,
            "timeline_rows": self.attempts,
            "duration_s": round(elapsed, 3),
            "achieved_rate_hz": round(self.frames / max(1e-9, elapsed), 3),
            "dropped_frames": self.dropped,
            "deformation_dropped": self.deformation_dropped,
            "camera_frames": self.writer.camera_frame_counts(),
            "camera_dropped": dict(self.camera_dropped),
            "last_error": self.last_error,
            "recording_root": str(self.writer.root),
            "validation": self.validation,
        }


if ROS_AVAILABLE:
    class RecordingNode(Node):
        def __init__(self) -> None:
            super().__init__("recording_node")
            self.declare_parameter(
                "recording_root", "/media/yns/T9/0824_yns_unscrew_cap"
            )
            self.declare_parameter("frequency", 30.0)
            self.declare_parameter("camera_max_age_s", 0.25)
            self.declare_parameter("shared_memory_prefix", "ur_sharpa_camera_v1")
            self.declare_parameter("video_ffmpeg", "ffmpeg")

            frequency = float(self.get_parameter("frequency").value)
            self.core = RecorderCore(
                str(self.get_parameter("recording_root").value),
                frequency,
                float(self.get_parameter("camera_max_age_s").value),
                video_ffmpeg=str(self.get_parameter("video_ffmpeg").value),
            )
            self._saving = False
            self._save_thread = None
            default_qos = QoSProfile(depth=10)
            camera_qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.status_pub = self.create_publisher(
                String,
                "/control/record_status",
                default_qos,
            )
            self.create_subscription(
                String,
                "/control/record_command",
                self._command,
                default_qos,
            )
            self.create_subscription(
                CameraSet,
                "/camera/set",
                self._camera,
                camera_qos,
            )
            self.create_subscription(
                UrStateFrame,
                "/ur/state",
                lambda msg: self.core.update("ur", msg),
                default_qos,
            )
            self.create_subscription(
                HandStateFrame,
                "/sharpa/state",
                lambda msg: self.core.update("hand", msg),
                default_qos,
            )
            self.create_subscription(
                TauFrame,
                "/sharpa/tau",
                lambda msg: self.core.update("tau", msg),
                default_qos,
            )
            self.create_subscription(
                WrenchFrame,
                "/sharpa/wrench",
                lambda msg: self.core.update("wrench", msg),
                default_qos,
            )
            self.create_subscription(
                DeformationFrame,
                "/sharpa/deformation",
                self.core.update_deformation,
                camera_qos,
            )
            self.create_subscription(
                String,
                "/control/record_config",
                self._config,
                default_qos,
            )
            self.create_subscription(
                String,
                "/control/record_label",
                self._label,
                default_qos,
            )
            self.create_timer(1.0, self._publish)

            shared_memory_prefix = str(
                self.get_parameter("shared_memory_prefix").value
            )
            self.camera_reader = SharedCameraReader(shared_memory_prefix)
            self._shared_camera_seen = False
            self._observed_camera_timestamps = None
            self._transport_camera_dropped = {
                name: 0 for name in CAMERA_NAMES
            }
            self.create_timer(
                1.0 / max(30.0, frequency * 4.0),
                self._poll_camera,
            )
            self.create_timer(1.0 / frequency, self.core.sample)

        def _command(self, msg: String) -> None:
            command = msg.data.strip()
            if command == "start":
                if not self._saving:
                    self.core.start()
            elif command == "stop":
                if self.core.recording and not self._saving:
                    self._saving = True
                    # Publish before starting disk finalization so Monitor can
                    # distinguish an accepted click from an idle button.
                    self._publish()
                    self._save_thread = threading.Thread(
                        target=self._save_recording,
                        daemon=True,
                    )
                    self._save_thread.start()
            elif command == "discard":
                if not self._saving:
                    self.core.stop(False)

        def _save_recording(self) -> None:
            try:
                self.core.stop(True)
            except Exception as exc:
                self.core.last_error = f"recording save failed: {exc}"
                self.get_logger().error(self.core.last_error)
            finally:
                self._saving = False
                self._publish()

        def _camera(self, msg: CameraSet) -> None:
            if not self._shared_camera_seen:
                self.core.update_camera(msg)

        def _poll_camera(self) -> None:
            pending = self.camera_reader.read_pending()
            dropped = self.camera_reader.dropped_frames
            for name in CAMERA_NAMES:
                previous = self._transport_camera_dropped[name]
                current = int(dropped.get(name, 0))
                if self.core.recording and current > previous:
                    count = current - previous
                    self.core.camera_dropped[name] += count
                    self.core._note_quality_error(
                        f"{name} camera transport ring overran by {count} frame(s)"
                    )
                self._transport_camera_dropped[name] = current
            if pending:
                self._shared_camera_seen = True
                for msg in pending:
                    self.core.update_camera(msg)
                return
            if self._shared_camera_seen:
                return
            msg = self.camera_reader.read()
            if msg is None:
                return
            self._shared_camera_seen = True
            timestamps = (
                msg.ego_timestamp_ns,
                msg.left_wrist_timestamp_ns,
                msg.right_wrist_timestamp_ns,
            )
            if timestamps == self._observed_camera_timestamps:
                return
            self._observed_camera_timestamps = timestamps
            self.core.update_camera(msg)

        def _config(self, msg: String) -> None:
            if self.core.recording or self._saving:
                return
            try:
                root = json.loads(msg.data).get("recording_root")
                if root:
                    self.core.writer.root = Path(str(root)).expanduser()
            except Exception as exc:
                self.get_logger().warning(f"invalid record config: {exc}")

        def _label(self, msg: String) -> None:
            try:
                if self._saving:
                    raise RuntimeError("cannot classify/drop while saving")
                if msg.data.strip().lower() == "drop":
                    removed = self.core.drop_last_saved()
                    self.get_logger().warning(
                        f"sample permanently deleted: {removed}"
                    )
                else:
                    destination = self.core.classify(msg.data)
                    self.get_logger().info(f"sample classified: {destination}")
            except Exception as exc:
                self.core.last_error = f"sample classification/drop failed: {exc}"
                self.get_logger().warning(self.core.last_error)

        def _publish(self) -> None:
            msg = String()
            status = self.core.status()
            status["saving"] = bool(self._saving)
            msg.data = json.dumps(status, separators=(",", ":"))
            self.status_pub.publish(msg)

        def destroy_node(self) -> Any:
            self.camera_reader.close()
            if self._save_thread is not None and self._save_thread.is_alive():
                self._save_thread.join(timeout=35.0)
            elif self.core.recording:
                self.core.stop(True)
            self.core.writer.close()
            return super().destroy_node()
else:
    class RecordingNode:  # pragma: no cover
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("rclpy is unavailable")


def main(args: list[str] | None = None) -> None:
    if not ROS_AVAILABLE:
        raise RuntimeError("rclpy is unavailable")
    rclpy.init(args=args)
    node = RecordingNode()
    try:
        rclpy.spin(node)
    except (
        KeyboardInterrupt,
        rclpy.executors.ExternalShutdownException,
        RCLError,
    ):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
