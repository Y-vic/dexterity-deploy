"""Episode recorder ROS node for the dual UR5 + SharpA contract."""

from __future__ import annotations

from ur_common.protection_config import setting
import json
from collections import deque
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from .episode_writer import EpisodeWriterCompat
from .video_encoding import check_nvenc
from .rollout import RolloutRecorder
from .execution_audit import archive_actor_trace, write_termination
from .alignment import TimestampAligner
from ur_common.raw_camera import RAW_ENCODING
from ur_common.recording_config import DEFAULT_HIL_RECORDING_ROOT, normalize_recording_root, RecordingPreferences
from ur_common.deformation_transport import DeformationReader, DEFAULT_PREFIX as DEFORMATION_PREFIX

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from std_msgs.msg import String
    from ur_common.camera_transport import SharedCameraReader
    from ur_msgs.msg import (
        CameraSet,
        ControlStatus,
        DeformationFrame,
        HandStateFrame,
        HandCommand,
        ArmCommand,
        HardwareCommandResult,
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
TELEOP_HAND_SOURCE = 2  # HandCommand.SOURCE_VIVE; AE01 teleop source id.
ACTOR_HAND_SOURCE = 3


class MissingHandAction(ValueError):
    pass


_FK_LOCK = threading.Lock()


def _fk(q44: Any) -> tuple[list[float], list[float]]:
    from ur_common.kinematics import UrCommonKinematics

    # Pinocchio's cached Data objects are mutable. Writer workers share the FK
    # facade, so keep initialization and evaluation in the same critical section.
    with _FK_LOCK:
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
    source_timestamp_ns: int = 0


class RecorderCore:
    """ROS-independent synchronizer; callbacks only update latest samples."""

    def __init__(
        self,
        root: str | Path,
        frequency: float = 30.0,
        camera_max_age_s: float | None = None,
        video_ffmpeg: str = "ffmpeg",
        raw_message_budget_bytes: int = 64 * 1024 * 1024,
        record_tactile: bool = True,
        video_backend: str = "mjpeg",
        ego_bitrate_mbps: float = 30.0,
        wrist_bitrate_mbps: float = 12.0,
        rollout_recording: bool = False,
        root_preferences: RecordingPreferences | None = None,
        alignment_wait_frames: float = 0,
    ) -> None:
        self.root_preferences = root_preferences
        self.writer = EpisodeWriterCompat(root, frequency, ffmpeg=video_ffmpeg,
                                         raw_message_budget_bytes=raw_message_budget_bytes,
                                         record_tactile=record_tactile,
                                         prepare_item=self._complete_item,
                                         video_backend=video_backend,
                                         ego_bitrate_mbps=ego_bitrate_mbps,
                                         wrist_bitrate_mbps=wrist_bitrate_mbps)
        self.record_tactile = bool(record_tactile)
        self.rollout_recording = bool(rollout_recording)
        self.hand_source = ACTOR_HAND_SOURCE if self.rollout_recording else TELEOP_HAND_SOURCE
        self.required_modalities = (MODALITY_NAMES if self.record_tactile else
                                    tuple(k for k in MODALITY_NAMES if k not in ("wrench", "deform")))
        self.frequency = float(frequency)
        self.camera_max_age_s = float(setting('recording', 'camera_max_age_s')
                                      if camera_max_age_s is None else camera_max_age_s)
        self.recording = False
        self.started = time.monotonic()
        self.stopped = 0.0
        self.validation: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._inference_condition = threading.Condition(self._lock)
        self.inference_rollout_id = ''
        self._pending_inferences = set()
        self.latest = {key: _Latest() for key in MODALITY_NAMES}
        self.aligner = (TimestampAligner(
            (*CAMERA_NAMES, "ur", "hand", "tau", *(("wrench", "deform") if self.record_tactile else ())),
            frequency, alignment_wait_frames) if alignment_wait_frames else None)
        self._capture_stop_ns = None
        self._diagnostic_history = deque(maxlen=30)
        self._reset_episode_counters()

    def configure_root(self, value: str, *, saving: bool = False, requested: bool = False) -> Path:
        with self._lock:
            if self.recording or self.writer.active or saving or requested:
                raise RuntimeError("录制、启动或保存期间不能修改采集目录")
            root = normalize_recording_root(value)
            root.mkdir(parents=True, exist_ok=True)
            if self.root_preferences is not None:
                self.root_preferences.save(root)
            self.writer.root = root
            return root

    def configure_capture_mode(self, rollout: bool, *, saving: bool = False) -> bool:
        with self._lock:
            if self.recording or self.writer.active or saving:
                return False
            self.rollout_recording = bool(rollout)
            self.hand_source = ACTOR_HAND_SOURCE if rollout else TELEOP_HAND_SOURCE
            return True

    def _reset_episode_counters(self) -> None:
        self.actor_trace_path = ''
        self.audit_rollout_id = ''
        self.execution_audit_required = False
        self.termination = {}
        self.frames = 0
        self.attempts = 0
        self.dropped = 0
        self.deformation_dropped = 0
        self.camera_dropped = {name: 0 for name in CAMERA_NAMES}
        self.last_error = ""
        self.quality_errors: list[str] = []
        self._seen_hand_action = False
        self.boundary_frames: dict[str, int] = {}
        self.capture_notice = ""
        self._last_video_timestamps = {name: 0 for name in CAMERA_NAMES}
        self._last_deformation_timestamp = 0
        self._sample_work_ms = deque(maxlen=300)
        self._sample_overruns = 0

    def update(self, key: str, msg: Any, received: float | None = None) -> Any:
        if key not in self.latest:
            raise KeyError(f"unknown recording modality {key!r}")
        if key == "camera":
            return self.update_camera(msg, received)
        if key == "deform":
            return self.update_deformation(msg, received)
        timestamp = time.time() if received is None else float(received)
        with self._lock:
            self.latest[key] = _Latest(msg, timestamp, _stamp_ns(msg))
            if self.aligner is not None and key in self.aligner.queues:
                self.aligner.push(key, _stamp_ns(msg), round(timestamp * 1e9), self.latest[key])
        return None

    def update_camera(self, msg: Any, received: float | None = None) -> bool:
        received = time.time() if received is None else float(received)
        with self._lock:
            self.latest["camera"] = _Latest(msg, received)
            if self.aligner is not None:
                for name in CAMERA_NAMES:
                    stamp = int(getattr(msg, name + "_timestamp_ns", 0))
                    self.aligner.push(name, stamp, round(received * 1e9),
                                      (bool(getattr(msg, name + "_valid", False)), msg.encoding))
            if not self.recording:
                return True
            return self._record_camera_set(msg, received)

    def _record_camera_set(self, msg: Any, received: float) -> bool:
        if str(getattr(msg, "encoding", "jpeg")) != self.writer.source_encoding:
            self._note_quality_error("camera encoding does not match recording backend")
            return False
        accepted = True
        timeline_row = max(0, self.attempts - 1)
        for name in CAMERA_NAMES:
            valid = bool(getattr(msg, f"{name}_valid", False))
            timestamp = int(getattr(msg, f"{name}_timestamp_ns", 0))
            payload = getattr(msg, name, b"")
            previous = self._last_video_timestamps[name]
            if (self.writer.video_backend == "hevc_nvenc"
                    and timestamp < self.writer._started_unix_ns):
                continue  # Do not encode frames retained from before record-start.
            if self._capture_stop_ns is not None and timestamp > self._capture_stop_ns:
                continue
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
                self._note_quality_error(f"{name} camera writer rejected frame (queue, payload or encoder)")
                accepted = False
                continue
            self._last_video_timestamps[name] = timestamp
        return accepted

    def update_deformation(self, msg: Any, received: float | None = None) -> bool:
        if not self.record_tactile:
            return False
        received = time.time() if received is None else float(received)
        with self._lock:
            previous = self.latest['deform']
            stamp = _stamp_ns(msg)
            if previous.msg is not None and stamp < previous.source_timestamp_ns:
                if self.recording:
                    self._note_quality_error('deformation timestamp moved backwards')
                return False
            if previous.msg is not None and stamp == previous.source_timestamp_ns:
                received = previous.received  # Cached repeats cannot refresh freshness.
            self.latest["deform"] = _Latest(msg, received, stamp)
            if self.aligner is not None and self.record_tactile:
                self.aligner.push("deform", stamp, round(received * 1e9), self.latest["deform"])
            if not self.recording:
                return True
            if self._capture_stop_ns is not None and stamp > self._capture_stop_ns:
                return True
            if 0 < stamp <= self._last_deformation_timestamp:
                return True  # Keep independent source frames, not repeated large payloads.
            timeline_row = max(0, self.attempts - 1)
            accepted = self.writer.add_deformation(
                msg,
                timeline_row,
                int(received * 1e9),
            )
            if not accepted:
                self.deformation_dropped += 1
                self._note_quality_error("deformation writer queue full")
            else:
                self._last_deformation_timestamp = stamp
            return accepted

    def start(self) -> bool:
        with self._lock:
            if self.recording or self.writer.active:
                return False
            self.validation = None
            self.writer.start()
            (self.writer.episode_dir / "raw" / "diagnostic_preroll.json").write_text(
                json.dumps(list(self._diagnostic_history)), encoding="utf-8")
            self.recording = True
            self.started = time.monotonic()
            self.stopped = 0.0
            self._reset_episode_counters()
            self._capture_stop_ns = None
            if self.aligner is not None:
                self._alignment_origin_ns = time.time_ns()
                self._alignment_origin_monotonic_ns = time.monotonic_ns()
                self.aligner.start(self._alignment_origin_ns)
                # A video reference must not point to a frame excluded at start.
                for name in CAMERA_NAMES:
                    self.aligner.last_used[name] = self.writer._started_unix_ns - 1
            self._pending_inferences.clear()
            camera = self.latest["camera"]
            if camera.msg is not None:
                self._record_camera_set(camera.msg, time.time())
            if self.aligner is not None and self.record_tactile:
                deformation = self.latest["deform"]
                if (deformation.msg is not None
                        and self._alignment_origin_ns - self.aligner.skew_ns
                        <= deformation.source_timestamp_ns <= self._alignment_origin_ns):
                    # The first matched hand bundle may predate record-start
                    # by a few ms; persist its already received tactile pixels.
                    self.update_deformation(deformation.msg, received=deformation.received)
            return True

    def record_message(self, topic, type_name, payload, source_timestamp_ns=0,
                       received_unix_ns=None, received_monotonic_ns=None):
        """Preserve a received message before any synchronized-row filtering."""
        with self._lock:
            if not self.recording:
                return False
            accepted = self.writer.message_journal.append(
                topic, type_name, payload, source_timestamp_ns,
                received_unix_ns, received_monotonic_ns)
            if not accepted:
                self._note_quality_error("raw message journal rejected a received message")
            return accepted

    def observe_diagnostic(self, data):
        try:
            payload = json.loads(data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._diagnostic_history.append(payload)

    def record_inference(self, data, payload):
        event = json.loads(data)
        request_id = event['request_id']
        kind = event['event']
        if not isinstance(request_id, str) or not request_id or kind not in (
                'requested', 'completed', 'failed', 'cancelled'):
            raise ValueError('invalid policy inference event')
        with self._inference_condition:
            if not self.inference_rollout_id or event.get('rollout_id') != self.inference_rollout_id:
                return False
            if not self.recording and request_id not in self._pending_inferences:
                return False
            if kind == 'requested':
                self._pending_inferences.add(request_id)
            elif request_id not in self._pending_inferences:
                self.inference_error('policy inference result without request: ' + request_id)
            accepted = self.writer.message_journal.append(
                '/policy/inference', 'std_msgs/msg/String', payload,
                event.get('completed_unix_ns', event.get('submitted_unix_ns', 0)),
                inference_json=data)
            if not accepted:
                self.inference_error('policy inference journal rejected an event')
            if kind != 'requested':
                self._pending_inferences.discard(request_id)
                self._inference_condition.notify_all()
            return accepted

    def inference_error(self, reason):
        with self._lock:
            self._note_quality_error(reason)
            self.writer.message_journal.fail(reason)

    def stop(self, save: bool = True) -> Path | None:
        with self._inference_condition:
            if self.recording and self.aligner is not None and save:
                self._capture_stop_ns = self._alignment_now_ns()
                deadline = time.monotonic_ns() + self.aligner.wait_ns
                while self.aligner.next_target_ns <= self._capture_stop_ns:
                    expired = time.monotonic_ns() >= deadline
                    self.flush_aligned(force=expired)
                    if self.aligner.next_target_ns > self._capture_stop_ns:
                        break
                    # ROS subscriptions and media drain remain live during stop.
                    self._inference_condition.wait(.001)
            self.recording = False
            self.stopped = time.monotonic()
            if self._pending_inferences and not self._inference_condition.wait_for(
                    lambda: not self._pending_inferences, timeout=12.0):
                self.inference_error('policy inference drain timed out: ' +
                                     ', '.join(sorted(self._pending_inferences)))
            rollout_id = self.inference_rollout_id or self.audit_rollout_id
            self.audit_rollout_id = rollout_id
            self.inference_rollout_id = ''
        if save and self.writer.active:
            termination = dict(self.termination or dict(outcome='truncated', timestamp_ns=time.time_ns(),
                action_id=None, annotation_source='recorder_stop', annotated_unix_ns=time.time_ns()))
            if self.execution_audit_required and not self.actor_trace_path:
                self._note_quality_error('Actor trace path missing for audited rollout')
            if self.actor_trace_path:
                # Close the bounded raw writer before reading final ACKs into the archive.
                if not self.writer.message_journal.close():
                    raise RuntimeError('raw message journal is still draining; retry stop')
                try:
                    archive = archive_actor_trace(self.actor_trace_path, self.writer.episode_dir, rollout_id)
                    termination['last_command'] = archive['last_command']
                    if archive['missing_receipts'] or archive['missing_device_receipts']:
                        self._note_quality_error('Actor archive has missing hardware receipts')
                except Exception as exc:
                    self._note_quality_error('Actor trace archive failed: ' + str(exc))
            write_termination(self.writer.episode_dir, termination)
        result = self.writer.stop(
            save=save,
            dropped_frames=self.dropped,
            quality_errors=self.quality_errors,
            boundary_frames=dict(self.boundary_frames),
            camera_input_dropped=dict(self.camera_dropped),
            sampling_work=self._sampling_work(),
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

    def _tactile_quality_errors(self, raw=None) -> list[str]:
        errors: list[str] = []
        for key, label in (("wrench", "wrench"), ("deform", "deformation")):
            msg = self.latest[key].msg
            missing: list[str] = []
            for side in ("left", "right"):
                valid = (raw['wrench_valid' if key == 'wrench' else 'deformation_valid'][
                    ('left', 'right').index(side)] if raw is not None else self._finger_valid(msg, side))
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

    def _boundary_hand_action(self, latest=None) -> bool:
        hand = (self.latest if latest is None else latest)['hand'].msg
        reason = str(getattr(hand, 'q_cmd_invalid_reason', ''))
        expected = reason == 'stopped' or (
            not self._seen_hand_action and reason in ('not_commanded', 'source_changed'))
        if not expected:
            return False
        self.dropped += 1  # Still an invalid row, never fabricated as a command.
        self.boundary_frames[reason] = self.boundary_frames.get(reason, 0) + 1
        self.capture_notice = ('正在收尾：停止后的无动作帧已标记无效' if self._seen_hand_action
                               else '等待首条有效双手动作，启动边界帧已标记无效')
        return True

    def _alignment_now_ns(self):
        return self._alignment_origin_ns + time.monotonic_ns() - self._alignment_origin_monotonic_ns

    def flush_aligned(self, now_ns=None, *, force=False):
        """Finalize ready targets in the recording worker; never wait on control."""
        with self._lock:
            if not self.recording or self.aligner is None or self.aligner.origin_ns is None:
                return 0
            now_ns = self._alignment_now_ns() if now_ns is None else int(now_ns)
            count = 0
            # Bound time spent holding the recorder lock during recovery from stalls.
            for _ in range(self.aligner.capacity):
                matched = self.aligner.pop_ready(now_ns, end_ns=self._capture_stop_ns, force=force)
                if matched is None:
                    break
                target, selected = matched
                latest = {key: (selected[key].value if selected.get(key) is not None else _Latest())
                          for key in MODALITY_NAMES if key != "camera"}
                camera = SimpleNamespace(encoding=self.writer.source_encoding)
                camera_received = []
                for name in CAMERA_NAMES:
                    frame = selected[name]
                    setattr(camera, name + "_timestamp_ns", frame.timestamp_ns if frame else 0)
                    setattr(camera, name + "_valid", bool(frame and frame.value[0]
                                                         and frame.value[1] == self.writer.source_encoding))
                    if frame:
                        camera_received.append(frame.received_ns)
                latest["camera"] = _Latest(camera, min(camera_received, default=0) / 1e9)
                self.sample(target / 1e9, latest=latest, target_ns=target, emitted_ns=now_ns)
                count += 1
            if self.aligner.overflows:
                self._note_quality_error("recording alignment source queue overflow")
            if self.aligner.rejected:
                self._note_quality_error("recording alignment source timestamp moved backwards")
            return count

    def sample(self, now: float | None = None, *, latest=None, target_ns=None, emitted_ns=None) -> bool:
        began = time.perf_counter()
        with self._lock:
            if not self.recording:
                return False
            try:
                return self._sample(now, latest=latest, target_ns=target_ns, emitted_ns=emitted_ns)
            finally:
                elapsed_ms = (time.perf_counter()-began)*1000.
                self._sample_work_ms.append(elapsed_ms)
                self._sample_overruns += elapsed_ms > 1000./self.frequency

    def _sample(self, now: float | None = None, *, latest=None, target_ns=None, emitted_ns=None) -> bool:
        now = time.time() if now is None else float(now)
        latest = self.latest if latest is None else latest
        with self._lock:
            if not self.recording:
                return False
            missing = [
                key
                for key in self.required_modalities
                if not self._fresh(latest[key], now)
            ]
            raw = self._build_raw_row(now, latest=latest)
            raw["sampled_unix_ns"] = time.time_ns() if emitted_ns is None else emitted_ns
            if emitted_ns is not None:
                raw["source_age_ms"] = [
                    max(0., (emitted_ns / 1e9 - latest[key].received) * 1000.)
                    if latest[key].msg is not None else float("inf") for key in MODALITY_NAMES]
            if target_ns is not None:
                raw["time_unix_ns"] = target_ns
            self.writer.add_raw_row(raw)
            self._seen_hand_action |= bool(raw['hand_q_cmd_valid'])
            timeline_index = self.attempts
            self.attempts += 1
            if missing:
                return self._drop_sample(
                    "missing/stale modalities: " + ", ".join(missing)
                )
            for error in (self._tactile_quality_errors(raw) if self.record_tactile else []):
                self._note_quality_error(error)
            try:
                item = self._build_item(now, tactile=raw, defer_fk=True, latest=latest)
            except MissingHandAction as exc:
                if self._boundary_hand_action(latest):
                    return False
                return self._drop_sample(str(exc))
            except Exception as exc:
                return self._drop_sample(str(exc))
            item["metadata"]["timeline_index"] = timeline_index
            if not self.writer.add_item(item):
                return self._drop_sample("writer queue full")
            self.frames += 1
            self.capture_notice = ""
            return True

    @staticmethod
    def _raw_values(msg: Any, field: str, size: int, valid: bool) -> list[float]:
        if not valid or msg is None:
            return [float("nan")] * size
        try:
            return _vec(getattr(msg, field), size)
        except Exception:
            return [float("nan")] * size

    def _build_raw_row(self, now: float, *, latest=None) -> dict[str, Any]:
        latest = self.latest if latest is None else latest
        source_fresh = {
            key: self._fresh(latest[key], now)
            for key in MODALITY_NAMES
        }
        source_valid = dict(source_fresh)
        cam = latest["camera"].msg
        ur = latest["ur"].msg
        hand = latest["hand"].msg
        tau = latest["tau"].msg
        wrench = latest["wrench"].msg
        deform = latest["deform"].msg

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

        wrench_values, wrench_valid = self._wrench_frame(wrench, source_fresh['wrench'])
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
            if latest[key].msg is None
            else max(0.0, (now - latest[key].received) * 1000.0)
            for key in MODALITY_NAMES
        ]
        values = self._raw_values
        row = {
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
                and int(getattr(hand, "q_cmd_source", 0)) == self.hand_source
                and (not self.rollout_recording or hasattr(hand, "is_hil"))
            ),
            "hand_q_cmd_timestamp_ns": (
                int(getattr(hand, "q_cmd_timestamp_ns", 0))
                if hand is not None else 0
            ),
            "hand_q_cmd_source": (
                int(getattr(hand, "q_cmd_source", 0))
                if hand is not None else 0
            ),
            "hand_q_cmd_invalid_reason": str(getattr(hand, "q_cmd_invalid_reason", "")),
            "hand_tau": [
                *values(tau, "left", 22, source_valid["tau"]),
                *values(tau, "right", 22, source_valid["tau"]),
            ],
            "wrench": wrench_values,
            "wrench_valid": wrench_valid,
            "deformation_valid": deformation_valid,
            "camera_timestamp_ns": [
                int(getattr(cam, f"{name}_timestamp_ns", 0))
                if source_valid["camera"] else 0
                for name in CAMERA_NAMES
            ],
        }
        if self.rollout_recording:
            row["is_hil"] = bool(row["hand_q_cmd_valid"] and hand.is_hil)
        return row

    @classmethod
    def _wrench_frame(cls, message, fresh):
        """Decode each hand once; a bad finger cannot erase the other nine."""
        values = np.zeros((2, 5, 6), dtype=np.float32)
        valid = np.zeros((2, 5), dtype=bool)
        if fresh:
            for index, side in enumerate(('left', 'right')):
                try:
                    hand = np.asarray(getattr(message, side), dtype=np.float32).reshape(5, 6)
                    valid[index] = np.asarray(cls._finger_valid(message, side)) & np.isfinite(hand).all(axis=1)
                    values[index] = np.where(valid[index, :, None], hand, 0.)
                except (ValueError, TypeError, AttributeError, OverflowError):
                    pass
        return values.tolist(), valid.tolist()

    @staticmethod
    def _complete_item(item):
        """Run FK in writer workers, outside the ROS sampler and sensor lock."""
        joints = item.pop('_hand_fk', None)
        if joints is not None:
            left, right = _fk(joints)
            item['states']['left_hand_positions']['positions'] = left
            item['states']['right_hand_positions']['positions'] = right
        return item

    def _build_item(self, now: float, *, tactile=None, defer_fk=False, latest=None) -> dict[str, Any]:
        latest = self.latest if latest is None else latest
        cam = latest["camera"].msg
        ur = latest["ur"].msg
        hand = latest["hand"].msg
        tau = latest["tau"].msg
        wrench = latest["wrench"].msg
        deform = latest["deform"].msg

        if str(getattr(cam, "encoding", "jpeg")) != self.writer.source_encoding:
            raise ValueError("camera encoding does not match recording backend")
        if not bool(getattr(ur, "valid", True)):
            raise ValueError("UR state invalid")
        if (
            not bool(getattr(hand, "left_valid", True))
            or not bool(getattr(hand, "right_valid", True))
        ):
            raise ValueError("SharpA hand state invalid")
        if (
            not bool(getattr(tau, "left_valid", True))
            or not bool(getattr(tau, "right_valid", True))
        ):
            raise ValueError("SharpA tau invalid")
        if self.record_tactile and (
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

        if not bool(getattr(hand, "q_cmd_valid", False)):
            raise MissingHandAction("SharpA hand action has no complete successful SDK command")
        if int(getattr(hand, "q_cmd_source", 0)) != self.hand_source:
            raise ValueError("SharpA hand action is not from " +
                             ("Actor rollout" if self.rollout_recording else "AE01 teleop"))
        q_cmd_timestamp_ns = int(getattr(hand, "q_cmd_timestamp_ns", 0))
        if q_cmd_timestamp_ns <= 0:
            raise ValueError("SharpA hand action timestamp is invalid")
        hand_state_timestamp_ns = int(getattr(hand, "timestamp_ns", 0))
        if q_cmd_timestamp_ns > hand_state_timestamp_ns:
            raise ValueError("SharpA hand action is newer than its paired hand state")

        q44 = _vec(hand.q, 44)
        q_cmd44 = _vec(hand.q_cmd, 44)
        left_tip, right_tip = (None, None) if defer_fk else _fk(q44)
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
        }
        if self.record_tactile:
            if tactile is None:
                values, valid = self._wrench_frame(wrench, self._fresh(latest['wrench'], now))
                tactile = dict(wrench=values, wrench_valid=valid)
            states["tactile"] = {
                side+'_tactile': dict(zip(FINGERS, tactile['wrench'][index]))
                for index, side in enumerate(('left', 'right'))
            }
        if hasattr(ur, "left_tcp_pose6") and hasattr(ur, "right_tcp_pose6"):
            states["left_arm"]["tcp_pose6"] = _vec(ur.left_tcp_pose6, 6)
            states["right_arm"]["tcp_pose6"] = _vec(ur.right_tcp_pose6, 6)
        # q and q_cmd come from one HandStateFrame: measured feedback plus the
        # most recent complete target successfully handed to the SDK.
        # Arm commands are intentionally not recorded, matching teleop data.
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
            "tactile": int(getattr(wrench, "timestamp_ns", 0)) / 1e9 if self.record_tactile else 0.,
            "joint_torque": int(getattr(tau, "timestamp_ns", 0)) / 1e9,
            "frame_end": time.time(),
        }
        validity = {
            kind: {
                side: self._finger_valid(msg, side)
                for side in ("left", "right")
            }
            for kind, msg in (("wrench", wrench), ("deformation", deform))
        }
        if self.record_tactile:
            validity['wrench'] = {side: tactile['wrench_valid'][index]
                                  for index, side in enumerate(('left', 'right'))}
        return {
            **({'_hand_fk': q44} if defer_fk else {}),
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
                "tactile_enabled": self.record_tactile,
                **({"is_hil": bool(hand.is_hil)} if self.rollout_recording else {}),
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
            "started_unix_ns": self.writer._started_unix_ns,
            "achieved_rate_hz": round(self.frames / max(1e-9, elapsed), 3),
            "dropped_frames": self.dropped,
            "deformation_dropped": self.deformation_dropped,
            "camera_frames": self.writer.camera_frame_counts(),
            "camera_dropped": dict(self.camera_dropped),
            "video_encoding": self.writer.video_status(),
            "last_error": self.last_error,
            "boundary_frames": dict(self.boundary_frames),
            "capture_notice": (self.capture_notice if self.recording else
                f"启停边界 {sum(self.boundary_frames.values())} 帧缺少动作，已保留无效标记"
                if self.boundary_frames else ""),
            "recording_root": str(self.writer.root),
            "saved_path": str(self.writer.last_result_dir or ''),
            "validation": self.validation,
            "tactile_enabled": self.record_tactile,
            "sampling_work": self._sampling_work(),
            "sampling_alignment": self.aligner.status() if self.aligner is not None else None,
            "writer_queues": self.writer.queue_status(),
            "raw_messages": (self.writer.message_journal.status()
                             if self.writer.message_journal is not None else None),
        }

    def _sampling_work(self):
        with self._lock:
            work = np.asarray(self._sample_work_ms)
            return dict(budget_ms=round(1000./self.frequency, 3),
                        p95_ms=round(float(np.percentile(work, 95)), 3) if work.size else 0.,
                        max_ms=round(float(work.max()), 3) if work.size else 0.,
                        overruns=int(self._sample_overruns), window_frames=int(work.size),
                        alignment=self.aligner.status() if self.aligner is not None else None)


if ROS_AVAILABLE:
    class RecordingNode(Node):
        def __init__(self) -> None:
            super().__init__("recording_node")
            self.declare_parameter("auto_record_rollout", False)
            self.declare_parameter("workbench_mode", False)
            self._workbench = bool(self.get_parameter("workbench_mode").value)
            self._auto_rollout = bool(self.get_parameter("auto_record_rollout").value)
            root_preferences = RecordingPreferences()
            self.declare_parameter(
                "recording_root", root_preferences.read(DEFAULT_HIL_RECORDING_ROOT)
            )
            self.declare_parameter("frequency", 30.0)
            self.declare_parameter("camera_max_age_s", setting("recording", "camera_max_age_s"))
            self.declare_parameter("shared_memory_prefix", "ur_sharpa_camera_v1")
            self.declare_parameter("video_ffmpeg", "ffmpeg")
            self.declare_parameter("video_backend", "hevc_nvenc")
            self.declare_parameter("ego_video_bitrate_mbps", 30.0)
            self.declare_parameter("wrist_video_bitrate_mbps", 12.0)
            self._raw_video = str(self.get_parameter("video_backend").value) == "hevc_nvenc"
            if self._raw_video:
                check_nvenc(str(self.get_parameter("video_ffmpeg").value))
                self.get_logger().info("NVENC HEVC startup check passed; GOP=2, B=0, lookahead=0")
            self.declare_parameter("raw_message_budget_mb", 64)
            self.declare_parameter("skip_tactile", False)
            self.declare_parameter("alignment_wait_frames", 2.0)
            self.declare_parameter("deformation_shm_prefix", DEFORMATION_PREFIX)
            self.declare_parameter("deformation_shared_memory", True)

            frequency = float(self.get_parameter("frequency").value)
            self.core = RecorderCore(
                str(self.get_parameter("recording_root").value),
                frequency,
                float(self.get_parameter("camera_max_age_s").value),
                video_ffmpeg=str(self.get_parameter("video_ffmpeg").value),
                raw_message_budget_bytes=int(self.get_parameter("raw_message_budget_mb").value) * 1024 * 1024,
                record_tactile=not bool(self.get_parameter("skip_tactile").value),
                video_backend=str(self.get_parameter("video_backend").value),
                ego_bitrate_mbps=float(self.get_parameter("ego_video_bitrate_mbps").value),
                wrist_bitrate_mbps=float(self.get_parameter("wrist_video_bitrate_mbps").value),
                rollout_recording=self._auto_rollout,
                root_preferences=root_preferences,
                alignment_wait_frames=float(self.get_parameter("alignment_wait_frames").value),
            )
            self.deformation_reader = (DeformationReader(str(self.get_parameter("deformation_shm_prefix").value))
                if self.core.record_tactile and bool(self.get_parameter("deformation_shared_memory").value) else None)
            self._transport_deformation_dropped = 0
            self._saving = False
            self._record_config_result = {}
            self._save_thread = None
            self.rollout = RolloutRecorder(self.core, self._begin_save, lambda: self._saving)
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
            if self._auto_rollout or self._workbench:
                self.create_subscription(String, '/hil/rollout', self._rollout_request, default_qos)
                self.create_timer(0.05, self._rollout_tick)
            if self._workbench:
                self.create_subscription(ControlStatus, '/control/status', self._control_mode, default_qos)
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
                lambda msg: self._sensor("ur", "/ur/state", "UrStateFrame", msg),
                default_qos,
            )
            self.create_subscription(
                HandStateFrame,
                "/sharpa/state",
                lambda msg: self._sensor("hand", "/sharpa/state", "HandStateFrame", msg),
                default_qos,
            )
            self.create_subscription(
                TauFrame,
                "/sharpa/tau",
                lambda msg: self._sensor("tau", "/sharpa/tau", "TauFrame", msg),
                default_qos,
            )
            if self.core.record_tactile:
                self.create_subscription(
                    WrenchFrame,
                    "/sharpa/wrench",
                    lambda msg: self._sensor("wrench", "/sharpa/wrench", "WrenchFrame", msg),
                    default_qos,
                )
                if self.deformation_reader is None:
                    self.create_subscription(
                        DeformationFrame, "/sharpa/deformation", self.core.update_deformation,
                        QoSProfile(depth=16, reliability=ReliabilityPolicy.RELIABLE),
                    )
            # Passive observers: do not add reliable backpressure to control.
            event_qos = QoSProfile(depth=256, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(
                HandCommand, "/sharpa/command",
                lambda msg: self._raw_message("/sharpa/command", "ur_msgs/msg/HandCommand", msg),
                event_qos,
            )
            self.create_subscription(
                HardwareCommandResult, "/sharpa/command_result",
                lambda msg: self._raw_message("/sharpa/command_result", "ur_msgs/msg/HardwareCommandResult", msg),
                event_qos,
            )
            self.create_subscription(
                String, "/deploy/execution_event",
                lambda msg: self._raw_message("/deploy/execution_event", "std_msgs/msg/String", msg),
                event_qos,
            )
            for message_type, topic in ((ArmCommand, '/ur/command'),
                                        (HardwareCommandResult, '/ur/command_result')):
                self.create_subscription(message_type, topic,
                    lambda msg, topic=topic, name=message_type.__name__: self._raw_message(
                        topic, 'ur_msgs/msg/' + name, msg), event_qos)
            self.create_subscription(String, '/policy/inference', self._inference, 256)
            for topic in ("/camera/acquisition_status", "/sharpa/acquisition_status"):
                self.create_subscription(
                    String, topic,
                    lambda msg, topic=topic: self._raw_message(topic, "std_msgs/msg/String", msg),
                    event_qos,
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
            self.create_timer(0.1 if self._auto_rollout else 1.0, self._publish)

            shared_memory_prefix = str(
                self.get_parameter("shared_memory_prefix").value
            )
            self.camera_reader = SharedCameraReader(
                shared_memory_prefix + ("_raw" if self._raw_video else ""),
                encoding=RAW_ENCODING if self._raw_video else "jpeg")
            self._shared_camera_seen = False
            self._observed_camera_timestamps = None
            self._transport_camera_dropped = {
                name: 0 for name in CAMERA_NAMES
            }
            self._camera_poll_stop = threading.Event()
            self._camera_poll_thread = None
            if self._raw_video or self.core.aligner is not None or self.deformation_reader is not None:
                # Media draining/alignment must not wait for ROS event callbacks.
                self._camera_poll_thread = threading.Thread(
                    target=self._camera_poll_loop, name="recording-raw-camera", daemon=True)
                self._camera_poll_thread.start()
            else:
                self.create_timer(1.0 / max(30.0, frequency * 4.0), self._poll_camera)
            if self.core.aligner is None:
                self.create_timer(1.0 / frequency, self.core.sample)

        def _camera_poll_loop(self):
            while not self._camera_poll_stop.is_set():
                try:
                    self._poll_camera()
                    self._poll_deformation()
                    self.core.flush_aligned()
                except Exception as exc:
                    with self.core._lock:
                        self.core.last_error = f"raw camera transport failed: {exc}"
                        if self.core.recording:
                            self.core._note_quality_error(self.core.last_error)
                    self.get_logger().error(self.core.last_error)
                self._camera_poll_stop.wait(0.001)

        def _poll_deformation(self):
            if self.deformation_reader is None:
                return
            pending = self.deformation_reader.read_pending()
            dropped = self.deformation_reader.dropped_frames
            with self.core._lock:
                if self.core.recording and dropped > self._transport_deformation_dropped:
                    count = dropped - self._transport_deformation_dropped
                    self.core.deformation_dropped += count
                    self.core._note_quality_error(f"deformation transport lost {count} frame(s)")
                self._transport_deformation_dropped = dropped
            for msg in pending:
                self.core.update_deformation(msg)

        def _command(self, msg: String) -> None:
            if self._auto_rollout:
                return  # Live execution owns the complete rollout lifecycle.
            command = msg.data.strip()
            if command == "start":
                if not self._saving:
                    self.core.start()
                    self._publish()
            elif command == "stop":
                self._begin_save()
            elif command == "discard":
                if not self._saving:
                    self.core.stop(False)

        def _control_mode(self, msg):
            rollout = int(msg.active_source) != 2
            if rollout != self._auto_rollout and self.core.configure_capture_mode(rollout, saving=self._saving):
                self._auto_rollout = rollout
                self.rollout = RolloutRecorder(self.core, self._begin_save, lambda: self._saving)
                self._publish()

        def _rollout_tick(self):
            if self._auto_rollout:
                self.rollout.tick()

        def _rollout_request(self, msg):
            if not self._auto_rollout:
                return
            try:
                self.rollout.request(json.loads(msg.data))
                self.rollout.tick()
                self._publish()
            except (ValueError, TypeError, AttributeError) as exc:
                self.get_logger().warning(f'invalid rollout request: {exc}')

        def _begin_save(self):
            if self.core.writer.active and not self._saving:
                self._saving = True
                self._publish()
                self._save_thread = threading.Thread(target=self._save_recording, daemon=True)
                self._save_thread.start()

        def _save_recording(self) -> None:
            try:
                saved = self.core.stop(True)
                if self._auto_rollout and saved is None:
                    self.rollout.error = 'rollout saved with recording errors; inspect RECORDING_ERRORS.json'
            except Exception as exc:
                self.core.last_error = f"recording save failed: {exc}"
                self.rollout.error = self.core.last_error
                self.get_logger().error(self.core.last_error)
            finally:
                self._saving = False
                self._publish()

        def _camera(self, msg: CameraSet) -> None:
            if not self._raw_video and not self._shared_camera_seen:
                self.core.update_camera(msg)

        def _inference(self, msg):
            try:
                self.core.record_inference(msg.data, serialize_message(msg))
            except Exception as exc:
                with self.core._lock:
                    if self.core.recording or self.core._pending_inferences:
                        self.core.inference_error('invalid policy inference event: ' + str(exc))

        def _raw_message(self, topic, type_name, msg, received_ns=None, monotonic_ns=None):
            if topic == "/camera/acquisition_status":
                self.core.observe_diagnostic(msg.data)
            if not self.core.recording:
                return
            received_ns = time.time_ns() if received_ns is None else received_ns
            monotonic_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
            try:
                payload = serialize_message(msg)
            except Exception as exc:
                with self.core._lock:
                    if self.core.recording:
                        self.core.writer.message_journal.fail(f"{topic} serialization failed: {exc}")
                return
            self.core.record_message(topic, type_name, payload,
                                     _stamp_ns(msg), received_ns, monotonic_ns)

        def _sensor(self, key, topic, type_name, msg):
            received_ns, monotonic_ns = time.time_ns(), time.monotonic_ns()
            self.core.update(key, msg, received=received_ns / 1e9)
            self._raw_message(topic, "ur_msgs/msg/" + type_name, msg,
                              received_ns, monotonic_ns)

        def _poll_camera(self) -> None:
            pending = self.camera_reader.read_pending()
            dropped = self.camera_reader.dropped_frames
            for name in CAMERA_NAMES:
                previous = self._transport_camera_dropped[name]
                current = int(dropped.get(name, 0))
                with self.core._lock:
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
            result = {}
            try:
                config = json.loads(msg.data)
                result["id"] = config.get("id", "")
                root = self.core.configure_root(
                    config.get("recording_root"), saving=self._saving,
                    requested=bool(self._auto_rollout and self.rollout.requested),
                )
                result.update(accepted=True, recording_root=str(root))
            except Exception as exc:
                result.update(accepted=False, reason=str(exc))
                self.get_logger().warning(f"invalid record config: {exc}")
            self._record_config_result = result
            self._publish()

        def _label(self, msg: String) -> None:
            try:
                if self._saving:
                    raise RuntimeError("cannot classify/drop while saving")
                label = msg.data.strip()
                if label.startswith("{"):
                    request = json.loads(label)
                    if request.get("saved_path") != str(self.core.writer.last_result_dir or ''):
                        raise RuntimeError("样本已变化，分类请求已拒绝")
                    label = request.get("category", "")
                if label.lower() == "drop":
                    removed = self.core.drop_last_saved()
                    self.get_logger().warning(
                        f"sample permanently deleted: {removed}"
                    )
                else:
                    destination = self.core.classify(label)
                    self.get_logger().info(f"sample classified: {destination}")
            except Exception as exc:
                self.core.last_error = f"sample classification/drop failed: {exc}"
                self.get_logger().warning(self.core.last_error)

        def _publish(self) -> None:
            msg = String()
            status = self.core.status()
            status["saving"] = bool(self._saving)
            status["config_result"] = dict(self._record_config_result)
            status["capture_mode"] = "rollout" if self._auto_rollout else "teleop"
            status["rollout_requested"] = bool(self._auto_rollout and self.rollout.requested)
            if self._auto_rollout:
                status.update(self.rollout.status())
            msg.data = json.dumps(status, separators=(",", ":"))
            self.status_pub.publish(msg)

        def destroy_node(self) -> Any:
            if self._save_thread is not None and self._save_thread.is_alive():
                self._save_thread.join(timeout=35.0)
            elif self.core.recording:
                self.core.stop(True)
            self._camera_poll_stop.set()
            if self._camera_poll_thread is not None:
                self._camera_poll_thread.join()
            self.camera_reader.close()
            if self.deformation_reader is not None:
                self.deformation_reader.close()
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
