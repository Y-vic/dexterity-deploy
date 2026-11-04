"""PND-style asynchronous episode writer.

Structured samples, tactile deformation, and the three camera streams use
independent queues. Raw BGR frames use NVENC HEVC; the legacy JPEG backend
remuxes MJPEG. ROS callbacks never wait for FFmpeg or filesystem I/O.
"""
from __future__ import annotations

from ur_common.raw_camera import unpack_bgr, RAW_ENCODING
from .video_encoding import nvenc_command, write_frame

import datetime
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .message_journal import MessageJournal
from .rate_metrics import cadence, MIN_VALID_RATE_HZ, RATE_TOLERANCE

CAMERA_NAMES = ("ego", "left_wrist", "right_wrist")
CAMERA_COLOR_KEYS = ("color_0", "color_1", "color_2")
FINGER_NAMES = ("小指", "无名指", "中指", "食指", "拇指")


def _complete_jpeg(data: bytes) -> bool:
    return len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def _validation_rate(frames: int, duration_s: float) -> float:
    return round(float(frames) / duration_s, 3) if duration_s > 0.0 else 0.0


def _missing_finger_summary(valid: np.ndarray, plane_names: tuple[str, ...]) -> list[dict]:
    if valid.ndim != 2 or valid.shape[1] != len(plane_names):
        return []
    missing = []
    total = int(valid.shape[0])
    for index, name in enumerate(plane_names):
        count = int(np.count_nonzero(~valid[:, index]))
        if count:
            missing.append({
                "finger": name,
                "missing_frames": count,
                "total_frames": total,
                "ratio_percent": round(100.0 * count / max(1, total), 2),
            })
    return missing


def validate_saved_episode(directory: Path, frequency: float, quality: dict) -> dict:
    """Read finalized files and measure valid rows, advancing sources and gaps."""
    directory = Path(directory)
    clock = np.load(directory / "timeline" / "clock.npz", allow_pickle=False)
    timeline_rows = int(len(clock["time_unix_ns"]))
    nominal_period = 1.0 / max(float(frequency), 1.0e-9)
    if timeline_rows > 1:
        elapsed = float(clock["time_unix_ns"][-1] - clock["time_unix_ns"][0]) / 1.0e9
        duration_s = max(nominal_period, elapsed + nominal_period)
    elif timeline_rows == 1:
        duration_s = nominal_period
    else:
        duration_s = 0.0
    minimum_hz = float(frequency) * (1. - RATE_TOLERANCE)
    sampling = cadence(clock['time_unix_ns'], frequency, duration_s)

    def item(key, label, frames, *, missing=0, details=None, files_ok=True,
             finger_missing=None, source_stamps=(), cadence_required=True):
        details = list(details or [])
        finger_missing = list(finger_missing or [])
        missing = max(0, int(missing))
        if not files_ok:
            details.insert(0, "file missing")
        rate = _validation_rate(int(frames), duration_s)
        rate_ok = bool(rate >= minimum_hz)
        timing = cadence(source_stamps, frequency, duration_s,
                         window_start_ns=int(clock['time_unix_ns'][0]) if timeline_rows else None)
        timing["cadence_required"] = cadence_required
        if not rate_ok:
            details.append(f"低于 {minimum_hz:g} Hz（目标 {frequency:g} Hz）")
        if cadence_required and not timing['stable']:
            details.append(f"源帧 {timing['source_hz']:g} Hz；重复 {timing['repeated_frames']} 帧；长间隔 {timing['gap_count']} 次")
        details.extend(
            f"{row['finger']} {row['ratio_percent']:.2f}%"
            for row in finger_missing
        )
        timing_ok = timing['stable'] if cadence_required else timing['backwards'] == 0
        ok = bool(files_ok and rate_ok and timing_ok and not finger_missing and not any(
            text in {"valid shape invalid", "video missing", "file missing"}
            for text in details
        ))
        return {
            "key": key,
            "label": label,
            "frames": int(frames),
            "duration_s": round(duration_s, 3),
            "hz": rate,
            "minimum_hz": minimum_hz,
            "timing": timing,
            "rate_ok": rate_ok,
            "missing": missing,
            "ok": ok,
            "details": details,
            "finger_missing": finger_missing,
        }

    items = []
    for name, label in (
        ("ego", "Ego Camera"),
        ("left_wrist", "Left Wrist Camera"),
        ("right_wrist", "Right Wrist Camera"),
    ):
        index_path = directory / "cameras" / name / "index.npz"
        video_path = directory / "cameras" / name / "rgb.mp4"
        count = 0
        source_stamps = []
        details = []
        if index_path.is_file():
            index = np.load(index_path, allow_pickle=False)
            valid = np.asarray(index["valid"], dtype=bool)
            count = int(np.count_nonzero(valid))
            source_stamps = index['source_timestamp_ns'][valid]
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            details.append("video missing")
        missing = max(0, timeline_rows - count)
        items.append(item(f"{name}_camera", label, count, missing=missing, details=details,
                          files_ok=index_path.is_file(), source_stamps=source_stamps))

    arm_path = directory / "ur" / "physical_12.npz"
    arm_count = 0
    arm_stamps = []
    if arm_path.is_file():
        arm = np.load(arm_path, allow_pickle=False)
        arm_count = int(np.count_nonzero(np.asarray(arm["valid"], dtype=bool)))
        arm_stamps = arm['timestamp_ns'][np.asarray(arm['valid'], dtype=bool)]
    items.append(item("arm_state", "Arm State", arm_count,
                      missing=timeline_rows-arm_count, files_ok=arm_path.is_file(), source_stamps=arm_stamps))

    joints_path = directory / "sharpa" / "joints_44.npz"
    hand_stamps, action_stamps = [], []
    if joints_path.is_file():
        joints = np.load(joints_path, allow_pickle=False)
        hand_valid = np.asarray(joints["q_valid"], dtype=bool)
        action_valid = np.asarray(joints["q_cmd_valid"], dtype=bool)
        if "q_cmd_source" in joints:
            action_valid &= np.isin(np.asarray(joints["q_cmd_source"], dtype=np.uint8), (2, 3))
        hand_count = int(np.count_nonzero(hand_valid))
        action_count = int(np.count_nonzero(action_valid))
        hand_stamps = joints['q_timestamp_ns'][hand_valid]
        action_stamps = joints['q_cmd_timestamp_ns'][action_valid]
    else:
        hand_count = action_count = 0
    items.append(item("hand_state", "Hand State", hand_count,
                      missing=timeline_rows-hand_count, files_ok=joints_path.is_file(), source_stamps=hand_stamps))
    items.append(item("hand_action", "Hand Action", action_count,
                      missing=timeline_rows-action_count, files_ok=joints_path.is_file(),
                      source_stamps=action_stamps, cadence_required=False))

    wrench_path = directory / "sharpa" / "tactile" / "force6d.npz"
    wrench_count = 0
    wrench_details = []
    wrench_finger_missing = []
    wrench_stamps = []
    if wrench_path.is_file():
        wrench = np.load(wrench_path, allow_pickle=False)
        wrench_valid = np.asarray(wrench["valid"], dtype=bool)
        if wrench_valid.ndim == 3 and wrench_valid.shape[1:] == (2, 5):
            received_frames = np.any(wrench_valid, axis=(1, 2))
            wrench_count = int(np.count_nonzero(received_frames))
            wrench_stamps = wrench['timestamp_ns'][received_frames]
            wrench_finger_missing = _missing_finger_summary(
                wrench_valid[received_frames].reshape(-1, 10),
                tuple(f"{side}{finger}" for side in ("左", "右") for finger in FINGER_NAMES),
            )
        else:
            wrench_details = ["valid shape invalid"]
    items.append(item("wrench", "Wrench", wrench_count,
                      missing=timeline_rows-wrench_count, details=wrench_details,
                      files_ok=wrench_path.is_file(),
                      finger_missing=wrench_finger_missing, source_stamps=wrench_stamps))

    deform_path = directory / "sharpa" / "tactile" / "deform_index.npz"
    deform_count = 0
    deform_details = []
    deform_finger_missing = []
    deform_stamps = []
    if deform_path.is_file():
        deform = np.load(deform_path, allow_pickle=False)
        deform_valid = np.asarray(deform["valid"], dtype=bool)
        if deform_valid.ndim == 2 and deform_valid.shape[1] == 10:
            # The independently arriving stream may map adjacent source frames
            # to the same nearest timeline row. Count actual saved frames.
            deform_count = int(np.count_nonzero(np.any(deform_valid, axis=1)))
            deform_stamps = deform['source_timestamp_ns'][np.any(deform_valid, axis=1)]
            deform_finger_missing = _missing_finger_summary(
                deform_valid,
                tuple(f"{side}{finger}" for side in ("右", "左") for finger in FINGER_NAMES),
            )
        else:
            deform_details = ["valid shape invalid"]
    items.append(item("deform", "Deform", deform_count,
                      missing=timeline_rows-deform_count, details=deform_details,
                      files_ok=deform_path.is_file(),
                      finger_missing=deform_finger_missing, source_stamps=deform_stamps))

    # Joint motor torques are separate from the fingertip's Tx/Ty/Tz.
    if joints_path.is_file():
        tau_valid = np.asarray(joints['tau_valid'], dtype=bool)
        tau_count = int(np.count_nonzero(tau_valid))
        items.append(item('joint_torque', 'Joint Torque', tau_count, missing=timeline_rows-tau_count,
                          source_stamps=joints['tau_timestamp_ns'][tau_valid]))

    disabled = [] if quality.get("tactile_enabled", True) else ["wrench", "deform"]
    items = [row for row in items if row["key"] not in disabled]
    quality_errors = [str(row.get("message", "")) for row in quality.get("errors", []) if row.get("message")]
    result = {
        "available": True,
        "episode": directory.name,
        "duration_s": round(duration_s, 3),
        "timeline_rows": timeline_rows,
        "minimum_hz": minimum_hz,
        "target_hz": float(frequency),
        "sampling": sampling,
        "ok": bool(quality.get("ok", False) and sampling['stable'] and all(row["ok"] for row in items)),
        "items": items,
        "errors": quality_errors,
        "disabled_modalities": disabled,
    }
    (directory / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _default_info(frequency: float, image_size=(640, 480)) -> dict:
    return {
        "version": "1.0.0",
        "date": datetime.date.today().isoformat(),
        "author": "unitree",
        "image": {
            "width": int(image_size[0]),
            "height": int(image_size[1]),
            "fps": frequency,
        },
        "depth": {
            "width": int(image_size[0]),
            "height": int(image_size[1]),
            "fps": frequency,
        },
        "audio": {"sample_rate": 16000, "channels": 1, "format": "PCM", "bits": 16},
        "tactile": {"shape": [1140, 3], "dtype": "int16", "fps": frequency},
        "joint_names": {
            "left_arm": [
                "kLeftShoulderPitch",
                "kLeftShoulderRoll",
                "kLeftShoulderYaw",
                "kLeftElbow",
                "kLeftWristRoll",
                "kLeftWristPitch",
                "kLeftWristyaw",
            ],
            "left_hand": [],
            "right_arm": [],
            "right_hand": [],
            "body": [],
        },
        "tactile_names": {"left_hand": [], "right_hand": []},
    }


class EpisodeWriterCompat:
    """Asynchronous, all-or-nothing episode writer.

    ``add_item`` is non-blocking.  If its bounded queue is full it returns
    ``False`` and the recorder reports a dropped frame; it never stalls ROS.
    ``discard`` drains the queue and removes the whole episode directory.
    """

    def __init__(
        self,
        root: str | Path,
        frequency: float = 30.0,
        workers: int | None = None,
        queue_size: int = 100,
        ffmpeg: str = "ffmpeg",
        raw_message_budget_bytes: int = 64 * 1024 * 1024,
        record_tactile: bool = True,
        video_backend: str = "mjpeg",
        ego_bitrate_mbps: float = 30.0,
        wrist_bitrate_mbps: float = 12.0,
        prepare_item=None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.frequency = float(frequency)
        self.workers = max(1, int(workers or os.environ.get("EPISODE_WRITER_THREADS", "4")))
        self.ffmpeg = str(ffmpeg).strip() or "ffmpeg"
        self.ffprobe = os.environ.get("EPISODE_WRITER_FFPROBE", "ffprobe")
        self.raw_message_budget_bytes = int(raw_message_budget_bytes)
        if self.raw_message_budget_bytes <= 0:
            raise ValueError("raw message budget must be positive")
        self.message_journal = None
        self.record_tactile = bool(record_tactile)
        self.prepare_item = prepare_item
        if video_backend not in ("mjpeg", "hevc_nvenc"):
            raise ValueError("unsupported video_backend")
        self.video_backend = video_backend
        self.video_codec = "hevc" if video_backend == "hevc_nvenc" else "mjpeg"
        self.source_encoding = RAW_ENCODING if video_backend == "hevc_nvenc" else "jpeg"
        self.video_bitrates = {"ego": float(ego_bitrate_mbps),
                               "left_wrist": float(wrist_bitrate_mbps),
                               "right_wrist": float(wrist_bitrate_mbps)}
        if any(not np.isfinite(v) or v <= 0 for v in self.video_bitrates.values()):
            raise ValueError("video bitrates must be finite and positive")
        self._camera_dimensions = {}
        self._camera_failed = {}
        self._video_stats = {}

        queue_size = max(1, int(queue_size))
        self._q = queue.Queue(maxsize=queue_size)
        self._deformation_q = queue.Queue(maxsize=queue_size)
        # A blocked FFmpeg stdin for one camera must not hold up the other two.
        # Keep one bounded queue and one writer thread per stream.
        # Raw BGR queues must be bounded: 64 HD720 frames are ~169 MiB.
        # Overflow is explicitly reported, never hidden by unbounded RAM use.
        self._camera_queues = {name: queue.Queue(maxsize=64 if self.video_backend == "hevc_nvenc" else 0)
                               for name in CAMERA_NAMES}
        self._threads: list[threading.Thread] = []
        self._stop = False

        self._lock = threading.RLock()
        self._pending = 0
        self._deformation_pending = 0
        self._camera_pending = 0
        self._errors: list[tuple[Any, str]] = []
        self._items: list[dict[str, Any]] = []
        self._raw_rows: list[dict[str, Any]] = []
        self._idx = -1

        self._deformation_lock = threading.Lock()
        self._deformation_handle = None
        self._deformation_frames = 0
        self._deformation_rows: list[dict[str, Any]] = []
        self._started_unix_ns = 0

        self._camera_lock = threading.Lock()
        self._camera_processes: dict[str, subprocess.Popen] = {}
        self._camera_logs: dict[str, Any] = {}
        self._camera_rows = {name: [] for name in CAMERA_NAMES}
        self._camera_probe: dict[str, dict[str, Any]] = {}

        self.active = False
        self.episode_dir: Path | None = None
        self.episode_id: str | None = None
        self.last_result_dir: Path | None = None
        self.last_validation: dict | None = None
        for _ in range(self.workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self._threads.append(worker)
        self._deformation_thread = threading.Thread(
            target=self._deformation_worker,
            daemon=True,
        )
        self._deformation_thread.start()
        self._camera_threads = []
        for name in CAMERA_NAMES:
            thread = threading.Thread(
                target=self._camera_worker,
                args=(name,),
                name=f"episode-{name}-video",
                daemon=True,
            )
            thread.start()
            self._camera_threads.append(thread)

    def _time_id(self) -> str:
        """Return a sortable, collision-resistant wall-clock recording id."""
        while True:
            episode_id = datetime.datetime.now().astimezone().strftime(
                "%Y-%m-%d_%H-%M-%S.%f"
            )
            names = (episode_id, f"{episode_id}.partial")
            if not any(
                (parent / name).exists()
                for parent in (self.root, self.root / "success", self.root / "fail")
                for name in names
            ):
                return episode_id
            time.sleep(0.000001)

    def _episode_info(self):
        info = _default_info(self.frequency)
        if not self.record_tactile:
            info.pop("tactile", None)
            info["tactile_enabled"] = False
        return info

    def start(self) -> bool:
        if self.active:
            return False
        self.last_result_dir = None
        self.last_validation = None
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "success").mkdir(exist_ok=True)
        (self.root / "fail").mkdir(exist_ok=True)
        self.episode_id = self._time_id()
        self.episode_dir = self.root / f"{self.episode_id}.partial"
        for relative_path in (
            "timeline",
            "ur",
            "sharpa",
            "cameras",
        ):
            (self.episode_dir / relative_path).mkdir(parents=True, exist_ok=True)
        for name in CAMERA_NAMES:
            (self.episode_dir / "cameras" / name).mkdir(
                parents=True,
                exist_ok=True,
            )
        deformation_path = (
            self.episode_dir / "sharpa" / "tactile" / "deform_u8.raw.tmp"
        )
        if self.record_tactile:
            deformation_path.parent.mkdir(parents=True, exist_ok=True)
            self._deformation_handle = deformation_path.open("wb")
        self._deformation_frames = 0
        self._deformation_rows = []
        self._started_unix_ns = time.time_ns()
        self._idx = -1
        self._items = []
        self._raw_rows = []
        self._errors = []
        self._camera_rows = {name: [] for name in CAMERA_NAMES}
        self._camera_probe = {}
        self._camera_dimensions = {}
        self._camera_failed = {}
        self._video_stats = {name: {"accepted": 0, "written": 0, "rejected": 0,
                                   "queue_peak": 0, "write_ms_sum": 0.0,
                                   "write_ms_max": 0.0, "queue_wait_ms_max": 0.0}
                             for name in CAMERA_NAMES}
        self.message_journal = MessageJournal(
            self.episode_dir / "raw", self.raw_message_budget_bytes)
        self._start_camera_streams()
        self.active = True
        return True

    def add_raw_row(self, row: dict[str, Any]) -> None:
        if self.active:
            with self._lock:
                self._raw_rows.append(row)

    def add_item(self, item: dict) -> bool:
        if not self.active:
            return False
        with self._lock:
            self._idx += 1
            item = dict(item)
            item["idx"] = self._idx
            self._pending += 1
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self._pending -= 1
            return False

    def add_deformation(self, msg: Any, timeline_row: int, time_unix_ns: int) -> bool:
        if not self.record_tactile:
            return False
        if not self.active:
            return False
        record = {
            "left": msg.left,
            "right": msg.right,
            "left_valid": list(msg.left_valid),
            "right_valid": list(msg.right_valid),
            "source_timestamp_ns": int(msg.timestamp_ns),
            "timeline_row": int(timeline_row),
            "time_unix_ns": int(time_unix_ns),
        }
        with self._lock:
            self._deformation_pending += 1
        try:
            self._deformation_q.put_nowait(record)
            return True
        except queue.Full:
            with self._lock:
                self._deformation_pending -= 1
            return False

    def add_camera_frame(self, name: str, jpeg: Any, source_timestamp_ns: int,
                         timeline_row: int, time_unix_ns: int) -> bool:
        if not self.active or name not in CAMERA_NAMES:
            return False
        with self._camera_lock:
            if name in self._camera_failed:
                self._video_stats[name]["rejected"] += 1
                return False
            if self.video_backend == "mjpeg" and name not in self._camera_processes:
                return False
        record = {
            "name": name,
            "jpeg": bytes(jpeg),
            "source_timestamp_ns": int(source_timestamp_ns),
            "timeline_row": int(timeline_row),
            "time_unix_ns": int(time_unix_ns),
            "enqueued_monotonic": time.monotonic(),
        }
        valid_payload = _complete_jpeg(record["jpeg"])
        if self.video_backend == "hevc_nvenc":
            try:
                width, height, _ = unpack_bgr(record["jpeg"])
                valid_payload = width % 2 == 0 and height % 2 == 0
            except ValueError:
                valid_payload = False
        if (
            not valid_payload
            or record["source_timestamp_ns"] <= 0
        ):
            with self._camera_lock:
                self._video_stats[name]["rejected"] += 1
            return False
        with self._lock:
            self._camera_pending += 1
        try:
            self._camera_queues[name].put_nowait(record)
            with self._camera_lock:
                stats = self._video_stats[name]
                stats["accepted"] += 1
                stats["queue_peak"] = max(stats["queue_peak"], self._camera_queues[name].qsize())
            return True
        except queue.Full:
            with self._camera_lock:
                self._video_stats[name]["rejected"] += 1
            with self._lock:
                self._camera_pending -= 1
            return False

    def _write_deformation(self, record: dict) -> None:
        left_bytes = bytes(record["left"])
        right_bytes = bytes(record["right"])
        plane_bytes = 5 * 240 * 240
        if len(left_bytes) != plane_bytes or len(right_bytes) != plane_bytes:
            raise ValueError("SharpA deformation shape invalid in writer")
        left_valid = [bool(value) for value in record["left_valid"]]
        right_valid = [bool(value) for value in record["right_valid"]]
        if len(left_valid) != 5 or len(right_valid) != 5:
            raise ValueError("SharpA deformation validity shape invalid in writer")
        frame_bytes = right_bytes + left_bytes
        with self._deformation_lock:
            if self._deformation_handle is None:
                raise RuntimeError("SharpA deformation stream is closed")
            raw_frame_index = self._deformation_frames
            self._deformation_handle.write(frame_bytes)
            self._deformation_frames += 1
            self._deformation_rows.append({
                "time_unix_ns": record["time_unix_ns"],
                "timeline_row": record["timeline_row"],
                "raw_frame_index": raw_frame_index,
                "source_timestamp_ns": record["source_timestamp_ns"],
                "valid": [*right_valid, *left_valid],
            })

    def _worker(self) -> None:
        while not self._stop or not self._q.empty():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.prepare_item is not None:
                    item = self.prepare_item(item)
                with self._lock:
                    self._items.append(item)
            except Exception as exc:
                with self._lock:
                    self._errors.append((item.get("idx"), str(exc)))
            finally:
                with self._lock:
                    self._pending -= 1
                self._q.task_done()

    def _deformation_worker(self) -> None:
        while not self._stop or not self._deformation_q.empty():
            try:
                record = self._deformation_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write_deformation(record)
            except Exception as exc:
                key = f"deformation:{record.get('source_timestamp_ns', 0)}"
                with self._lock:
                    self._errors.append((key, str(exc)))
            finally:
                with self._lock:
                    self._deformation_pending -= 1
                self._deformation_q.task_done()

    def video_status(self):
        with self._camera_lock:
            return {"backend": self.video_backend, "hardware_encode": self.video_backend == "hevc_nvenc",
                    "gop": 2 if self.video_backend == "hevc_nvenc" else 1,
                    "bitrate_mbps": dict(self.video_bitrates),
                    "cameras": {name: {**values, "queue_depth": self._camera_queues[name].qsize(),
                                       "error": self._camera_failed.get(name, "")}
                                for name, values in self._video_stats.items()}}

    def queue_status(self):
        with self._lock:
            return dict(structured=dict(depth=self._q.qsize(), pending=self._pending,
                                        capacity=self._q.maxsize),
                        deformation=dict(depth=self._deformation_q.qsize(), pending=self._deformation_pending,
                                         capacity=self._deformation_q.maxsize))

    def _start_raw_stream(self, name, width, height):
        camera_dir = self.episode_dir / "cameras" / name
        command = nvenc_command(self.ffmpeg, width, height, self.frequency,
                                self.video_bitrates[name], camera_dir / "rgb.mp4.tmp")
        (camera_dir / "encoder_command.json").write_text(json.dumps(command, indent=2))
        log = (camera_dir / "ffmpeg.log.tmp").open("wb")
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=log, bufsize=0, start_new_session=True)
            os.set_blocking(process.stdin.fileno(), False)
        except Exception:
            log.close()
            raise
        with self._camera_lock:
            self._camera_processes[name] = process
            self._camera_logs[name] = log
            self._camera_dimensions[name] = (width, height)
        return process

    def _start_camera_streams(self) -> None:
        if self.video_backend == "hevc_nvenc":
            return  # Open lazily with the first raw frame's actual dimensions.
        executable = shutil.which(self.ffmpeg)
        if executable is None:
            self._errors.append((None, f"ffmpeg not found: {self.ffmpeg}"))
            return
        for name in CAMERA_NAMES:
            camera_dir = self.episode_dir / "cameras" / name
            output_path = camera_dir / "rgb.mp4.tmp"
            log_path = camera_dir / "ffmpeg.log.tmp"
            command = [
                executable, "-hide_banner", "-loglevel", "warning", "-y",
                "-f", "image2pipe", "-vcodec", "mjpeg",
                "-framerate", f"{self.frequency:g}", "-i", "pipe:0",
                "-map", "0:v:0", "-an", "-c:v", "copy",
                "-movflags", "+faststart", "-f", "mp4", str(output_path),
            ]
            log_handle = None
            try:
                log_handle = log_path.open("wb")
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=log_handle,
                    start_new_session=True,
                )
                with self._camera_lock:
                    self._camera_processes[name] = process
                    self._camera_logs[name] = log_handle
            except Exception as exc:
                if log_handle is not None:
                    log_handle.close()
                self._errors.append((None, f"{name} MP4 start failed: {exc}"))

    def _write_camera_frame(self, record: dict) -> None:
        name = record["name"]
        if name in self._camera_failed:
            raise RuntimeError(self._camera_failed[name])
        with self._camera_lock:
            process = self._camera_processes.get(name)
        payload = record["jpeg"]
        if self.video_backend == "hevc_nvenc":
            width, height, payload = unpack_bgr(payload)
            if process is None:
                process = self._start_raw_stream(name, width, height)
            if self._camera_dimensions[name] != (width, height):
                raise RuntimeError(f"{name} resolution changed during recording")
        if process is None or process.stdin is None:
            raise RuntimeError(f"{name} MP4 stream is closed")
        if process.poll() is not None:
            raise RuntimeError(f"{name} ffmpeg exited with code {process.returncode}")
        # Each process has exactly one owning camera thread, so the potentially
        # blocking pipe write does not need the global camera metadata lock.
        began = time.monotonic()
        if self.video_backend == "hevc_nvenc":
            try:
                write_frame(process, payload)
            except Exception as exc:
                # A partial raw frame cannot be followed by another frame:
                # terminate the stream to avoid permanently shifted pixels.
                with self._camera_lock:
                    self._camera_failed[name] = str(exc)
                process.kill()
                raise
        else:
            process.stdin.write(payload)
        elapsed_ms = (time.monotonic()-began)*1000
        with self._camera_lock:
            stats = self._video_stats[name]
            stats["written"] += 1
            stats["write_ms_sum"] += elapsed_ms
            stats["write_ms_max"] = max(stats["write_ms_max"], elapsed_ms)
            stats["queue_wait_ms_max"] = max(stats["queue_wait_ms_max"], (began-record["enqueued_monotonic"])*1000)
            frame_index = len(self._camera_rows[name])
            self._camera_rows[name].append({
                "time_unix_ns": record["time_unix_ns"],
                "timeline_row": record["timeline_row"],
                "frame_index": frame_index,
                "source_timestamp_ns": record["source_timestamp_ns"],
                "byte_count": len(record["jpeg"]),
            })

    def _camera_worker(self, name: str) -> None:
        camera_queue = self._camera_queues[name]
        while not self._stop or not camera_queue.empty():
            try:
                record = camera_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write_camera_frame(record)
            except Exception as exc:
                key = (
                    f"camera:{record.get('name', '')}:"
                    f"{record.get('source_timestamp_ns', 0)}"
                )
                if self.video_backend == "hevc_nvenc":
                    with self._camera_lock:
                        self._camera_failed[name] = str(exc)
                        process = self._camera_processes.get(name)
                    if process is not None and process.poll() is None:
                        process.kill()
                with self._lock:
                    self._errors.append((key, str(exc)))
            finally:
                with self._lock:
                    self._camera_pending -= 1
                camera_queue.task_done()

    def _drain(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                no_pending_work = (
                    self._pending == 0
                    and self._deformation_pending == 0
                    and self._camera_pending == 0
                )
            queues_empty = (
                self._q.empty()
                and self._deformation_q.empty()
                and all(camera_queue.empty() for camera_queue in self._camera_queues.values())
            )
            if no_pending_work and queues_empty:
                return True
            time.sleep(0.01)
        return False

    def _discard_current_episode(self) -> None:
        path = self.episode_dir
        if path is not None and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        self.active = False
        self.last_result_dir = None
        self.last_validation = None

    def _finalize_failed_episode(
        self,
        quality: dict[str, Any],
        items: list[dict[str, Any]],
        *,
        write_raw_contract: bool,
    ) -> None:
        if self.episode_id is None:
            raise RuntimeError("cannot finalize an episode before it starts")
        # Preserve the same timestamp-only directory contract for every saved
        # recording.  Error state lives in the manifest and marker files, not
        # in the directory name.
        saved_dir = self.root / self.episode_id
        if self.episode_dir is not None and self.episode_dir.exists():
            self.episode_dir.rename(saved_dir)

        if write_raw_contract and self._raw_rows:
            try:
                self._write_raw_contract(saved_dir, items)
                payload = {"info": self._episode_info(), "data": items}
                (saved_dir / "data.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=4),
                    encoding="utf-8",
                )
            except Exception as exc:
                quality["errors"].append({
                    "index": None,
                    "message": f"error sample finalize: {exc}",
                })

        self._write_manifest(saved_dir, quality, "error")
        (saved_dir / "RECORDING_ERRORS.json").write_text(
            json.dumps(quality, indent=2),
            encoding="utf-8",
        )
        (saved_dir / "ERROR").touch()
        try:
            self.last_validation = validate_saved_episode(
                saved_dir,
                self.frequency,
                quality,
            )
        except Exception as exc:
            self.last_validation = {
                "available": False,
                "ok": False,
                "episode": saved_dir.name,
                "items": [],
                "errors": [f"validation failed: {exc}"],
            }
        self.last_result_dir = saved_dir
        self.episode_dir = saved_dir
        self.active = False

    def stop(
        self,
        save: bool = True,
        timeout: float = 30.0,
        dropped_frames: int = 0,
        quality_errors: list[str] | None = None,
        camera_input_dropped: dict[str, int] | None = None,
        boundary_frames: dict[str, int] | None = None,
        sampling_work: dict | None = None,
    ) -> Path | None:
        if not self.active:
            return None

        if self.message_journal is not None:
            if not self.message_journal.close(timeout):
                raise RuntimeError("raw message journal is still draining; retry stop before starting another episode")
            raw_status = self.message_journal.status()
            if not raw_status["complete"]:
                with self._lock:
                    self._errors.append((None, "raw message journal incomplete: " +
                                         "; ".join(raw_status["errors"])))
        else:
            raw_status = None
        drained = self._drain(timeout)
        if not drained:
            raise RuntimeError("writer queues still draining; retry stop before starting another episode")
        deformation_error = self._close_deformation_stream(save)
        camera_errors = self._close_camera_streams(save)
        with self._lock:
            errors = list(self._errors)
            items = sorted(self._items, key=lambda item: item["idx"])
        with self._deformation_lock:
            deformation_rows = list(self._deformation_rows)

        if deformation_error is not None:
            errors.append((None, deformation_error))
        errors.extend((None, error) for error in camera_errors)
        if self.video_backend == "hevc_nvenc":
            for name, count in (camera_input_dropped or {}).items():
                if count:
                    errors.append((None, f"{name} raw camera input lost/rejected {count} frame(s)"))
        if save and items and self.record_tactile and not deformation_rows:
            errors.append((None, "SharpA deformation stream is empty"))
        if save and items:
            with self._camera_lock:
                for name in CAMERA_NAMES:
                    if not self._camera_rows[name]:
                        errors.append((None, f"{name} camera stream is empty"))

        reported = [
            {"index": None, "message": str(error)}
            for error in (quality_errors or [])
        ]
        quality = {
            "ok": bool(save and drained and not errors and items),
            "errors": [
                {"index": index, "message": message}
                for index, message in errors
            ],
            "warnings": reported,
            "boundary_frames": dict(boundary_frames or {}),
            "frame_count": len(items),
            "timeline_rows": len(self._raw_rows),
            "camera_frames": self.camera_frame_counts(),
            "raw_messages": raw_status,
            "tactile_enabled": self.record_tactile,
            "sampling_work": dict(sampling_work or {}),
            "video_encoding": self.video_status(),
            "camera_input_dropped": dict(camera_input_dropped or {}),
            "dropped": (
                int(dropped_frames)
                + max(0, self._idx + 1 - len(items))
            ),
        }
        if not save:
            self._discard_current_episode()
            return None
        if not quality["ok"]:
            self._finalize_failed_episode(
                quality,
                items,
                write_raw_contract=True,
            )
            return None

        if self.episode_dir is None or self.episode_id is None:
            raise RuntimeError("active writer has no episode directory")
        payload = {"info": self._episode_info(), "data": items}
        try:
            self._write_raw_contract(self.episode_dir, items)
        except Exception as exc:
            quality["ok"] = False
            quality["errors"].append({
                "index": None,
                "message": f"raw contract: {exc}",
            })
            self._finalize_failed_episode(
                quality,
                items,
                write_raw_contract=False,
            )
            return None

        (self.episode_dir / "data.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        self._write_manifest(self.episode_dir, quality, "partial")
        complete = self.root / self.episode_id
        self.episode_dir.rename(complete)
        self._write_manifest(complete, quality, "complete")
        (complete / "COMPLETE").touch()
        self.last_validation = validate_saved_episode(complete, self.frequency, quality)
        # COMPLETE means the files were finalized. It must not make dropped,
        # duplicated or low-rate sensor data pass the manifest's quality gate.
        quality['files_ok'] = quality['ok']
        quality['ok'] = bool(self.last_validation['ok'])
        quality['capture_ok'] = quality['ok']
        if not quality['ok']:
            (complete / 'CAPTURE_QUALITY_FAILED').touch()
        self._write_manifest(complete, quality, 'complete')
        self._write_checksums(complete)
        self.active = False
        self.episode_dir = complete
        self.last_result_dir = complete
        return complete

    def classify(self, category: str) -> Path:
        """Move the last saved sample into root/success or root/fail."""
        category = str(category).strip().lower()
        if category not in {"success", "fail"}:
            raise ValueError("sample category must be success or fail")
        if self.active:
            raise RuntimeError("cannot classify while recording")
        source = self.last_result_dir
        if source is None or not source.is_dir():
            raise RuntimeError("no saved sample is waiting for classification")
        allowed_parents = {
            self.root.resolve(),
            (self.root / "success").resolve(),
            (self.root / "fail").resolve(),
        }
        if source.resolve().parent not in allowed_parents:
            raise RuntimeError("saved sample is outside the recording root")
        destination_dir = self.root / category
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if source.resolve() != destination.resolve():
            if destination.exists():
                raise RuntimeError(f"classified sample already exists: {destination}")
            source.rename(destination)
        from .execution_audit import write_termination
        termination_path = destination / 'termination.json'
        previous = json.loads(termination_path.read_text()) if termination_path.exists() else {}
        # Classification can happen much later than the physical end of an episode.
        outcome = 'success' if category == 'success' else 'failure'
        if previous.get('outcome') != outcome:
            previous = dict(previous, outcome=outcome, annotation_source='recording_classification',
                annotated_unix_ns=time.time_ns(), previous_annotation=previous)
            write_termination(destination, previous)
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["category"] = category
            manifest["updated_at"] = time.time()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        validation_path = destination / "validation.json"
        if validation_path.is_file():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        else:
            validation = dict(self.last_validation or {})
        validation["category"] = category
        validation["result_path"] = str(destination)
        validation_path.write_text(
            json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.last_validation = validation
        self.last_result_dir = destination
        self.episode_dir = destination
        if (destination / "checksums.sha256").is_file():
            self._write_checksums(destination)
        return destination

    def delete_last_saved(self) -> Path:
        """Permanently delete exactly the last saved sample directory."""
        if self.active:
            raise RuntimeError("cannot drop a sample while recording")
        source = self.last_result_dir
        if source is None or not source.is_dir():
            raise RuntimeError("no saved sample is waiting to be dropped")
        resolved = source.resolve()
        allowed_parents = {
            self.root.resolve(),
            (self.root / "success").resolve(),
            (self.root / "fail").resolve(),
        }
        if resolved.parent not in allowed_parents:
            raise RuntimeError("saved sample is outside the recording root")
        timestamp_name = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{6}"
        if re.fullmatch(timestamp_name, resolved.name) is None:
            raise RuntimeError("refusing to delete a non-sample directory")
        shutil.rmtree(resolved)
        self.last_result_dir = None
        self.last_validation = None
        self.episode_dir = None
        return resolved

    def _close_deformation_stream(self, save: bool) -> str | None:
        if not self.record_tactile:
            return None
        with self._deformation_lock:
            handle = self._deformation_handle
            self._deformation_handle = None
            if handle is not None:
                handle.flush()
                handle.close()
        if not save or self.episode_dir is None:
            return None
        raw_path = self.episode_dir / "sharpa" / "tactile" / "deform_u8.raw.tmp"
        compressed_path = self.episode_dir / "sharpa" / "tactile" / "deform_u8.zst"
        try:
            subprocess.run(
                ["zstd", "-3", "-q", "-f", str(raw_path), "-o", str(compressed_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            raw_path.unlink()
            return None
        except Exception as exc:
            return f"SharpA deformation compression failed: {exc}"

    def _close_camera_streams(self, save: bool) -> list[str]:
        errors = []
        with self._camera_lock:
            processes = dict(self._camera_processes)
            logs = dict(self._camera_logs)
            self._camera_processes.clear()
            self._camera_logs.clear()
        for name in CAMERA_NAMES:
            process = processes.get(name)
            log_handle = logs.get(name)
            returncode = None
            try:
                if process is not None:
                    if process.stdin is not None:
                        try:
                            process.stdin.close()
                        except (BrokenPipeError, OSError):
                            pass
                    try:
                        returncode = process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            returncode = process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            returncode = process.wait(timeout=2.0)
            except Exception as exc:
                errors.append(f"{name} MP4 stop failed: {exc}")
            finally:
                if log_handle is not None:
                    log_handle.close()
            if self.episode_dir is None:
                continue
            camera_dir = self.episode_dir / "cameras" / name
            log_tmp = camera_dir / "ffmpeg.log.tmp"
            log_path = camera_dir / "ffmpeg.log"
            if log_tmp.exists():
                log_tmp.rename(log_path)
            if not save:
                continue
            tmp_path = camera_dir / "rgb.mp4.tmp"
            final_path = camera_dir / "rgb.mp4"
            if returncode == 0 and tmp_path.is_file() and tmp_path.stat().st_size > 0:
                tmp_path.rename(final_path)
                try:
                    probe = self._probe_camera_video(final_path)
                    with self._camera_lock:
                        self._camera_probe[name] = probe
                        expected_frames = len(self._camera_rows[name])
                    if probe["frame_count"] != expected_frames:
                        errors.append(
                            f"{name} MP4 frame mismatch: "
                            f"video={probe['frame_count']}, index={expected_frames}"
                        )
                    if probe["codec_name"] != self.video_codec:
                        errors.append(f"{name} MP4 codec mismatch: {probe['codec_name']}")
                except Exception as exc:
                    errors.append(f"{name} MP4 validation failed: {exc}")
            else:
                size = tmp_path.stat().st_size if tmp_path.is_file() else 0
                errors.append(f"{name} MP4 finalize failed: returncode={returncode}, bytes={size}")
        return errors

    def _probe_camera_video(self, path: Path) -> dict[str, Any]:
        executable = shutil.which(self.ffprobe)
        if executable is None:
            raise RuntimeError(f"ffprobe not found: {self.ffprobe}")
        result = subprocess.run(
            [
                executable, "-v", "error", "-err_detect", "explode",
                "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stderr.strip():
            raise RuntimeError("video decode errors: " + result.stderr[-2000:])
        streams = json.loads(result.stdout).get("streams", [])
        if len(streams) != 1:
            raise RuntimeError(f"expected one video stream, got {len(streams)}")
        stream = streams[0]
        count = stream.get("nb_read_frames", stream.get("nb_frames", 0))
        return {
            "codec_name": str(stream.get("codec_name", "")),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "avg_frame_rate": str(stream.get("avg_frame_rate", "")),
            "frame_count": int(count),
        }

    def camera_frame_counts(self) -> dict[str, int]:
        with self._camera_lock:
            return {name: len(self._camera_rows[name]) for name in CAMERA_NAMES}

    def _write_manifest(
        self,
        directory: Path,
        quality: dict[str, Any],
        state: str,
    ) -> None:
        payload = {
            "schema": "ur_sharpa_sample.v1",
            "state": state,
            "sample_rate_hz": self.frequency,
            "quality": quality,
            "updated_at": time.time(),
        }
        (directory / "manifest.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _write_raw_contract(self, directory, items):
        raw_rows = list(self._raw_rows)
        time_unix_ns = np.asarray([x["time_unix_ns"] for x in raw_rows], dtype=np.int64)
        np.savez_compressed(directory / "timeline" / "clock.npz",
                            time_unix_ns=time_unix_ns,
                            sampled_unix_ns=np.asarray([x.get("sampled_unix_ns", x["time_unix_ns"])
                                                        for x in raw_rows], dtype=np.int64),
                            elapsed_ns=time_unix_ns - time_unix_ns[0],
                            sample_rate_hz=np.asarray(self.frequency, dtype=np.float64),
                            **({"is_hil": np.asarray([x["is_hil"] for x in raw_rows], dtype=bool)}
                               if "is_hil" in raw_rows[0] else {}))
        source_names = np.asarray(raw_rows[0]["source_names"])
        source_valid = np.asarray([x["source_valid"] for x in raw_rows], dtype=bool)
        receive_age_ms = np.asarray([x["source_age_ms"] for x in raw_rows], dtype=np.float32)
        source_timestamp_ns = np.asarray([x["source_timestamp_ns"] for x in raw_rows], dtype=np.int64)
        source_age_ms = ((time_unix_ns[:, None]-source_timestamp_ns)/1e6).astype(np.float32)
        source_age_ms[source_timestamp_ns <= 0] = np.nan
        np.savez_compressed(directory / "timeline" / "source_quality.npz",
                            source_names=source_names, valid=source_valid, age_ms=receive_age_ms,
                            receive_age_ms=receive_age_ms, source_age_ms=source_age_ms,
                            source_timestamp_ns=source_timestamp_ns,
                            enabled=np.asarray([True, True, True, True,
                                                self.record_tactile, self.record_tactile]))
        arm_q = np.asarray([x["ur_q"] for x in raw_rows], dtype=np.float32)
        wrist = np.asarray([x["ur_wrist"] for x in raw_rows], dtype=np.float32)
        tcp = np.asarray([x["ur_tcp"] for x in raw_rows], dtype=np.float32)
        np.savez_compressed(directory / "ur" / "physical_12.npz", q=arm_q,
                            raw_tcp_pose6=tcp, canonical_wrist_pose9=wrist,
                            timestamp_ns=source_timestamp_ns[:,1], valid=source_valid[:,1])
        hand_q = np.asarray([x["hand_q"] for x in raw_rows], dtype=np.float32)
        hand_q_cmd = np.asarray([x["hand_q_cmd"] for x in raw_rows], dtype=np.float32)
        hand_q_cmd_valid = np.asarray([x["hand_q_cmd_valid"] for x in raw_rows], dtype=bool)
        hand_q_cmd_timestamp_ns = np.asarray([x["hand_q_cmd_timestamp_ns"] for x in raw_rows], dtype=np.int64)
        hand_q_cmd_source = np.asarray([x["hand_q_cmd_source"] for x in raw_rows], dtype=np.uint8)
        hand_q_cmd_invalid_reason = np.asarray(
            [x.get("hand_q_cmd_invalid_reason", "") for x in raw_rows], dtype="U32")
        hand_tau = np.asarray([x["hand_tau"] for x in raw_rows], dtype=np.float32)
        np.savez_compressed(directory / "sharpa" / "joints_44.npz", q=hand_q,
                            q_cmd=hand_q_cmd, q_cmd_valid=hand_q_cmd_valid,
                            q_cmd_timestamp_ns=hand_q_cmd_timestamp_ns,
                            q_cmd_source=hand_q_cmd_source, tau=hand_tau,
                            q_cmd_invalid_reason=hand_q_cmd_invalid_reason,
                            q_timestamp_ns=source_timestamp_ns[:,2], tau_timestamp_ns=source_timestamp_ns[:,3],
                            q_valid=source_valid[:,2], tau_valid=source_valid[:,3])
        if self.record_tactile:
            wrench = np.asarray([x["wrench"] for x in raw_rows], dtype=np.float32)
            wrench_valid = np.asarray([x["wrench_valid"] for x in raw_rows], dtype=bool)
            np.savez_compressed(directory / "sharpa" / "tactile" / "force6d.npz",
                                wrench=wrench, force=wrench[..., :3], torque=wrench[..., 3:],
                                timestamp_ns=source_timestamp_ns[:,4],
                                valid=wrench_valid)
        with self._camera_lock:
            camera_rows = {name:list(self._camera_rows[name]) for name in CAMERA_NAMES}
            camera_probe = {name:dict(self._camera_probe.get(name,{})) for name in CAMERA_NAMES}
        camera_schema = {}
        for camera_name in CAMERA_NAMES:
            rows = camera_rows[camera_name]
            camera_dir = directory / "cameras" / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            camera_times = np.asarray([row["time_unix_ns"] for row in rows], dtype=np.int64)
            source_timestamps = np.asarray([row["source_timestamp_ns"] for row in rows], dtype=np.int64)
            frame_indices = np.asarray([row["frame_index"] for row in rows], dtype=np.int32)
            timeline_rows = np.asarray([row["timeline_row"] for row in rows], dtype=np.int32)
            byte_counts = np.asarray([row["byte_count"] for row in rows], dtype=np.int32)
            np.savez_compressed(
                camera_dir / "index.npz",
                time_unix_ns=camera_times,
                elapsed_ns=camera_times - self._started_unix_ns,
                timeline_row=timeline_rows,
                frame_index=frame_indices,
                source_timestamp_ns=source_timestamps,
                byte_count=byte_counts,
                valid=np.ones(len(rows), dtype=bool),
            )
            achieved_rate = 0.0
            if len(source_timestamps) > 1 and source_timestamps[-1] > source_timestamps[0]:
                achieved_rate = float((len(source_timestamps)-1)*1e9/(source_timestamps[-1]-source_timestamps[0]))
            video_path = camera_dir / "rgb.mp4"
            meta = {
                "owner": "recording_node",
                "source_encoding": self.source_encoding,
                "container": "mp4",
                "video_codec": self.video_codec,
                "encoder": self.video_backend,
                "hardware_encode": self.video_backend == "hevc_nvenc",
                "gop": 2 if self.video_backend == "hevc_nvenc" else 1,
                "b_frames": 0,
                "bitrate_mbps": self.video_bitrates[camera_name] if self.video_backend == "hevc_nvenc" else None,
                "byte_count_scope": "input_payload_bytes",

                "ffmpeg": self.ffmpeg,
                "path": "rgb.mp4",
                "index_path": "index.npz",
                "frame_count": len(rows),
                "nominal_fps": self.frequency,
                "source_rate_hz": round(achieved_rate, 6),
                "bytes": video_path.stat().st_size if video_path.is_file() else 0,
                "ok": video_path.is_file() and video_path.stat().st_size > 0 and bool(rows),
                "probe": camera_probe[camera_name],
            }
            (camera_dir / "rgb_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            camera_schema[camera_name] = {
                "rgb_file": f"cameras/{camera_name}/rgb.mp4",
                "index_file": f"cameras/{camera_name}/index.npz",
                "meta_file": f"cameras/{camera_name}/rgb_meta.json",
            }
        if self.record_tactile:
            with self._deformation_lock:
                deformation_rows = list(self._deformation_rows)
            deformation_times = np.asarray(
                [row["time_unix_ns"] for row in deformation_rows],
                dtype=np.int64,
            )
            deformation_timeline_rows = np.asarray([row["timeline_row"] for row in deformation_rows], dtype=np.int32)
            deformation_raw_indices = np.asarray([row["raw_frame_index"] for row in deformation_rows], dtype=np.int32)
            deformation_timestamps = np.asarray([row["source_timestamp_ns"] for row in deformation_rows], dtype=np.int64)
            deformation_valid = np.asarray([row["valid"] for row in deformation_rows], dtype=bool).reshape(-1, 10)
            np.savez_compressed(directory / "sharpa" / "tactile" / "deform_index.npz",
                                time_unix_ns=deformation_times,
                                elapsed_ns=deformation_times - self._started_unix_ns,
                                timeline_row=deformation_timeline_rows,
                                raw_frame_index=deformation_raw_indices,
                                source_timestamp_ns=deformation_timestamps,
                                sensor_time=np.repeat((deformation_timestamps.astype(np.float64) / 1e9)[:,None], 10, axis=1),
                                valid=deformation_valid)
        schema = {
            "schema": "ur_sharpa_sample.v1", "sample_rate_hz": self.frequency,
            "wrist_pose9": "[x,y,z,r00,r10,r20,r01,r11,r21] canonical root-from-wrist",
            "raw_tcp_pose6": "[x,y,z,rx,ry,rz] UR RTDE base TCP",
            "hand_action": "last fresh full 44D target successfully dispatched by sharpa_node",
            "hand_action_source": "HandCommand.SOURCE_*; teleop requires SOURCE_VIVE (AE01); rollout requires SOURCE_ACTOR",
            "camera_views": list(CAMERA_NAMES),
            "cameras": camera_schema,
            "tactile_finger_order": ["pinky", "ring", "middle", "index", "thumb"],
            "tactile_wrench": {
                "enabled": self.record_tactile, "file": "sharpa/tactile/force6d.npz",
                "wrench_shape": ["T", 2, 5, 6], "force_shape": ["T", 2, 5, 3],
                "torque_shape": ["T", 2, 5, 3], "hand_order": ["left", "right"],
                "axes": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "force_unit": "N", "torque_unit": "N*m", "frame": "fingertip-local",
                "invalid": "zero placeholder; read valid[T,2,5]",
                "timestamp_ns": "original /sharpa/wrench publication timestamp; aligned to timeline rows",
            },
            "tactile_deform": {
                "data_file": "sharpa/tactile/deform_u8.zst",
                "index_file": "sharpa/tactile/deform_index.npz",
                "raw_logical_shape": [-1, 10, 240, 240],
                "plane_order": [
                    "right_pinky", "right_ring", "right_middle", "right_index", "right_thumb",
                    "left_pinky", "left_ring", "left_middle", "left_index", "left_thumb",
                ],
                "dtype": "uint8",
                "compression": "zstd",
            },
        }
        schema["tactile_enabled"] = self.record_tactile
        schema["timeline_clock"] = {
            "time_unix_ns": "sampling target time; aligned mode uses a fixed-rate grid",
            "sampled_unix_ns": "actual row assembly time, stored separately from source timestamps",
        }
        if "is_hil" in raw_rows[0]:
            schema["is_hil"] = "metadata.is_hil and timeline/clock.npz is_hil[T]: HIL flag of paired successfully dispatched hand command; invalid rows are false"
            schema["hand_action"] = "last full 44D Actor target successfully dispatched by sharpa_node"
            schema["hand_action_source"] = "HandCommand.SOURCE_ACTOR; policy and HIL share this outlet"
        if not self.record_tactile:
            schema["tactile_deform"] = {"enabled": False}
        (directory / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    @staticmethod
    def _write_checksums(directory):
        rows = []
        for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "checksums.sha256"):
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(chunk)
            digest = checksum.hexdigest()
            rows.append(f"{digest}  {path.relative_to(directory)}")
        (directory / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def discard(self) -> None:
        self.stop(save=False)

    def close(self) -> None:
        if self.message_journal is not None and not self.message_journal.status()["closed"]:
            if not self.message_journal.close(30):
                raise RuntimeError("raw message journal is still draining")
        self._stop = True
        self._drain(2)
        self._close_deformation_stream(False)
        self._close_camera_streams(False)
        for thread in self._threads:
            thread.join(timeout=1)
        self._deformation_thread.join(timeout=1)
        for thread in self._camera_threads:
            thread.join(timeout=1)
