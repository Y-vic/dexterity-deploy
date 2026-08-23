"""PND-style asynchronous episode writer.

Structured samples, tactile deformation, and the three camera streams use
independent queues. Camera JPEG payloads are remuxed into MP4 without image
re-encoding, so ROS callbacks never wait for FFmpeg or filesystem I/O.
"""
from __future__ import annotations

import datetime
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

CAMERA_NAMES = ("ego", "left_wrist", "right_wrist")
CAMERA_COLOR_KEYS = ("color_0", "color_1", "color_2")
FINGER_NAMES = ("小指", "无名指", "中指", "食指", "拇指")
MIN_VALID_RATE_HZ = 25.0


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
    """Read the finalized files and return the eight-item Monitor summary."""
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

    def item(key, label, frames, *, missing=0, details=None, files_ok=True,
             finger_missing=None):
        details = list(details or [])
        finger_missing = list(finger_missing or [])
        missing = max(0, int(missing))
        if not files_ok:
            details.insert(0, "file missing")
        rate = _validation_rate(int(frames), duration_s)
        rate_ok = bool(rate >= MIN_VALID_RATE_HZ)
        if not rate_ok:
            details.append(f"低于 {MIN_VALID_RATE_HZ:g} Hz")
        details.extend(
            f"{row['finger']} {row['ratio_percent']:.2f}%"
            for row in finger_missing
        )
        ok = bool(files_ok and rate_ok and not finger_missing and not any(
            text in {"valid shape invalid", "video missing", "file missing"}
            for text in details
        ))
        return {
            "key": key,
            "label": label,
            "frames": int(frames),
            "duration_s": round(duration_s, 3),
            "hz": rate,
            "minimum_hz": MIN_VALID_RATE_HZ,
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
        details = []
        if index_path.is_file():
            index = np.load(index_path, allow_pickle=False)
            valid = np.asarray(index["valid"], dtype=bool)
            count = int(np.count_nonzero(valid))
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            details.append("video missing")
        missing = max(0, timeline_rows - count)
        items.append(item(f"{name}_camera", label, count, missing=missing, details=details,
                          files_ok=index_path.is_file()))

    arm_path = directory / "ur" / "physical_12.npz"
    arm_count = 0
    if arm_path.is_file():
        arm = np.load(arm_path, allow_pickle=False)
        arm_count = int(np.count_nonzero(np.asarray(arm["valid"], dtype=bool)))
    items.append(item("arm_state", "Arm State", arm_count,
                      missing=timeline_rows-arm_count, files_ok=arm_path.is_file()))

    joints_path = directory / "sharpa" / "joints_44.npz"
    if joints_path.is_file():
        joints = np.load(joints_path, allow_pickle=False)
        hand_valid = np.asarray(joints["q_valid"], dtype=bool)
        action_valid = np.asarray(joints["q_cmd_valid"], dtype=bool)
        if "q_cmd_source" in joints:
            action_valid &= np.asarray(joints["q_cmd_source"], dtype=np.uint8) == 2
        hand_count = int(np.count_nonzero(hand_valid))
        action_count = int(np.count_nonzero(action_valid))
    else:
        hand_count = action_count = 0
    items.append(item("hand_state", "Hand State", hand_count,
                      missing=timeline_rows-hand_count, files_ok=joints_path.is_file()))
    items.append(item("hand_action", "Hand Action", action_count,
                      missing=timeline_rows-action_count, files_ok=joints_path.is_file()))

    wrench_path = directory / "sharpa" / "tactile" / "force6d.npz"
    wrench_count = 0
    wrench_details = []
    wrench_finger_missing = []
    if wrench_path.is_file():
        wrench = np.load(wrench_path, allow_pickle=False)
        wrench_valid = np.asarray(wrench["valid"], dtype=bool)
        if wrench_valid.ndim == 3 and wrench_valid.shape[1:] == (2, 5):
            wrench_count = int(np.count_nonzero(np.all(wrench_valid, axis=(1, 2))))
            wrench_finger_missing = _missing_finger_summary(
                wrench_valid.reshape(-1, 10),
                tuple(f"{side}{finger}" for side in ("左", "右") for finger in FINGER_NAMES),
            )
        else:
            wrench_details = ["valid shape invalid"]
    items.append(item("wrench", "Wrench", wrench_count,
                      missing=timeline_rows-wrench_count, details=wrench_details,
                      files_ok=wrench_path.is_file(),
                      finger_missing=wrench_finger_missing))

    deform_path = directory / "sharpa" / "tactile" / "deform_index.npz"
    deform_count = 0
    deform_details = []
    deform_finger_missing = []
    if deform_path.is_file():
        deform = np.load(deform_path, allow_pickle=False)
        deform_valid = np.asarray(deform["valid"], dtype=bool)
        if deform_valid.ndim == 2 and deform_valid.shape[1] == 10:
            deform_count = int(np.count_nonzero(np.all(deform_valid, axis=1)))
            deform_finger_missing = _missing_finger_summary(
                deform_valid,
                tuple(f"{side}{finger}" for side in ("右", "左") for finger in FINGER_NAMES),
            )
        else:
            deform_details = ["valid shape invalid"]
    items.append(item("deform", "Deform", deform_count,
                      missing=timeline_rows-deform_count, details=deform_details,
                      files_ok=deform_path.is_file(),
                      finger_missing=deform_finger_missing))

    quality_errors = [str(row.get("message", "")) for row in quality.get("errors", []) if row.get("message")]
    result = {
        "available": True,
        "episode": directory.name,
        "duration_s": round(duration_s, 3),
        "timeline_rows": timeline_rows,
        "minimum_hz": MIN_VALID_RATE_HZ,
        "ok": bool(quality.get("ok", False) and all(row["ok"] for row in items)),
        "items": items,
        "errors": quality_errors,
    }
    (directory / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _default_info(frequency: float, image_size=(640, 480)) -> dict:
    return {
        "version": "1.0.0", "date": datetime.date.today().isoformat(), "author": "unitree",
        "image": {"width": int(image_size[0]), "height": int(image_size[1]), "fps": frequency},
        "depth": {"width": int(image_size[0]), "height": int(image_size[1]), "fps": frequency},
        "audio": {"sample_rate": 16000, "channels": 1, "format": "PCM", "bits": 16},
        "tactile": {"shape": [1140, 3], "dtype": "int16", "fps": frequency},
        "joint_names": {"left_arm": ["kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristyaw"], "left_hand": [], "right_arm": [], "right_hand": [], "body": []},
        "tactile_names": {"left_hand": [], "right_hand": []},
    }


class EpisodeWriterCompat:
    """Asynchronous, all-or-nothing episode writer.

    ``add_item`` is non-blocking.  If its bounded queue is full it returns
    ``False`` and the recorder reports a dropped frame; it never stalls ROS.
    ``discard`` drains the queue and removes the whole episode directory.
    """

    def __init__(self, root: str | Path, frequency=30.0, workers: int | None = None,
                 queue_size=100, ffmpeg="ffmpeg"):
        self.root = Path(root).expanduser(); self.frequency = float(frequency)
        self.workers = max(1, int(workers or os.environ.get("EPISODE_WRITER_THREADS", "4")))
        self.ffmpeg = str(ffmpeg).strip() or "ffmpeg"; self.ffprobe=os.environ.get("EPISODE_WRITER_FFPROBE","ffprobe")
        queue_size=max(1, int(queue_size)); self._q = queue.Queue(maxsize=queue_size); self._deformation_q=queue.Queue(maxsize=queue_size); self._camera_q=queue.Queue(maxsize=queue_size*3); self._threads=[]; self._stop=False
        self._lock=threading.RLock(); self._pending=0; self._deformation_pending=0; self._camera_pending=0; self._errors=[]; self._items=[]; self._raw_rows=[]; self._idx=-1
        self._deformation_lock=threading.Lock(); self._deformation_handle=None; self._deformation_frames=0; self._deformation_rows=[]; self._started_unix_ns=0
        self._camera_lock=threading.Lock(); self._camera_processes={}; self._camera_logs={}; self._camera_rows={name:[] for name in CAMERA_NAMES}; self._camera_probe={}
        self.active=False; self.episode_dir: Path | None=None; self.episode_id=None
        self.last_result_dir: Path | None = None
        self.last_validation: dict | None = None
        for _ in range(self.workers):
            t=threading.Thread(target=self._worker, daemon=True); t.start(); self._threads.append(t)
        self._deformation_thread=threading.Thread(target=self._deformation_worker, daemon=True); self._deformation_thread.start()
        self._camera_thread=threading.Thread(target=self._camera_worker, daemon=True); self._camera_thread.start()

    def _next_id(self):
        ids=[]
        for parent in (self.root, self.root / "success", self.root / "fail"):
            if not parent.exists():
                continue
            for p in parent.iterdir():
                if p.is_dir() and (p.name.startswith("sample_") and ".error_" not in p.name):
                    try: ids.append(int(p.name.split("_")[1].split(".")[0]))
                    except Exception: pass
        return max(ids, default=-1)+1

    def start(self):
        if self.active: return False
        self.last_result_dir = None; self.last_validation = None
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "success").mkdir(exist_ok=True)
        (self.root / "fail").mkdir(exist_ok=True)
        self.episode_id=self._next_id(); self.episode_dir=self.root/f"sample_{self.episode_id:04d}.partial"
        for d in ("timeline", "ur", "sharpa", "sharpa/tactile", "cameras"):
            (self.episode_dir/d).mkdir(parents=True, exist_ok=True)
        for name in CAMERA_NAMES:
            (self.episode_dir/"cameras"/name).mkdir(parents=True, exist_ok=True)
        self._deformation_handle=(self.episode_dir/"sharpa"/"tactile"/"deform_u8.raw.tmp").open("wb")
        self._deformation_frames=0; self._deformation_rows=[]; self._started_unix_ns=time.time_ns()
        self._idx=-1; self._items=[]; self._raw_rows=[]; self._errors=[]; self._camera_rows={name:[] for name in CAMERA_NAMES}; self._camera_probe={}
        self._start_camera_streams(); self.active=True
        return True

    def add_raw_row(self, row):
        if self.active:
            with self._lock: self._raw_rows.append(row)

    def add_item(self, item: dict) -> bool:
        if not self.active: return False
        with self._lock: self._idx += 1; item=dict(item); item["idx"]=self._idx; self._pending += 1
        try: self._q.put_nowait(item); return True
        except queue.Full:
            with self._lock: self._pending -= 1
            return False

    def add_deformation(self, msg: Any, timeline_row: int, time_unix_ns: int) -> bool:
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
            if name not in self._camera_processes:
                return False
        record = {
            "name": name,
            "jpeg": bytes(jpeg),
            "source_timestamp_ns": int(source_timestamp_ns),
            "timeline_row": int(timeline_row),
            "time_unix_ns": int(time_unix_ns),
        }
        if not record["jpeg"] or record["source_timestamp_ns"] <= 0:
            return False
        with self._lock:
            self._camera_pending += 1
        try:
            self._camera_q.put_nowait(record)
            return True
        except queue.Full:
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

    def _worker(self):
        while not self._stop or not self._q.empty():
            try: item=self._q.get(timeout=.1)
            except queue.Empty: continue
            try:
                with self._lock: self._items.append(item)
            except Exception as exc:
                with self._lock: self._errors.append((item.get("idx"),str(exc)))
            finally:
                with self._lock: self._pending -= 1
                self._q.task_done()

    def _deformation_worker(self):
        while not self._stop or not self._deformation_q.empty():
            try: record=self._deformation_q.get(timeout=.1)
            except queue.Empty: continue
            try:
                self._write_deformation(record)
            except Exception as exc:
                with self._lock: self._errors.append((f"deformation:{record.get('source_timestamp_ns', 0)}",str(exc)))
            finally:
                with self._lock: self._deformation_pending -= 1
                self._deformation_q.task_done()

    def _start_camera_streams(self) -> None:
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
        with self._camera_lock:
            process = self._camera_processes.get(name)
            if process is None or process.stdin is None:
                raise RuntimeError(f"{name} MP4 stream is closed")
            if process.poll() is not None:
                raise RuntimeError(f"{name} ffmpeg exited with code {process.returncode}")
            process.stdin.write(record["jpeg"])
            frame_index = len(self._camera_rows[name])
            self._camera_rows[name].append({
                "time_unix_ns": record["time_unix_ns"],
                "timeline_row": record["timeline_row"],
                "frame_index": frame_index,
                "source_timestamp_ns": record["source_timestamp_ns"],
                "byte_count": len(record["jpeg"]),
            })

    def _camera_worker(self):
        while not self._stop or not self._camera_q.empty():
            try: record=self._camera_q.get(timeout=.1)
            except queue.Empty: continue
            try:
                self._write_camera_frame(record)
            except Exception as exc:
                with self._lock: self._errors.append((f"camera:{record.get('name', '')}:{record.get('source_timestamp_ns', 0)}",str(exc)))
            finally:
                with self._lock: self._camera_pending -= 1
                self._camera_q.task_done()

    def _drain(self, timeout=30):
        end=__import__('time').monotonic()+timeout
        while __import__('time').monotonic()<end:
            with self._lock: pending=self._pending; deformation_pending=self._deformation_pending; camera_pending=self._camera_pending
            if pending==0 and deformation_pending==0 and camera_pending==0 and self._q.empty() and self._deformation_q.empty() and self._camera_q.empty(): return True
            __import__('time').sleep(.01)
        return False

    def stop(self, save=True, timeout=30, dropped_frames=0, quality_errors=None):
        if not self.active: return None
        drained=self._drain(timeout)
        deformation_error = self._close_deformation_stream(save)
        camera_errors = self._close_camera_streams(save)
        with self._lock: errors=list(self._errors); items=sorted(self._items,key=lambda x:x["idx"])
        with self._deformation_lock: deformation_rows=list(self._deformation_rows)
        if deformation_error is not None:
            errors.append((None, deformation_error))
        errors.extend((None, error) for error in camera_errors)
        if save and items and not deformation_rows:
            errors.append((None, "SharpA deformation stream is empty"))
        if save and items:
            with self._camera_lock:
                for name in CAMERA_NAMES:
                    if not self._camera_rows[name]:
                        errors.append((None, f"{name} camera stream is empty"))
        reported = [{"index": None, "message": str(e)} for e in (quality_errors or [])]
        quality = {"ok": bool(save and drained and not errors and bool(items)),
                   "errors": [{"index": i, "message": e} for i, e in errors],
                   "warnings": reported,
                   "frame_count": len(items), "timeline_rows": len(self._raw_rows),
                   "camera_frames": self.camera_frame_counts(),
                   "dropped": int(dropped_frames) + max(0, self._idx + 1 - len(items))}
        if not save:
            path = self.episode_dir
            if path and path.exists(): shutil.rmtree(path, ignore_errors=True)
            self.active = False; self.last_result_dir = None; self.last_validation = None; return None
        if not quality["ok"]:
            error_dir = self._error_path()
            if self.episode_dir and self.episode_dir.exists():
                self.episode_dir.rename(error_dir)
            if error_dir:
                if self._raw_rows:
                    try:
                        self._write_raw_contract(error_dir, items)
                        payload = {"info": _default_info(self.frequency), "data": items}
                        (error_dir / "data.json").write_text(
                            json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
                    except Exception as exc:
                        quality["errors"].append({"index": None, "message": f"error sample finalize: {exc}"})
                self._write_manifest(error_dir, quality, "error")
                (error_dir / "RECORDING_ERRORS.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
                (error_dir / "ERROR").touch()
                try:
                    self.last_validation = validate_saved_episode(error_dir, self.frequency, quality)
                except Exception as exc:
                    self.last_validation = {"available": False, "ok": False, "episode": error_dir.name, "items": [], "errors": [f"validation failed: {exc}"]}
                self.last_result_dir = error_dir
                self.episode_dir = error_dir
            self.active=False; return None
        payload={"info":_default_info(self.frequency),"data":items}
        try:
            self._write_raw_contract(self.episode_dir, items)
        except Exception as exc:
            error_dir = self._error_path()
            if self.episode_dir and self.episode_dir.exists():
                self.episode_dir.rename(error_dir)
            quality["ok"] = False; quality["errors"].append({"index": None, "message": f"raw contract: {exc}"})
            self._write_manifest(error_dir, quality, "error")
            (error_dir / "RECORDING_ERRORS.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
            (error_dir / "ERROR").touch()
            self.last_result_dir = error_dir
            self.episode_dir = error_dir
            try:
                self.last_validation = validate_saved_episode(error_dir, self.frequency, quality)
            except Exception as validation_exc:
                self.last_validation = {"available": False, "ok": False, "episode": error_dir.name, "items": [], "errors": [f"validation failed: {validation_exc}"]}
            self.active = False; return None
        (self.episode_dir/"data.json").write_text(json.dumps(payload,ensure_ascii=False,indent=4),encoding="utf-8")
        self._write_manifest(self.episode_dir, quality, "partial")
        complete = self.root / f"sample_{self.episode_id:04d}"
        self.episode_dir.rename(complete)
        self._write_manifest(complete, quality, "complete")
        (complete / "COMPLETE").touch()
        self.last_validation = validate_saved_episode(complete, self.frequency, quality)
        self._write_checksums(complete)
        self.active=False; self.episode_dir=complete; self.last_result_dir=complete; return complete

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
        if re.fullmatch(r"sample_\d+(?:\.error_\d+)?", resolved.name) is None:
            raise RuntimeError("refusing to delete a non-sample directory")
        shutil.rmtree(resolved)
        self.last_result_dir = None
        self.last_validation = None
        self.episode_dir = None
        return resolved

    def _close_deformation_stream(self, save: bool) -> str | None:
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
                        errors.append(f"{name} MP4 frame mismatch: video={probe['frame_count']}, index={expected_frames}")
                    if probe["codec_name"] != "mjpeg":
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
                executable, "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
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

    def _error_path(self):
        base = self.root / f"sample_{self.episode_id:04d}"
        index = 1
        while (self.root / f"{base.name}.error_{index:03d}").exists():
            index += 1
        return self.root / f"{base.name}.error_{index:03d}"

    def _write_manifest(self, directory, quality, state):
        payload = {"schema": "ur_sharpa_sample.v1", "state": state,
                   "sample_rate_hz": self.frequency, "quality": quality,
                   "updated_at": time.time()}
        (directory / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_raw_contract(self, directory, items):
        raw_rows = list(self._raw_rows)
        time_unix_ns = np.asarray([x["time_unix_ns"] for x in raw_rows], dtype=np.int64)
        np.savez_compressed(directory / "timeline" / "clock.npz",
                            time_unix_ns=time_unix_ns,
                            elapsed_ns=time_unix_ns - time_unix_ns[0],
                            sample_rate_hz=np.asarray(self.frequency, dtype=np.float64))
        source_names = np.asarray(raw_rows[0]["source_names"])
        source_valid = np.asarray([x["source_valid"] for x in raw_rows], dtype=bool)
        source_age_ms = np.asarray([x["source_age_ms"] for x in raw_rows], dtype=np.float32)
        source_timestamp_ns = np.asarray([x["source_timestamp_ns"] for x in raw_rows], dtype=np.int64)
        np.savez_compressed(directory / "timeline" / "source_quality.npz",
                            source_names=source_names, valid=source_valid, age_ms=source_age_ms,
                            source_timestamp_ns=source_timestamp_ns)
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
        hand_tau = np.asarray([x["hand_tau"] for x in raw_rows], dtype=np.float32)
        np.savez_compressed(directory / "sharpa" / "joints_44.npz", q=hand_q,
                            q_cmd=hand_q_cmd, q_cmd_valid=hand_q_cmd_valid,
                            q_cmd_timestamp_ns=hand_q_cmd_timestamp_ns,
                            q_cmd_source=hand_q_cmd_source, tau=hand_tau,
                            q_timestamp_ns=source_timestamp_ns[:,2], tau_timestamp_ns=source_timestamp_ns[:,3],
                            q_valid=source_valid[:,2], tau_valid=source_valid[:,3])
        wrench = np.asarray([x["wrench"] for x in raw_rows], dtype=np.float32)
        wrench_valid = np.asarray([x["wrench_valid"] for x in raw_rows], dtype=bool)
        np.savez_compressed(directory / "sharpa" / "tactile" / "force6d.npz",
                            wrench=wrench, timestamp_ns=source_timestamp_ns[:,4],
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
                "source_encoding": "jpeg",
                "container": "mp4",
                "video_codec": "mjpeg",
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
        with self._deformation_lock:
            deformation_rows = list(self._deformation_rows)
        deformation_times = np.asarray([row["time_unix_ns"] for row in deformation_rows], dtype=np.int64)
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
            "hand_action": "last fresh full 44D Manus teleop target received by sharpa_node",
            "hand_action_source": "HandCommand.SOURCE_*; teleop recording requires SOURCE_VIVE (Manus)",
            "camera_views": list(CAMERA_NAMES),
            "cameras": camera_schema,
            "tactile_finger_order": ["pinky", "ring", "middle", "index", "thumb"],
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
        (directory / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    @staticmethod
    def _write_checksums(directory):
        rows = []
        for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "checksums.sha256"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(directory)}")
        (directory / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def discard(self): return self.stop(save=False)

    def close(self):
        self._stop=True
        self._drain(2)
        self._close_deformation_stream(False)
        self._close_camera_streams(False)
        for t in self._threads: t.join(timeout=1)
        self._deformation_thread.join(timeout=1)
        self._camera_thread.join(timeout=1)
