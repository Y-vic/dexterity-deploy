"""Bounded asynchronous ROS bag, RGB and annotated demo recording."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import select
import subprocess
import threading
import time

from PIL import Image

from .demo_overlay import render_overlay, TorqueHistory


SENSOR_TOPICS = {
    "/ur/state": "ur_msgs/msg/UrStateFrame",
    "/sharpa/state": "ur_msgs/msg/HandStateFrame",
    "/sharpa/tau": "ur_msgs/msg/TauFrame",
    "/sharpa/wrench": "ur_msgs/msg/WrenchFrame",
    "/sharpa/deformation": "ur_msgs/msg/DeformationFrame",
    "/camera/set": "ur_msgs/msg/CameraSet",
}


def open_bag(path):
    import rosbag2_py

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for topic, type_name in SENSOR_TOPICS.items():
        writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=type_name,
                                                   serialization_format="cdr"))
    return writer


class VideoPipe:
    def __init__(self, path, size, fps):
        self.log = path.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
            f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        os.set_blocking(self.process.stdin.fileno(), False)

    def write(self, image):
        remaining = memoryview(image.tobytes())
        descriptor = self.process.stdin.fileno()
        deadline = time.monotonic() + 5
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0 or not select.select([], [descriptor], [], timeout)[1]:
                raise RuntimeError("video encoder stalled")
            try:
                remaining = remaining[os.write(descriptor, remaining):]
            except BlockingIOError:
                continue

    def close(self):
        try:
            self.process.stdin.close()
            try:
                code = self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("video encoder did not finalize")
            if code:
                raise RuntimeError(f"video encoder failed ({code})")
        finally:
            self.log.close()


def acquisition_timing(frame_count, fps, first_frame_ns, last_frame_ns):
    """Return real acquisition duration and the MP4 timestamp scale factor."""
    nominal_duration = frame_count / fps if frame_count and fps > 0 else 0.0
    if frame_count > 1 and first_frame_ns > 0 and last_frame_ns > first_frame_ns:
        acquisition_duration = (last_frame_ns - first_frame_ns) / 1e9 + 1 / fps
    else:
        acquisition_duration = nominal_duration
    scale = acquisition_duration / nominal_duration if nominal_duration else 1.0
    return acquisition_duration, scale


def retime_video(path, scale):
    """Stretch MP4 timestamps without re-encoding so dropped frames do not accelerate playback."""
    if abs(scale - 1.0) < 1e-3:
        return
    temporary = path.with_name(path.stem + ".retimed" + path.suffix)
    try:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-itsscale", f"{scale:.9f}", "-i", str(path), "-map", "0:v:0",
            "-c", "copy", "-movflags", "+faststart", str(temporary),
        ], check=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DemoSession:
    def __init__(self, root, camera_info, fps=30, bag_factory=open_bag, queue_size=256):
        self.started_ns = time.time_ns()
        self.name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        self.path = Path(root).expanduser() / (self.name + ".partial")
        self.path.mkdir(parents=True)
        self.camera_info = camera_info
        self.fps = fps
        self._queue = queue.Queue(maxsize=queue_size)
        self._frame_queue = queue.Queue(maxsize=min(queue_size, 60))
        self._lock = threading.Lock()
        self._accepting = True
        self.dropped = Counter()
        self.counts = Counter()
        self.source_quality = {}
        self.errors = []
        self.frames = 0
        self.first_frame_ns = 0
        self.last_frame_ns = 0
        self.invalid_overlay_frames = 0
        self._torque_history = TorqueHistory()
        self._bag = bag_factory(self.path / "sensors")
        self._bag_thread = threading.Thread(target=self._run_bag, name="demo-bag", daemon=True)
        self._bag_thread.start()
        self._thread = threading.Thread(target=self._run, name="demo-writer", daemon=True)
        self._thread.start()

    def error(self, message):
        with self._lock:
            if str(message) not in self.errors:
                self.errors.append(str(message))

    def enqueue(self, kind, payload):
        with self._lock:
            if not self._accepting:
                return False
            try:
                target = self._queue if kind in SENSOR_TOPICS else self._frame_queue
                target.put_nowait((kind, payload))
                return True
            except queue.Full:
                self.dropped[kind] += 1
                return False

    def observe(self, topic, msg, received_ns):
        if topic == "/camera/set":
            sources = [("/camera/" + name, int(getattr(msg, name + "_timestamp_ns", 0)),
                        bool(getattr(msg, name + "_valid", False)) and bool(getattr(msg, name, b"")))
                       for name in ("ego", "left_wrist", "right_wrist")]
        else:
            valid = bool(getattr(msg, "valid", True))
            for side in ("left", "right"):
                value = getattr(msg, side + "_valid", True)
                valid = valid and (all(value) if hasattr(value, "__iter__") else bool(value))
            sources = [(topic, int(getattr(msg, "timestamp_ns", 0)), valid)]
        with self._lock:
            for name, stamp, valid in sources:
                row = self.source_quality.setdefault(name, {"frames": 0, "invalid": 0,
                                                            "gaps": 0, "last_ns": 0})
                if not valid or not 0 <= received_ns - stamp <= 250_000_000:
                    row["invalid"] += 1
                    continue
                if stamp == row["last_ns"]:
                    continue
                if stamp < row["last_ns"] or stamp - (row["last_ns"] or self.started_ns) > 250_000_000:
                    row["gaps"] += 1
                row["frames"] += 1
                row["last_ns"] = stamp

    def _run_bag(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                kind, (stamp, data) = item
                try:
                    self._bag.write(kind, data, stamp)
                    self.counts[kind] += 1
                except Exception as exc:
                    self.error(exc)
        except Exception as exc:
            self.error(exc)
        finally:
            # Releasing SequentialWriter closes SQLite and writes bag metadata.
            self._bag = None

    def _run(self):
        videos = []
        try:
            with (self.path / "frames.jsonl").open("w", encoding="utf-8") as index:
                while True:
                    item = self._frame_queue.get()
                    if item is None:
                        break
                    kind, payload = item
                    try:
                        rgb, metrics, planes, source = payload
                        frame_ns = int(source.get("received_ns", 0))
                        if frame_ns > 0:
                            self.first_frame_ns = self.first_frame_ns or frame_ns
                            self.last_frame_ns = frame_ns
                        raw = Image.fromarray(rgb, "RGB")
                        self._torque_history.update(metrics)
                        annotated = render_overlay(rgb, metrics, planes, self._torque_history)
                        if not videos:
                            videos.append(VideoPipe(self.path / "d435_rgb.mp4", raw.size, self.fps))
                            videos.append(VideoPipe(self.path / "demo.mp4", annotated.size, self.fps))
                        videos[0].write(raw)
                        videos[1].write(annotated)
                        index.write(json.dumps({"frame_index": self.frames,
                                                "camera": source, **metrics}, allow_nan=False) + "\n")
                        self.frames += 1
                        if any(f["joint_square_sum_rad2"] is None or f["joint_torque_square_sum_nm2"] is None
                               or not f["deformation_valid"] for f in metrics["fingers"]):
                            self.invalid_overlay_frames += 1
                    except Exception as exc:
                        self.error(exc)
        except Exception as exc:
            self.error(exc)
        finally:
            for video in videos:
                try:
                    video.close()
                except Exception as exc:
                    self.error(exc)

    def stop(self):
        with self._lock:
            self._accepting = False
            stopped_ns = time.time_ns()
        for worker, pending in ((self._bag_thread, self._queue), (self._thread, self._frame_queue)):
            while worker.is_alive():
                try:
                    pending.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue
        self._bag_thread.join()
        self._thread.join()
        playback_duration_s, timing_scale = acquisition_timing(
            self.frames, self.fps, self.first_frame_ns, self.last_frame_ns)
        try:
            for name in ("d435_rgb.mp4", "demo.mp4"):
                path = self.path / name
                if path.exists():
                    retime_video(path, timing_scale)
        except Exception as exc:
            self.error(f"video timestamp correction failed: {exc}")
        missing = [topic for topic in SENSOR_TOPICS if not self.counts[topic]]
        expected_sources = [topic for topic in SENSOR_TOPICS if topic != "/camera/set"] + [
            "/camera/" + name for name in ("ego", "left_wrist", "right_wrist")]
        source_errors = []
        for name in expected_sources:
            row = self.source_quality.get(name, {})
            if (not row.get("frames") or row.get("invalid") or row.get("gaps")
                    or stopped_ns - row.get("last_ns", 0) > 250_000_000):
                source_errors.append(name)
        complete = bool(self.frames and not self.errors and not self.dropped
                        and not missing and not source_errors and not self.invalid_overlay_frames)
        manifest = {
            "schema_version": 1, "complete": complete,
            "started_ns": self.started_ns, "stopped_ns": stopped_ns,
            "frames": self.frames, "sensor_messages": dict(self.counts),
            "missing_topics": missing, "dropped": dict(self.dropped),
            "source_quality": self.source_quality, "invalid_sources": source_errors,
            "invalid_overlay_frames": self.invalid_overlay_frames, "errors": self.errors,
            "camera": self.camera_info, "fps": self.fps,
            "playback_duration_s": playback_duration_s,
            "effective_frame_rate_hz": self.frames / playback_duration_s if playback_duration_s else 0.0,
            "video_timing": "MP4 timestamps follow D435 acquisition elapsed time; dropped frames are not accelerated",
            "finger_order": "left then right; thumb,index,middle,ring,pinky",
            "joint_metric": "sum(q_i*q_i) in rad^2; per-hand slices [0:5,5:9,9:13,13:17,17:22]",
            "torque_metric": "sum(tau_i*tau_i) in (N*m)^2 from /sharpa/tau; same per-finger slices as joint_metric",
            "sensor_storage": "ROS 2 sqlite3 bag; original sensor CDR messages; camera sets reconstructed from full-rate shared memory when available",
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        destination = self.path.with_name(self.name)
        self.path.rename(destination)
        self.path = destination
        return {"path": str(destination), **manifest}
