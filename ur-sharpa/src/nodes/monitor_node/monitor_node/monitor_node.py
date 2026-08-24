"""Operator dashboard for the teleop stack.

The web process never talks to hardware directly. Machine actions are sent to
the hidden status node; only validated, completed samples can be uploaded.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import re
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import String
    from ur_msgs.msg import CameraSet, ControlStatus, HandStateFrame, UrStateFrame
    from ur_common.camera_transport import SharedCameraReader
    ROS_AVAILABLE = True
except Exception:  # pragma: no cover
    ROS_AVAILABLE = False
    Node = object  # type: ignore


HTML = (Path(__file__).with_name("static") / "index.html").read_text(
    encoding="utf-8"
)

DEFAULT_REMOTE_DATA_PATH = (
    "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/raw_data"
)


class MonitorController:
    RECORDING_NAME = re.compile(
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{6}"
    )

    def __init__(self, recording_root="./episodes", upload_host="BAAI", upload_path=DEFAULT_REMOTE_DATA_PATH, publish_command=None, publish_bias=None, publish_config=None, publish_label=None, status_timeout_s=2.0):
        self.recording_root = str(Path(recording_root).expanduser())
        self.upload_host = str(upload_host)
        self.upload_path = str(upload_path)
        self.publish_command = publish_command
        self.publish_bias = publish_bias
        self.publish_config = publish_config
        self.publish_label = publish_label
        self.status_timeout_s = float(status_timeout_s)
        self.status = {}
        self.upstream = {}
        self._upstream_received = {}
        self._status_received = 0.0
        self.cameras = {}
        self._camera_received = {}
        self._camera_timestamps = {}
        self._camera_samples = {
            name: deque(maxlen=120)
            for name in ("ego", "left_wrist", "right_wrist")
        }
        self._uploading = False
        self.message = ""
        self._lock = threading.RLock()
        self._camera_condition = threading.Condition(self._lock)

    @staticmethod
    def _recording_counts(recording_root):
        root = Path(recording_root)

        def count(parent):
            try:
                return sum(
                    1 for path in parent.iterdir()
                    if path.is_dir() and MonitorController.RECORDING_NAME.fullmatch(path.name)
                )
            except OSError:
                return 0

        return {
            "success": count(root / "success"),
            "fail": count(root / "fail"),
            "undefined": count(root),
        }

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            status = dict(self.status)
            status_age = max(0.0, now - self._status_received) if self._status_received else float("inf")
            status["age_s"] = round(status_age, 3) if status_age != float("inf") else None
            status["online"] = status_age <= self.status_timeout_s
            upstream = {}
            for key, value in self.upstream.items():
                payload = dict(value) if isinstance(value, dict) else {"value": value}
                age = max(0.0, now - self._upstream_received.get(key, 0.0))
                payload["age_s"] = round(age, 3)
                payload["online"] = age <= self.status_timeout_s
                upstream[key] = payload
            camera_status = {}
            for name in ("ego", "left_wrist", "right_wrist"):
                received = self._camera_received.get(name, 0.0)
                age = max(0.0, now - received) if received else float("inf")
                online = bool(self.cameras.get(name)) and age <= self.status_timeout_s
                samples = self._camera_samples[name]
                while samples and now - samples[0] > 2.0:
                    samples.popleft()
                fps = 0.0
                if online and len(samples) >= 2:
                    elapsed = samples[-1] - samples[0]
                    if elapsed > 0.0:
                        fps = (len(samples) - 1) / elapsed
                camera_status[name] = {
                    "online": online,
                    "valid": online,
                    "age_s": round(age, 3) if age != float("inf") else None,
                    "fps_hz": round(fps, 1),
                    "source_timestamp_ns": self._camera_timestamps.get(name, 0),
                }
            return {"status": status, "upstream": upstream,
                    "camera_status": camera_status,
                    "recording_root": self.recording_root, "upload_host": self.upload_host,
                    "upload_path": self.upload_path, "uploading": self._uploading,
                    "recording_counts": self._recording_counts(self.recording_root),
                    "message": self.message}

    def camera_frame(self, name):
        if name not in ("ego", "left_wrist", "right_wrist"):
            return None
        with self._lock:
            received = self._camera_received.get(name, 0.0)
            if not self.cameras.get(name) or time.monotonic() - received > self.status_timeout_s:
                return None
            return self.cameras[name], self._camera_timestamps.get(name, 0)

    def wait_camera_frame(self, name, after_timestamp=0, timeout=1.0):
        if name not in ("ego", "left_wrist", "right_wrist"):
            return None
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._camera_condition:
            while True:
                timestamp = self._camera_timestamps.get(name, 0)
                received = self._camera_received.get(name, 0.0)
                if (
                    timestamp != int(after_timestamp)
                    and self.cameras.get(name)
                    and time.monotonic() - received <= self.status_timeout_s
                ):
                    return self.cameras[name], timestamp
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._camera_condition.wait(remaining)

    def update_status(self, value):
        with self._lock:
            self.status = dict(value)
            self._status_received = time.monotonic()

    def update_upstream(self, key, value, received=None):
        with self._lock:
            self.upstream[key] = value
            self._upstream_received[key] = time.monotonic() if received is None else float(received)

    def update_cameras(self, value, timestamps=None, received=None):
        with self._camera_condition:
            now = time.monotonic() if received is None else float(received)
            timestamps = timestamps or {}
            updated = False
            for key in ("ego", "left_wrist", "right_wrist"):
                data = value.get(key, b"")
                if not data:
                    continue
                source_timestamp = int(timestamps.get(key, 0))
                if source_timestamp <= 0:
                    source_timestamp = time.time_ns()
                if source_timestamp == self._camera_timestamps.get(key):
                    continue
                self.cameras[key] = bytes(data)
                self._camera_timestamps[key] = source_timestamp
                self._camera_received[key] = now
                self._camera_samples[key].append(now)
                updated = True
            if updated:
                self._camera_condition.notify_all()

    def action(self, body):
        action = str(body.get("action", ""))
        accepted = True
        with self._lock:
            if action == "set_paths":
                if int(self.status.get("record_state", 0)) == 1:
                    self.message = "paths cannot change while recording"
                    accepted = False
                    result = self.snapshot(); result["accepted"] = accepted
                    return result
                self.recording_root = str(Path(str(body.get("recording_root", self.recording_root))).expanduser())
                self.upload_path = str(body.get("upload_path", self.upload_path)).strip()
                if self.publish_config:
                    self.publish_config({"recording_root": self.recording_root})
                self.message = "paths updated"
            elif action == "save_bias":
                if self.publish_bias: self.publish_bias("save_current")
                self.message = "bias save requested"
            elif action == "upload":
                if self._uploading:
                    self.message = "upload already in progress"
                    accepted = False
                else:
                    self._uploading = True
                    self.message = "upload in progress"
                    threading.Thread(target=self._upload_background, daemon=True).start()
            elif action in {"mark_success", "mark_fail", "drop_sample"}:
                category = "drop" if action == "drop_sample" else action.removeprefix("mark_")
                recorder = self.upstream.get("record_status", {})
                validation = recorder.get("validation") if isinstance(recorder, dict) else None
                if int(self.status.get("record_state", 0)) == 1:
                    self.message = "cannot classify while recording"
                    accepted = False
                elif not isinstance(validation, dict) or not validation.get("available"):
                    self.message = "no saved sample is waiting for classification"
                    accepted = False
                else:
                    if self.publish_label:
                        self.publish_label(category)
                    self.message = (
                        "sample permanent deletion requested"
                        if category == "drop"
                        else f"sample {category} classification requested"
                    )
            elif action in {"start_ur", "run_ur", "reset_ur", "start_sharpa", "reset_both", "start_record", "stop_record", "discard_record", "pause", "stop"}:
                if self.publish_command: self.publish_command(action)
                self.message = f"{action} requested"
            else:
                self.message = f"unknown action: {action}"
                accepted = False
            result = self.snapshot(); result["accepted"] = accepted
            return result

    def _upload_latest(self):
        with self._lock:
            root = Path(self.recording_root)
            upload_host = self.upload_host
            upload_path = self.upload_path
        samples = []
        for category in ("", "success", "fail"):
            parent = root if not category else root / category
            if not parent.is_dir():
                continue
            for path in parent.iterdir():
                if (
                    path.is_dir()
                    and self.RECORDING_NAME.fullmatch(path.name)
                ):
                    samples.append((path.stat().st_mtime_ns, category, path))
        samples = [(category, path) for _, category, path in sorted(samples, reverse=True)]
        if not samples: return "upload failed: no completed sample"
        category, sample = samples[0]; manifest = sample / "manifest.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if not (sample / "COMPLETE").is_file() or data.get("state") != "complete" or not data.get("quality", {}).get("ok", False): return "upload failed: sample sanity check failed"
        except Exception as exc:
            return f"upload failed: invalid manifest ({exc})"
        if not self._verify_checksums(sample): return "upload failed: local checksum mismatch"
        if not upload_path: return "upload failed: upload path is empty"
        remote_parent = upload_path.rstrip("/")
        if category:
            remote_parent = f"{remote_parent}/{category}"
        target = f"{upload_host}:{remote_parent}/"
        try:
            subprocess.run(["ssh", upload_host, f"mkdir -p -- {shlex.quote(remote_parent)}"], check=True, timeout=60)
            subprocess.run(["rsync", "-a", "--delete", "--partial", "--delay-updates", str(sample) + "/", target + sample.name + ".partial/"], check=True, timeout=600)
            remote_partial = f"{remote_parent}/{sample.name}.partial"
            remote_complete = f"{remote_parent}/{sample.name}"
            verify = f"cd {shlex.quote(remote_partial)} && sha256sum -c checksums.sha256"
            subprocess.run(["ssh", upload_host, verify], check=True, timeout=600)
            move = f"test ! -e {shlex.quote(remote_complete)} && mv -- {shlex.quote(remote_partial)} {shlex.quote(remote_complete)}"
            subprocess.run(["ssh", upload_host, move], check=True, timeout=60)
        except Exception as exc:
            return f"upload failed: {exc}"
        relative = f"{category}/{sample.name}" if category else sample.name
        return f"uploaded {relative}"

    def _upload_background(self):
        try:
            result = self._upload_latest()
        except Exception as exc:
            result = f"upload failed: {exc}"
        with self._lock:
            self.message = result
            self._uploading = False

    @staticmethod
    def _verify_checksums(sample):
        checksum_file = sample / "checksums.sha256"
        try:
            for line in checksum_file.read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                if hashlib.sha256((sample / relative).read_bytes()).hexdigest() != digest:
                    return False
            return True
        except (OSError, ValueError):
            return False


class _Handler(BaseHTTPRequestHandler):
    controller = None
    def log_message(self, *_args): pass
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            data = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        if self.path.startswith("/api/status"): self._json(self.controller.snapshot()); return
        if self.path.startswith("/api/camera-stream/"):
            name = self.path.split("?", 1)[0].rsplit("/", 1)[-1]
            self._camera_stream(name); return
        if self.path.startswith("/api/camera/"):
            name = self.path.split("?", 1)[0].rsplit("/", 1)[-1]
            frame = self.controller.camera_frame(name)
            if frame is None: self.send_error(404); return
            data, timestamp = frame
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store"); self.send_header("X-Source-Timestamp-Ns", str(timestamp)); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        self.send_error(404)
    def _camera_stream(self, name):
        if name not in ("ego", "left_wrist", "right_wrist"):
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        timestamp = 0
        try:
            while True:
                frame = self.controller.wait_camera_frame(name, timestamp, timeout=1.0)
                if frame is None:
                    continue
                data, timestamp = frame
                header = (
                    f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n"
                    f"X-Source-Timestamp-Ns: {timestamp}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
    def do_POST(self):
        if not self.path.startswith("/api/action"): self.send_error(404); return
        try:
            n = int(self.headers.get("content-length", "0")); self._json(self.controller.action(json.loads(self.rfile.read(n) or b"{}")))
        except Exception as exc: self._json({"accepted": False, "message": str(exc)}, 400)
    def _json(self, payload, code=200):
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(); self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


if ROS_AVAILABLE:
    class MonitorNode(Node):
        def __init__(self):
            super().__init__("monitor_node")
            for name, default in (("port", 8080), ("recording_root", "/media/yns/T9/0824_yns_unscrew_cap"), ("upload_host", "BAAI"), ("upload_path", DEFAULT_REMOTE_DATA_PATH), ("status_timeout_s", 2.0)):
                self.declare_parameter(name, default)
            self.declare_parameter("shared_memory_prefix", "ur_sharpa_camera_v1")
            qos = QoSProfile(depth=10); camera_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.controller = MonitorController(self.get_parameter("recording_root").value, self.get_parameter("upload_host").value, self.get_parameter("upload_path").value, status_timeout_s=self.get_parameter("status_timeout_s").value)
            self.command_pub = self.create_publisher(String, "/control/ui_command", qos); self.bias_pub = self.create_publisher(String, "/control/bias_request", qos); self.config_pub = self.create_publisher(String, "/control/record_config", qos); self.label_pub = self.create_publisher(String, "/control/record_label", qos)
            self.controller.publish_command = lambda a: self._publish(self.command_pub, a); self.controller.publish_bias = lambda a: self._publish(self.bias_pub, a); self.controller.publish_config = lambda value: self._publish(self.config_pub, json.dumps(value, separators=(",", ":"))); self.controller.publish_label = lambda value: self._publish(self.label_pub, value)
            self.create_subscription(ControlStatus, "/control/status", self._status, qos); self.create_subscription(CameraSet, "/camera/set", self._camera, camera_qos)
            self.camera_reader = SharedCameraReader(str(self.get_parameter("shared_memory_prefix").value)); self._shared_camera_seen = False; self.create_timer(1.0 / 60.0, self._poll_shared_cameras)
            self.create_subscription(UrStateFrame, "/ur/state", self._ur, qos); self.create_subscription(HandStateFrame, "/sharpa/state", self._sharpa, qos)
            for topic, key in (("/control/bias_status", "bias_status"), ("/control/record_status", "record_status"), ("/teleop/vive_status", "vive_status"), ("/teleop/manus_status", "manus_status")):
                self.create_subscription(String, topic, self._upstream(key), qos)
            self.http = ThreadingHTTPServer(("0.0.0.0", int(self.get_parameter("port").value)), _Handler); self.http.daemon_threads = True; _Handler.controller = self.controller
            self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True); self.http_thread.start()
        def _publish(self, pub, value): msg = String(); msg.data = str(value); pub.publish(msg)
        def _status(self, msg):
            self.controller.update_status({"ur_state": msg.ur_state, "sharpa_state": msg.sharpa_state, "record_state": msg.record_state, "active_source": msg.active_source, "gate_reason": msg.gate_reason, "session_id": msg.session_id})
        def _ur(self, msg):
            self.controller.update_upstream("ur_status", {"valid": bool(msg.valid), "normal_mode": bool(msg.normal_mode), "message": msg.status_message})
        def _sharpa(self, msg):
            self.controller.update_upstream("sharpa_status", {
                "valid": bool(msg.left_valid or msg.right_valid),
                "left_valid": bool(msg.left_valid),
                "right_valid": bool(msg.right_valid),
            })
        def _upstream(self, key):
            def cb(msg):
                try: self.controller.update_upstream(key, json.loads(msg.data))
                except Exception: self.controller.update_upstream(key, msg.data)
            return cb
        def _camera(self, msg):
            if self._shared_camera_seen: return
            self.controller.update_cameras(
                {"ego": bytes(msg.ego) if msg.ego_valid else b"", "left_wrist": bytes(msg.left_wrist) if msg.left_wrist_valid else b"", "right_wrist": bytes(msg.right_wrist) if msg.right_wrist_valid else b""},
                {"ego": msg.ego_timestamp_ns, "left_wrist": msg.left_wrist_timestamp_ns, "right_wrist": msg.right_wrist_timestamp_ns},
            )
        def _poll_shared_cameras(self):
            msg = self.camera_reader.read()
            if msg is None: return
            self._shared_camera_seen = True
            self.controller.update_cameras(
                {"ego": msg.ego if msg.ego_valid else b"", "left_wrist": msg.left_wrist if msg.left_wrist_valid else b"", "right_wrist": msg.right_wrist if msg.right_wrist_valid else b""},
                {"ego": msg.ego_timestamp_ns, "left_wrist": msg.left_wrist_timestamp_ns, "right_wrist": msg.right_wrist_timestamp_ns},
            )
        def destroy_node(self):
            self.camera_reader.close(); self.http.shutdown(); self.http.server_close(); return super().destroy_node()
else:
    class MonitorNode:  # pragma: no cover
        def __init__(self, *_args, **_kwargs): raise RuntimeError("rclpy is unavailable")


def main(args=None):
    if not ROS_AVAILABLE: raise RuntimeError("rclpy is unavailable")
    rclpy.init(args=args); node = MonitorNode()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError): pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
