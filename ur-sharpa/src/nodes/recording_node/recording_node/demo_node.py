"""Independent D435 demo service; never sends robot motion commands."""

from __future__ import annotations

import array
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import threading
import time

import numpy as np

from .demo_overlay import SensorHistory, jpeg_bytes, render_overlay, TorqueHistory
from .demo_recorder import DemoSession, SENSOR_TOPICS


SAMPLE_DIRECTORY_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{6}")


def normalize_recording_root(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("recording root is required")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError("recording root must be an absolute path")
    path = path.resolve(strict=False)
    if path == Path("/"):
        raise ValueError("recording root cannot be filesystem root")
    return path


def select_d435(devices, serial="", excluded=()):
    matches = [device for device in devices
               if "D435" in device["name"].upper() and device["serial"] not in excluded
               and (not serial or device["serial"] == serial)]
    if len(matches) != 1:
        raise RuntimeError("D435 not found" if not matches else "Multiple D435 cameras; set demo_camera_serial")
    return matches[0]["serial"]


class DemoService:
    def __init__(self, root, serial="", excluded=(), fps=30, width=1920, height=1080):
        self.root = str(Path(root).expanduser().resolve(strict=False))
        self.serial, self.excluded, self.fps = serial, excluded, fps
        self.width, self.height = int(width), int(height)
        self.history = SensorHistory()
        self.preview_torque_history = TorqueHistory()
        self.lock = threading.RLock()
        self.quit = threading.Event()
        self.session = None
        self.saving = False
        self.result = None
        self.deleted_path = ""
        self.preview = None
        self.last_frame_ns = 0
        self.camera_info = {}
        self.camera_error = "Waiting for D435"
        self.command_error = ""
        self.save_thread = None
        self.thread = threading.Thread(target=self._capture, name="demo-d435", daemon=True)

    def status(self):
        with self.lock:
            connected = time.time_ns() - self.last_frame_ns < 1_000_000_000
            session = self.session
            return {"connected": connected, "camera": dict(self.camera_info),
                    "camera_error": self.camera_error, "error": self.command_error,
                    "recording": session is not None and not self.saving, "saving": self.saving,
                    "duration_s": round((time.time_ns() - session.started_ns) / 1e9, 1) if session else 0,
                    "frames": session.frames if session else 0,
                    "dropped": dict(session.dropped) if session else {},
                    "writer_errors": list(session.errors) if session else [],
                    "root": self.root, "result": self.result,
                    "deleted_path": self.deleted_path}

    def command(self, action, root=None):
        with self.lock:
            if action == "start":
                if self.saving:
                    raise ValueError("Demo is still saving")
                if self.session:
                    return self.status()
                if not self.status()["connected"]:
                    raise ValueError(self.camera_error or "D435 disconnected")
                if root is not None:
                    self.root = str(normalize_recording_root(root))
                self.session = DemoSession(self.root, dict(self.camera_info), self.fps)
                self.result = None
                self.deleted_path = ""
                self.command_error = ""
            elif action == "stop":
                if self.session and not self.saving:
                    self.saving = True
                    self.save_thread = threading.Thread(target=self._save, name="demo-finalize", daemon=True)
                    self.save_thread.start()
            elif action == "delete":
                if self.session or self.saving:
                    raise ValueError("stop recording and wait for saving before deleting")
                if not self.result or not self.result.get("path"):
                    raise ValueError("there is no recently saved demo to delete")
                root_path = Path(self.root).resolve(strict=False)
                target = Path(str(self.result["path"]))
                try:
                    resolved = target.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise ValueError("recent demo directory no longer exists") from exc
                if (target.is_symlink() or resolved.parent != root_path
                        or not SAMPLE_DIRECTORY_PATTERN.fullmatch(resolved.name)):
                    raise ValueError("refusing to delete a path outside the current recording root")
                if not resolved.is_dir():
                    raise ValueError("recent demo directory no longer exists")
                shutil.rmtree(resolved)
                self.result = None
                self.deleted_path = str(resolved)
                self.command_error = ""
            else:
                raise ValueError("expected start, stop or delete")
            return self.status()

    def _save(self):
        session = self.session
        try:
            result = session.stop()
            with self.lock:
                self.result = result
        except Exception as exc:
            with self.lock:
                self.command_error = f"Demo save failed: {exc}; retained at {session.path}"
        finally:
            with self.lock:
                self.session = None
                self.saving = False

    def sensor(self, topic, msg, stamp, serialize):
        key = {"/sharpa/state": "hand", "/sharpa/tau": "tau", "/sharpa/wrench": "wrench",
               "/sharpa/deformation": "deform"}.get(topic)
        if key:
            self.history.update(key, msg, stamp)
        with self.lock:
            if self.session and not self.saving:
                self.session.observe(topic, msg, stamp)
                self.session.enqueue(topic, (stamp, serialize(msg)))

    def transport_drop(self, name, count):
        with self.lock:
            if self.session and not self.saving:
                self.session.error(f"{name} camera shared-memory overrun: {count} frame(s)")

    def _capture(self):
        while not self.quit.is_set():
            pipeline = None
            try:
                from ur_common.drivers import ensure_optional_runtime_paths
                ensure_optional_runtime_paths()
                import pyrealsense2 as rs

                context = rs.context()
                devices = [{"name": d.get_info(rs.camera_info.name),
                            "serial": d.get_info(rs.camera_info.serial_number)}
                           for d in context.query_devices()]
                serial = select_d435(devices, self.serial, self.excluded)
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
                device_name = next(d["name"] for d in devices if d["serial"] == serial)
                profile = pipeline.start(config)
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

                def intrinsics(stream):
                    value = stream.get_intrinsics()
                    return {key: getattr(value, key) for key in ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs")} | {"model": str(value.model)}

                with self.lock:
                    self.camera_info = {"serial": serial, "name": device_name,
                                        "rotation_clockwise_degrees": 180,
                                        "intrinsics_coordinate_space": "native sensor image before rotation",
                                        "width": self.width, "height": self.height, "fps": self.fps,
                                        "streams": ["color"], "color_intrinsics": intrinsics(color_profile)}
                preview_at = 0
                previous_frame = None
                rate_started, rate_count = time.monotonic(), 0
                while not self.quit.is_set():
                    frames = pipeline.wait_for_frames(2000)
                    stamp = time.time_ns()
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    number = color.get_frame_number()
                    if number == previous_frame:
                        continue
                    # Correct the mounted camera before preview and both video encoders.
                    rgb = np.ascontiguousarray(np.asanyarray(color.get_data())[::-1, ::-1])
                    metrics, planes = self.history.snapshot(stamp)
                    self.preview_torque_history.update(metrics)
                    source = {"received_ns": stamp, "color_timestamp_ms": color.get_timestamp(),
                              "color_frame_number": number,
                              "color_timestamp_domain": str(color.get_frame_timestamp_domain())}
                    with self.lock:
                        self.last_frame_ns = stamp
                        self.camera_error = ""
                        rate_count += 1
                        elapsed = time.monotonic() - rate_started
                        if elapsed >= 2:
                            self.camera_info["measured_fps"] = round(rate_count / elapsed, 1)
                            rate_started, rate_count = time.monotonic(), 0
                        if self.session and not self.saving:
                            if previous_frame is not None and number != previous_frame + 1:
                                self.session.error(f"D435 color frame gap: {previous_frame} -> {number}")
                            self.session.enqueue("d435", (rgb, metrics, planes, source))
                    previous_frame = number
                    if time.monotonic() - preview_at >= 0.1:
                        preview = jpeg_bytes(render_overlay(
                            rgb, metrics, planes, self.preview_torque_history))
                        with self.lock:
                            self.preview = preview
                        preview_at = time.monotonic()
            except Exception as exc:
                with self.lock:
                    self.camera_error = str(exc)
                    self.last_frame_ns = 0
                    self.preview = None
                    if self.session and not self.saving:
                        self.session.error(f"D435 disconnected: {exc}")
                        self.command("stop")
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
            self.quit.wait(2)

    def close(self):
        self.quit.set()
        self.thread.join(timeout=5)
        self.command("stop")
        if self.save_thread:
            self.save_thread.join()


def make_demo_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, content_type="application/json", code=200):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                self.send(json.dumps(service.status()).encode())
            elif self.path == "/frame.jpg":
                with service.lock:
                    preview = service.preview if service.status()["connected"] else None
                self.send(preview or b"D435 unavailable", "image/jpeg" if preview else "text/plain", 200 if preview else 503)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/command":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 1024:
                    raise ValueError("invalid body size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("expected JSON object")
                self.send(json.dumps(service.command(
                    payload.get("action"), payload.get("root"))).encode())
            except (ValueError, TypeError) as exc:
                self.send(json.dumps({"error": str(exc)}).encode(), code=409)
            except Exception as exc:
                self.send(json.dumps({"error": str(exc)}).encode(), code=500)

        def log_message(self, *_):
            pass
    return Handler


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.utilities import get_message
    from ur_common.camera_transport import SharedCameraReader

    class DemoNode(Node):
        def __init__(self):
            super().__init__("demo_node")
            for name, default in (("root", "demo_recordings"), ("serial", ""), ("port", 8089),
                                  ("width", 1920), ("height", 1080), ("fps", 30),
                                  ("shared_memory_prefix", "ur_sharpa_camera_v1"),
                                  ("left_wrist_serial", "230322270627")):
                self.declare_parameter(name, default)
            self.service = DemoService(str(self.get_parameter("root").value),
                                       str(self.get_parameter("serial").value),
                                       (str(self.get_parameter("left_wrist_serial").value), "218622279586"),
                                       fps=int(self.get_parameter("fps").value),
                                       width=int(self.get_parameter("width").value),
                                       height=int(self.get_parameter("height").value))
            self.reader = SharedCameraReader(str(self.get_parameter("shared_memory_prefix").value))
            self.shared_seen = False
            self.dropped = {}
            self.camera_type = get_message(SENSOR_TOPICS["/camera/set"])
            qos = QoSProfile(depth=30, reliability=ReliabilityPolicy.BEST_EFFORT)
            for topic, type_name in SENSOR_TOPICS.items():
                self.create_subscription(get_message(type_name), topic,
                                         lambda msg, topic=topic: self.receive(topic, msg), qos)
            self.create_timer(1 / 120, self.poll_cameras)
            self.server = ThreadingHTTPServer(("127.0.0.1", int(self.get_parameter("port").value)),
                                              make_demo_handler(self.service))
            self.http_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.http_thread.start()
            self.service.thread.start()

        def receive(self, topic, msg):
            if topic == "/camera/set" and self.shared_seen:
                return
            self.service.sensor(topic, msg, time.time_ns(), serialize_message)

        def poll_cameras(self):
            pending = self.reader.read_pending()
            for name, count in self.reader.dropped_frames.items():
                difference = count - self.dropped.get(name, 0)
                if difference > 0:
                    self.service.transport_drop(name, difference)
                self.dropped[name] = count
            if pending:
                self.shared_seen = True
            for frame in pending:
                msg = self.camera_type()
                msg.encoding = frame.encoding
                now = time.time_ns()
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(now, 1_000_000_000)
                for name in ("ego", "left_wrist", "right_wrist"):
                    setattr(msg, name, array.array("B", getattr(frame, name)))
                    for suffix in ("_valid", "_timestamp_ns"):
                        setattr(msg, name + suffix, getattr(frame, name + suffix))
                self.service.sensor("/camera/set", msg, time.time_ns(), serialize_message)

        def destroy_node(self):
            self.server.shutdown()
            self.server.server_close()
            self.http_thread.join()
            self.service.close()
            self.reader.close()
            return super().destroy_node()

    rclpy.init(args=args)
    node = DemoNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
