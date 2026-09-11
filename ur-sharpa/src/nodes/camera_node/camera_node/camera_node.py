"""Three named JPEG views with independent sampling timestamps."""

from __future__ import annotations

import array
import pathlib
import threading
import time
from typing import Any


class CameraSetCore:
    NAMES = ("ego", "left_wrist", "right_wrist")

    @classmethod
    def make(cls, frames: dict[str, tuple[int, bytes] | None], now_ns: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"encoding": "jpeg"}
        for name in cls.NAMES:
            item = frames.get(name)
            stamp, data = item if item is not None else (0, b"")
            if not isinstance(data, (bytes, bytearray)):
                raise ValueError(f"{name} JPEG must be bytes")
            payload[f"{name}_timestamp_ns"] = int(stamp)
            payload[name] = bytes(data)
            payload[f"{name}_valid"] = bool(stamp > 0 and data)
        payload["publish_timestamp_ns"] = int(now_ns or time.time_ns())
        return payload


class CompleteCameraSetGate:
    """Pass complete sets only after every view has advanced."""

    def __init__(self) -> None:
        self._last_timestamps: tuple[int, ...] | None = None

    def accept(self, payload: dict[str, Any]) -> bool:
        if not all(payload[f"{name}_valid"] for name in CameraSetCore.NAMES):
            return True
        timestamps = tuple(
            int(payload[f"{name}_timestamp_ns"]) for name in CameraSetCore.NAMES
        )
        if self._last_timestamps is not None and any(
            current == previous
            for current, previous in zip(timestamps, self._last_timestamps)
        ):
            return False
        self._last_timestamps = timestamps
        return True


class FrameProgressWatchdog:
    """Detect a connected camera that stops producing new frames."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError("camera frame timeout must be positive")
        self._last_timestamp_ns = 0
        self._last_progress_at = 0.0

    def reset(self, now: float, timestamp_ns: int = 0) -> None:
        self._last_timestamp_ns = int(timestamp_ns)
        self._last_progress_at = float(now)

    def stalled(self, timestamp_ns: int, now: float) -> bool:
        timestamp_ns = int(timestamp_ns)
        now = float(now)
        if timestamp_ns > 0 and timestamp_ns != self._last_timestamp_ns:
            self._last_timestamp_ns = timestamp_ns
            self._last_progress_at = now
            return False
        return now - self._last_progress_at >= self.timeout_s


def zed_device_diagnostic(camera: Any) -> str:
    """Return an actionable ZED-only diagnosis without opening another camera."""

    sl = getattr(camera, "sl", None)
    if sl is None:
        return "ZED Python API did not load; no fallback ego camera will be used"
    try:
        sdk_version = str(sl.Camera.get_sdk_version())
    except Exception:
        sdk_version = "unknown"
    try:
        devices = list(sl.Camera.get_device_list())
    except Exception as exc:
        return (
            f"ZED SDK {sdk_version} device enumeration failed: {exc}; "
            "no fallback ego camera will be used"
        )
    descriptions = []
    available = False
    for device in devices:
        state = str(getattr(device, "camera_state", "UNKNOWN"))
        serial = int(getattr(device, "serial_number", 0) or 0)
        path = str(getattr(device, "path", "") or "<empty>")
        model = str(getattr(device, "camera_model", "UNKNOWN"))
        descriptions.append(
            f"model={model}, state={state}, serial={serial}, path={path}"
        )
        available = available or state.strip().upper() == "AVAILABLE"
    if available:
        return (
            f"ZED SDK {sdk_version} sees an available device "
            f"({'; '.join(descriptions)}), but opening or frame capture failed; "
            "no fallback ego camera will be used"
        )
    detected = "; ".join(descriptions) if descriptions else "no ZED devices enumerated"
    return (
        f"ZED SDK {sdk_version} has no available main video device ({detected}). "
        "An MCU/HID entry alone cannot provide video. Flip or replug the ZED Type-C "
        "cable and connect it directly to a USB 3.x port, then rerun ZED_Diagnostic; "
        "no fallback ego camera will be used"
    )


def _read_file(path: str) -> tuple[int, bytes] | None:
    if not path: return None
    file = pathlib.Path(path).expanduser()
    try: return time.time_ns(), file.read_bytes()
    except OSError: return None


def _camera_frame(
    name: str,
    snapshot: Any,
    source_path: str,
    *,
    use_hardware: bool,
) -> tuple[int, bytes] | None:
    if snapshot is not None:
        return int(snapshot.timestamp_ns), bytes(snapshot.jpeg)
    if use_hardware and name == "ego":
        return None
    return _read_file(source_path)


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from ur_msgs.msg import CameraSet
    from ur_common.camera_transport import SharedCameraWriter
    from ur_common.drivers import HardwareConnectionError, HardwareDependencyError, RealSenseWristCamera, ZedEgoCamera

    class CameraNode(Node):
        def __init__(self):
            super().__init__("camera_node")
            for name in CameraSetCore.NAMES: self.declare_parameter(f"{name}_source", "")
            self.declare_parameter("use_hardware", True)
            self.declare_parameter("ego_zed_resolution", "HD720")
            self.declare_parameter("ego_zed_fps", 30)
            self.declare_parameter("ego_zed_serial_number", 0)
            self.declare_parameter("ego_first_frame_timeout_s", 5.0)
            self.declare_parameter("ego_frame_stall_timeout_s", 3.0)
            self.declare_parameter("diagnostic_interval_s", 10.0)
            self.declare_parameter("left_wrist_serial", "230322270627")
            self.declare_parameter("wrist_camera_fps", 30)
            self.declare_parameter("wrist_camera_width", 480)
            self.declare_parameter("wrist_camera_height", 270)
            self.declare_parameter("preview_fps", 10.0)
            self.declare_parameter("shared_memory_prefix", "ur_sharpa_camera_v1")
            self.declare_parameter("shared_memory_bytes", 4194304)
            self.declare_parameter("reconnect_interval_s", 2.0)
            wrist_camera_fps = int(self.get_parameter("wrist_camera_fps").value)
            wrist_camera_width = int(self.get_parameter("wrist_camera_width").value)
            wrist_camera_height = int(self.get_parameter("wrist_camera_height").value)
            use_hardware = bool(self.get_parameter("use_hardware").value)
            zed_serial = int(self.get_parameter("ego_zed_serial_number").value)
            self._use_hardware = use_hardware
            self._cameras = {}
            self._camera_candidates = {}
            self._next_connect_attempt = {}
            self._connect_threads = {}
            self._last_connect_errors = {}
            self._last_connect_log_at = {}
            self._last_camera_errors = {}
            self._last_camera_drop_counts = {}
            self._closing = False
            self._ego_first_frame_timeout_s = float(
                self.get_parameter("ego_first_frame_timeout_s").value
            )
            if self._ego_first_frame_timeout_s <= 0:
                raise ValueError("ego_first_frame_timeout_s must be positive")
            self._diagnostic_interval_s = float(
                self.get_parameter("diagnostic_interval_s").value
            )
            if self._diagnostic_interval_s <= 0:
                raise ValueError("diagnostic_interval_s must be positive")
            self._ego_watchdog = FrameProgressWatchdog(
                float(self.get_parameter("ego_frame_stall_timeout_s").value)
            )
            self._shared_camera = SharedCameraWriter(
                prefix=str(self.get_parameter("shared_memory_prefix").value),
                max_payload_bytes=int(self.get_parameter("shared_memory_bytes").value),
            )
            self._last_shared_signatures = {}
            self._reconnect_interval_s = float(self.get_parameter("reconnect_interval_s").value)
            if use_hardware:
                candidates = {
                    "ego": ZedEgoCamera(
                        resolution=str(self.get_parameter("ego_zed_resolution").value),
                        frequency_hz=int(self.get_parameter("ego_zed_fps").value),
                        serial_number=zed_serial if zed_serial > 0 else None,
                    ),
                    "left_wrist": RealSenseWristCamera(
                        label="left_wrist",
                        serial_number=str(self.get_parameter("left_wrist_serial").value),
                        width=wrist_camera_width,
                        height=wrist_camera_height,
                        frequency_hz=wrist_camera_fps,
                    ),
                    "right_wrist": RealSenseWristCamera(
                        label="right_wrist",
                        serial_number="218622279586",
                        width=wrist_camera_width,
                        height=wrist_camera_height,
                        frequency_hz=wrist_camera_fps,
                    ),
                }
                self._camera_candidates = candidates
                # Cameras are independent devices.  One busy/missing USB
                # stream must not hide frames from the cameras that opened
                # successfully; that made a single RealSense failure look
                # like a complete camera outage in the unified launch.
                for name, camera in candidates.items():
                    self._connect_camera(name, camera, initial=True)
            camera_qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.pub = self.create_publisher(CameraSet, "/camera/set", camera_qos)
            self._publish_gate = CompleteCameraSetGate()
            self._preview_period_s = 1.0 / max(0.1, float(self.get_parameter("preview_fps").value))
            self._last_preview_publish = 0.0
            poll_hz = max(30, wrist_camera_fps * 4) if use_hardware else 30
            self.timer = self.create_timer(1.0 / poll_hz, self._tick)

        def _connect_camera(self, name, camera, initial=False):
            try:
                previous = camera.read()
                previous_timestamp = (
                    int(previous.timestamp_ns) if previous is not None else 0
                )
                previous_capture_count = int(getattr(camera, "capture_count", 0))
                camera.connect()
                if self._closing:
                    camera.close()
                    return
                camera.start()
                first = camera.read()
                if name == "ego":
                    deadline = time.monotonic() + self._ego_first_frame_timeout_s
                    while (
                        (
                            first is None
                            or int(first.timestamp_ns) == previous_timestamp
                            or int(getattr(camera, "capture_count", 0))
                            <= previous_capture_count
                        )
                        and not self._closing
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.02)
                        first = camera.read()
                    if (
                        first is None
                        or int(first.timestamp_ns) == previous_timestamp
                        or int(getattr(camera, "capture_count", 0))
                        <= previous_capture_count
                    ):
                        detail = zed_device_diagnostic(camera)
                        raise HardwareConnectionError(
                            "ZED opened but produced no new ego frame before "
                            f"{self._ego_first_frame_timeout_s:.1f}s timeout; {detail}"
                        )
                    self._ego_watchdog.reset(
                        time.monotonic(), int(first.timestamp_ns)
                    )
                self._cameras[name] = camera
                self._next_connect_attempt.pop(name, None)
                self._last_connect_errors.pop(name, None)
                if not initial:
                    self.get_logger().info(f"camera {name} reconnected")
            except (HardwareDependencyError, HardwareConnectionError, OSError) as exc:
                camera.close()
                self._next_connect_attempt[name] = time.monotonic() + self._reconnect_interval_s
                error = str(exc)
                if name == "ego" and "no fallback ego camera" not in error:
                    error = f"{error}; {zed_device_diagnostic(camera)}"
                now = time.monotonic()
                if (
                    initial
                    or error != self._last_connect_errors.get(name)
                    or now - self._last_connect_log_at.get(name, 0.0)
                    >= self._diagnostic_interval_s
                ):
                    self.get_logger().warning(f"camera {name} unavailable, retrying: {error}")
                    self._last_connect_log_at[name] = now
                self._last_connect_errors[name] = error

        def _recycle_camera(self, name, camera, reason):
            camera.close()
            if self._closing:
                return
            self.get_logger().warning(f"camera {name} restarting: {reason}")
            self._connect_camera(name, camera)

        def _schedule_recycle(self, name, reason):
            thread = self._connect_threads.get(name)
            if thread is not None and thread.is_alive():
                return
            camera = self._cameras.pop(name, None)
            if camera is None:
                return
            thread = threading.Thread(
                target=self._recycle_camera,
                args=(name, camera, reason),
                name=f"{name}-reconnect",
                daemon=True,
            )
            self._connect_threads[name] = thread
            thread.start()

        def _reconnect_missing(self):
            now = time.monotonic()
            for name, camera in self._camera_candidates.items():
                if name in self._cameras or now < self._next_connect_attempt.get(name, 0.0):
                    continue
                thread = self._connect_threads.get(name)
                if thread is not None and thread.is_alive():
                    continue
                available = getattr(camera, "device_available", None)
                if available is not None and not available():
                    self._next_connect_attempt[name] = now + self._reconnect_interval_s
                    continue
                thread = threading.Thread(
                    target=self._connect_camera,
                    args=(name, camera),
                    name=f"{name}-reconnect",
                    daemon=True,
                )
                self._connect_threads[name] = thread
                thread.start()

        def _tick(self):
            self._reconnect_missing()
            data = {}
            pending_frames = {}
            for name in CameraSetCore.NAMES:
                camera = self._cameras.get(name)
                drain = getattr(camera, "read_pending", None)
                pending = list(drain()) if drain is not None else []
                snapshot = (
                    pending[-1]
                    if pending
                    else camera.read() if camera is not None else None
                )
                if (
                    name == "ego"
                    and camera is not None
                    and self._ego_watchdog.stalled(
                        int(snapshot.timestamp_ns) if snapshot is not None else 0,
                        time.monotonic(),
                    )
                ):
                    self._schedule_recycle(
                        "ego",
                        "ZED frame timestamp stopped advancing for "
                        f"{self._ego_watchdog.timeout_s:.1f}s",
                    )
                    snapshot = None
                    camera = None
                    pending = []
                pending_frames[name] = pending
                data[name] = _camera_frame(
                    name,
                    snapshot,
                    str(self.get_parameter(f"{name}_source").value),
                    use_hardware=self._use_hardware,
                )
                error = getattr(camera, "last_capture_error", "") or getattr(
                    camera, "last_encode_error", ""
                ) if camera is not None else ""
                if error and error != self._last_camera_errors.get(name):
                    self.get_logger().warning(f"camera {name} capture failed: {error}")
                self._last_camera_errors[name] = error
                if camera is not None:
                    dropped = (
                        int(getattr(camera, "dropped_raw_count", 0)),
                        int(getattr(camera, "dropped_encoded_count", 0)),
                    )
                    previous_dropped = self._last_camera_drop_counts.get(name, (0, 0))
                    if dropped != previous_dropped and sum(dropped) > sum(previous_dropped):
                        self.get_logger().warning(
                            f"camera {name} queue dropped frames: "
                            f"raw={dropped[0]} encoded={dropped[1]}"
                        )
                    self._last_camera_drop_counts[name] = dropped
            payload = CameraSetCore.make(data)
            for name in CameraSetCore.NAMES:
                for frame in pending_frames[name]:
                    timestamp = int(frame.timestamp_ns)
                    jpeg = bytes(frame.jpeg)
                    self._shared_camera.write(name, timestamp, jpeg, True)
                    self._last_shared_signatures[name] = (timestamp, True)
                signature = (payload[f"{name}_timestamp_ns"], payload[f"{name}_valid"])
                if signature != self._last_shared_signatures.get(name):
                    self._shared_camera.write(name, signature[0], payload[name], signature[1])
                    self._last_shared_signatures[name] = signature
            now = time.monotonic()
            if now - self._last_preview_publish < self._preview_period_s:
                return
            if not self._publish_gate.accept(payload):
                return
            self._last_preview_publish = now
            msg = CameraSet(); msg.header.stamp = self.get_clock().now().to_msg(); msg.encoding = payload["encoding"]; msg.ego = array.array("B", payload["ego"]); msg.left_wrist = array.array("B", payload["left_wrist"]); msg.right_wrist = array.array("B", payload["right_wrist"]); msg.ego_timestamp_ns = payload["ego_timestamp_ns"]; msg.left_wrist_timestamp_ns = payload["left_wrist_timestamp_ns"]; msg.right_wrist_timestamp_ns = payload["right_wrist_timestamp_ns"]; msg.ego_valid = payload["ego_valid"]; msg.left_wrist_valid = payload["left_wrist_valid"]; msg.right_wrist_valid = payload["right_wrist_valid"]; self.pub.publish(msg)

        def destroy_node(self):
            self._closing = True
            for thread in self._connect_threads.values():
                thread.join(timeout=2.0)
            for camera in self._camera_candidates.values(): camera.close()
            self._shared_camera.close()
            return super().destroy_node()

    rclpy.init(args=args); node = CameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
