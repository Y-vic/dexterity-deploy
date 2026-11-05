"""Latest JPEG preview cache, independent of recording and policy buffers."""

from collections import deque
import threading
import time


class CameraPreview:
    def __init__(self, status_timeout_s=2.0):
        self.status_timeout_s = float(status_timeout_s)
        self.cameras = {}
        self._camera_received = {}
        self._camera_timestamps = {}
        self._camera_samples = {
            name: deque(maxlen=120) for name in ("ego", "left_wrist", "right_wrist")
        }
        self._lock = threading.RLock()
        self._camera_condition = threading.Condition(self._lock)

    def camera_frame(self, name):
        with self._lock:
            received = self._camera_received.get(name, 0.0)
            if not self.cameras.get(name) or time.monotonic() - received > self.status_timeout_s:
                return None
            return self.cameras[name], self._camera_timestamps.get(name, 0)

    def wait_camera_frame(self, name, after_timestamp=0, timeout=1.0):
        if name not in self._camera_samples:
            return None
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._camera_condition:
            while True:
                frame = self.camera_frame(name)
                if frame is not None and frame[1] != int(after_timestamp):
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._camera_condition.wait(remaining)

    def update_cameras(self, value, timestamps=None, received=None):
        with self._camera_condition:
            now = time.monotonic() if received is None else float(received)
            timestamps = timestamps or {}
            updated = False
            for name in self._camera_samples:
                data = value.get(name, b"")
                if not data:
                    continue
                stamp = int(timestamps.get(name, 0))
                if stamp <= 0:
                    stamp = time.time_ns()
                if stamp == self._camera_timestamps.get(name):
                    continue
                self.cameras[name] = bytes(data)
                self._camera_timestamps[name] = stamp
                self._camera_received[name] = now
                self._camera_samples[name].append(now)
                updated = True
            if updated:
                self._camera_condition.notify_all()

    def update_message(self, message):
        if str(message.encoding).lower() not in ("jpeg", "jpg"):
            return
        self.update_cameras(
            {name: getattr(message, name) if getattr(message, name + "_valid") else b""
             for name in self._camera_samples},
            {name: getattr(message, name + "_timestamp_ns") for name in self._camera_samples},
        )


class OnDemandCameraPreview(CameraPreview):
    """Deployment preview: read shared memory only for an HTTP frame request.

    The existing status subscription supplies a fallback reference while a
    viewer is active. There is no preview timer or background JPEG copying.
    """

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        self._request_lock = threading.Lock()
        self._requested_at = 0.0
        self._source = None

    def accept_source(self, message):
        with self._lock:
            self._source = message if time.monotonic() - self._requested_at < self.status_timeout_s else None

    def camera_frame(self, name):
        from .camera_transport import SharedCameraReader
        with self._request_lock:
            with self._lock:
                self._requested_at = time.monotonic()
                source = self._source
            reader = SharedCameraReader(self.prefix)
            try:
                message = reader.read()
            finally:
                reader.close()
            for candidate in (message, source):
                if candidate is not None:
                    self.update_message(candidate)
                    result = super().camera_frame(name)
                    if result is not None:
                        return result
            return super().camera_frame(name)

    def release(self):
        with self._request_lock, self._lock:
            self._requested_at = 0.0
            self._source = None
            self.cameras.clear()
            self._camera_received.clear()
            self._camera_timestamps.clear()
            for samples in self._camera_samples.values():
                samples.clear()
