#!/usr/bin/env python3
"""Receive the PND ZED RTP stream and publish model-sized RGB frames."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from array import array
from typing import BinaryIO

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ws_msgs.msg import ModelImage, Status

from ws_io.tcp_server import make_status_msg


class RobotVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("robot_vision")

        self.declare_parameter("rtp_port", 5601)
        self.declare_parameter("enable_decode", True)
        self.declare_parameter("output_width", 320)
        self.declare_parameter("output_height", 160)
        self.declare_parameter("decode_backend", "auto")
        self.declare_parameter("gst_pipeline", "")
        self.declare_parameter("gst_launch", "gst-launch-1.0")
        self.declare_parameter("model_image_topic", "/ws/robot_vision")
        self.declare_parameter("status_topic", "/ws/robot_vision/status")
        self.declare_parameter("status_period", 0.5)

        self.rtp_port = self._valid_port("rtp_port", self.get_parameter("rtp_port").value)
        self.enable_decode = bool(self.get_parameter("enable_decode").value)
        self.output_width = int(self.get_parameter("output_width").value)
        self.output_height = int(self.get_parameter("output_height").value)
        self.decode_backend = str(self.get_parameter("decode_backend").value).strip().lower()
        self.gst_pipeline = str(self.get_parameter("gst_pipeline").value)
        self.gst_launch = str(self.get_parameter("gst_launch").value).strip() or "gst-launch-1.0"
        self.model_image_topic = str(self.get_parameter("model_image_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_period = float(self.get_parameter("status_period").value)
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("output_width and output_height must be positive")
        if self.status_period <= 0.0:
            raise ValueError("status_period must be positive")
        if self.decode_backend not in {"auto", "opencv", "gst_launch"}:
            raise ValueError("decode_backend must be auto, opencv, or gst_launch")

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture_proc: subprocess.Popen[bytes] | None = None
        self.listening = False
        self.active_backend = "disabled"
        self.frames_received = 0
        self.frames_published = 0
        self.last_frame_seq: int | None = None
        self.last_frame_time: float | None = None
        self.last_error = ""
        self.last_decoder_stderr = ""

        self.model_image_pub = self.create_publisher(
            ModelImage,
            self.model_image_topic,
            10,
        )
        self.status_pub = self.create_publisher(Status, self.status_topic, 10)
        self.status_timer = self.create_timer(self.status_period, self._publish_status)
        self.worker: threading.Thread | None = None
        if self.enable_decode:
            self.worker = threading.Thread(
                target=self._capture_loop,
                name="robot_vision-rtp",
                daemon=True,
            )
            self.worker.start()

        self.get_logger().info(
            "robot_vision: "
            f"rtp=udp://0.0.0.0:{self.rtp_port}, "
            f"decode={self.enable_decode}, backend={self.decode_backend}, "
            f"model_image={self.model_image_topic}"
        )

    def _capture_loop(self) -> None:
        backends = (
            ["opencv", "gst_launch"]
            if self.decode_backend == "auto"
            else [self.decode_backend]
        )
        errors: list[str] = []
        for backend in backends:
            if self.stop_event.is_set():
                return
            error = (
                self._capture_loop_opencv()
                if backend == "opencv"
                else self._capture_loop_gst_launch()
            )
            if self.stop_event.is_set():
                return
            if error:
                errors.append(error)
                with self.lock:
                    self.last_error = "; ".join(errors[-3:])
            if self.decode_backend != "auto":
                return
        if errors:
            self.get_logger().error("robot_vision decode failed: " + "; ".join(errors))

    def _capture_loop_opencv(self) -> str | None:
        try:
            import cv2
        except Exception as exc:  # noqa: BLE001 - optional runtime dependency.
            return f"opencv import failed: {exc}"

        pipeline = self.gst_pipeline or self._default_gst_pipeline(self.rtp_port)
        cap = None
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError("failed to open RTP GStreamer pipeline")
            with self.lock:
                self.listening = True
                self.active_backend = "opencv_gstreamer"
                self.last_error = ""
            while not self.stop_event.is_set():
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    with self.lock:
                        self.last_error = "RTP decode returned no frame"
                    time.sleep(0.02)
                    continue
                with self.lock:
                    self.frames_received += 1
                    seq = self.frames_received
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if (
                    frame_rgb.shape[1] != self.output_width
                    or frame_rgb.shape[0] != self.output_height
                ):
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (self.output_width, self.output_height),
                        interpolation=cv2.INTER_AREA,
                    )
                self._publish_rgb_frame(seq, frame_rgb.tobytes())
        except Exception as exc:  # noqa: BLE001 - keep status publisher alive.
            return f"opencv RTP decode failed: {exc}"
        finally:
            with self.lock:
                self.listening = False
            if cap is not None:
                cap.release()
        return None

    def _capture_loop_gst_launch(self) -> str | None:
        gst_launch = shutil.which(self.gst_launch)
        if not gst_launch:
            return f"{self.gst_launch!r} not found in PATH"
        frame_bytes = self.output_width * self.output_height * 3
        command = self._gst_launch_command(gst_launch, self.rtp_port)
        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            with self.lock:
                self.capture_proc = proc
                self.listening = True
                self.active_backend = "gst_launch"
                self.last_error = ""
                self.last_decoder_stderr = ""
            threading.Thread(
                target=self._drain_stderr,
                args=(proc,),
                name="robot_vision-gst-stderr",
                daemon=True,
            ).start()

            assert proc.stdout is not None
            while not self.stop_event.is_set():
                frame = self._read_exact(proc.stdout, frame_bytes)
                if frame is None:
                    returncode = proc.poll()
                    if returncode is not None:
                        with self.lock:
                            stderr_tail = self.last_decoder_stderr
                        detail = f": {stderr_tail}" if stderr_tail else ""
                        return f"gst-launch exited with code {returncode}{detail}"
                    time.sleep(0.01)
                    continue
                with self.lock:
                    self.frames_received += 1
                    seq = self.frames_received
                self._publish_rgb_frame(seq, frame)
        except Exception as exc:  # noqa: BLE001 - keep status publisher alive.
            return f"gst-launch RTP decode failed: {exc}"
        finally:
            with self.lock:
                if self.capture_proc is proc:
                    self.capture_proc = None
                self.listening = False
            self._terminate_process(proc)
        return None

    def _publish_rgb_frame(self, seq: int, frame_rgb: bytes) -> None:
        stamp_ns = time.time_ns()
        msg = ModelImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "robot_vision"
        msg.frame_seq = seq
        msg.stamp_ns = stamp_ns
        msg.width = self.output_width
        msg.height = self.output_height
        msg.encoding = "rgb8"
        msg.data = array("B", frame_rgb)
        self.model_image_pub.publish(msg)
        with self.lock:
            self.frames_published += 1
            self.last_frame_seq = seq
            self.last_frame_time = time.monotonic()
            self.last_error = ""

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        stream = proc.stderr
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, b""):
                text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    with self.lock:
                        self.last_decoder_stderr = text[-1000:]
        except Exception:
            pass

    def _read_exact(self, stream: BinaryIO, nbytes: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = nbytes
        while remaining > 0 and not self.stop_event.is_set():
            chunk = stream.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining:
            return None
        return b"".join(chunks)

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self.lock:
            frames_received = self.frames_received
            frames_published = self.frames_published
            last_frame_seq = self.last_frame_seq
            last_frame_age_ms = (
                None
                if self.last_frame_time is None
                else round((now - self.last_frame_time) * 1000.0, 1)
            )
            listening = self.listening
            active_backend = self.active_backend
            last_error = self.last_error
            last_decoder_stderr = self.last_decoder_stderr
        payload = {
            "node": "robot_vision",
            "mode": f"rtp_h264_{active_backend}" if self.enable_decode else "decode_disabled",
            "rtp_port": self.rtp_port,
            "model_image_topic": self.model_image_topic,
            "rtp_decode_enabled": self.enable_decode,
            "decode_backend": self.decode_backend,
            "active_backend": active_backend,
            "output": {
                "width": self.output_width,
                "height": self.output_height,
                "encoding": "rgb8",
            },
            "listening": listening,
            "model_images_published": frames_published,
            "frames_received": frames_received,
            "last_frame_seq": last_frame_seq,
            "last_frame_age_ms": last_frame_age_ms,
            "last_error": last_error,
            "last_decoder_stderr": last_decoder_stderr,
            "time_ns": time.time_ns(),
        }
        msg = make_status_msg(
            "robot_vision",
            (not self.enable_decode) or (listening and not last_error),
            payload,
            self.get_clock().now().to_msg(),
        )
        self.status_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.stop_event.set()
        with self.lock:
            proc = self.capture_proc
        self._terminate_process(proc)
        if self.worker is not None:
            self.worker.join(timeout=1.0)
        return super().destroy_node()

    @staticmethod
    def _valid_port(name: str, value: object) -> int:
        port = int(value)
        if port <= 0 or port > 65535:
            raise ValueError(f"{name} must be in [1, 65535]")
        return port

    @staticmethod
    def _default_gst_pipeline(port: int) -> str:
        return (
            f"udpsrc port={int(port)} "
            'caps="application/x-rtp,media=video,encoding-name=H264,'
            'payload=96,clock-rate=90000" '
            "! rtph264depay ! h264parse ! avdec_h264 "
            "! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false"
        )

    def _gst_launch_command(self, gst_launch: str, port: int) -> list[str]:
        return [
            gst_launch,
            "-q",
            "udpsrc",
            f"port={int(port)}",
            "caps=application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000",
            "!",
            "rtph264depay",
            "!",
            "h264parse",
            "!",
            "avdec_h264",
            "!",
            "videoconvert",
            "!",
            "videoscale",
            "!",
            f"video/x-raw,format=RGB,width={self.output_width},height={self.output_height}",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[bytes] | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotVisionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
