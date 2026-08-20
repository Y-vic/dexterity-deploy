#!/usr/bin/env python3
"""ZED RGB source/status node.

ZED stays sample-agnostic. It owns the single remote camera/WebRTC/NVENC source
and reports source health; monitor owns t_record, sample directories, and file
writing.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import rclpy
import websockets
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

from zed_node.browser_ui import (
    DEFAULT_BROWSER_UI_HOST,
    DEFAULT_BROWSER_UI_PORT,
    BrowserUiServer,
)
from zed_node.remote import (
    DEFAULT_QUEST_STREAM_BIND_HOST,
    DEFAULT_QUEST_STREAM_PORT,
    DEFAULT_MONITOR_STREAM_HOST,
    DEFAULT_MONITOR_STREAM_PORT,
    DEFAULT_INFERENCE_STREAM_HOST,
    DEFAULT_INFERENCE_STREAM_PORT,
    RemoteConfig,
    RemoteProcess,
    RTP_PAYLOAD_TYPE,
    VIDEO_LAYOUTS,
    check_hardware,
    hardware_command,
    start_remote,
    stop_video_processes,
)


DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_BITRATE = 8000000
DEFAULT_WEBRTC_PORT = 8443
DEFAULT_WATCHDOG_FAILURE_THRESHOLD = 5
DEFAULT_WATCHDOG_RESTART_COOLDOWN_S = 20.0


@dataclass
class SignalStatus:
    connected: bool = False
    producer_count: int = 0
    producer_ids: list[str] = field(default_factory=list)
    last_message: str = ""
    last_error: str = ""
    latency_ms: float = 0.0

    @property
    def has_producer(self) -> bool:
        return self.connected and self.producer_count > 0


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return bool(value)


def _elapsed_ms(started_at: float) -> float:
    return round((time.monotonic() - started_at) * 1000.0, 3)


def _age_ms(stamp: float | None, now: float) -> float | None:
    if stamp is None:
        return None
    value = (now - stamp) * 1000.0
    return round(value, 3) if math.isfinite(value) else None


def _producer_id(producer: Any) -> str:
    if isinstance(producer, dict):
        value = (
            producer.get("id")
            or producer.get("peerId")
            or producer.get("producerId")
            or producer.get("name")
        )
        return str(value) if value is not None else ""
    return str(producer) if producer is not None else ""


async def check_webrtc_signal(
    host: str, port: int, timeout: float = 3.0
) -> SignalStatus:
    started_at = time.monotonic()
    websocket = None
    try:
        websocket = await asyncio.wait_for(
            websockets.connect(
                f"ws://{host}:{port}",
                ping_interval=None,
                proxy=None,
            ),
            timeout=timeout,
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "setPeerStatus",
                    "roles": ["listener"],
                    "meta": {"name": "zed_probe"},
                }
            )
        )
        await websocket.send(json.dumps({"type": "list"}))
        producer_count = 0
        producer_ids: list[str] = []
        last_message = ""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            remaining = max(0.1, deadline - asyncio.get_running_loop().time())
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            last_message = str(data.get("type") or "")
            if data.get("type") == "list":
                producers = data.get("producers") or []
                producer_count = max(producer_count, len(producers))
                producer_ids = [_producer_id(producer) for producer in producers]
                producer_ids = [
                    producer_id for producer_id in producer_ids if producer_id
                ]
                if producer_count:
                    return SignalStatus(
                        connected=True,
                        producer_count=producer_count,
                        producer_ids=producer_ids,
                        last_message=last_message,
                        latency_ms=_elapsed_ms(started_at),
                    )
            if data.get("type") == "peerStatusChanged" and "producer" in data.get(
                "roles", []
            ):
                producer_id = _producer_id(data)
                return SignalStatus(
                    connected=True,
                    producer_count=1,
                    producer_ids=[producer_id] if producer_id else [],
                    last_message=last_message,
                    latency_ms=_elapsed_ms(started_at),
                )
        return SignalStatus(
            connected=True,
            producer_count=producer_count,
            producer_ids=producer_ids,
            last_message=last_message or "no_message",
            latency_ms=_elapsed_ms(started_at),
        )
    except (
        ConnectionRefusedError,
        ConnectionClosed,
        InvalidURI,
        InvalidHandshake,
        OSError,
        asyncio.TimeoutError,
    ) as exc:
        return SignalStatus(
            connected=False,
            producer_count=0,
            last_message=type(exc).__name__,
            last_error=str(exc),
            latency_ms=_elapsed_ms(started_at),
        )
    finally:
        if websocket:
            try:
                await websocket.close()
            except Exception:
                pass


class ZedNode(Node):
    def __init__(self) -> None:
        super().__init__("zed")

        self.declare_parameter("jetson_host", "10.10.20.126")
        self.declare_parameter("jetson_user", "pnd-humanoid")
        self.declare_parameter(
            "jetson_webrtc_root",
            "/home/pnd-humanoid/Documents/pnd_teleoperation/external",
        )
        self.declare_parameter("webrtc_port", DEFAULT_WEBRTC_PORT)
        self.declare_parameter("start_remote_pipeline", True)
        self.declare_parameter("video_bitrate", DEFAULT_VIDEO_BITRATE)
        self.declare_parameter("video_fps", DEFAULT_VIDEO_FPS)
        self.declare_parameter("video_width", DEFAULT_VIDEO_WIDTH)
        self.declare_parameter("video_height", DEFAULT_VIDEO_HEIGHT)
        self.declare_parameter("video_layout", "mono")
        self.declare_parameter("monitor_stream_enabled", True)
        self.declare_parameter("monitor_stream_transport", "rtp")
        self.declare_parameter("monitor_stream_host", DEFAULT_MONITOR_STREAM_HOST)
        self.declare_parameter("monitor_stream_port", DEFAULT_MONITOR_STREAM_PORT)
        self.declare_parameter("inference_stream_enabled", True)
        self.declare_parameter("inference_stream_host", DEFAULT_INFERENCE_STREAM_HOST)
        self.declare_parameter("inference_stream_port", DEFAULT_INFERENCE_STREAM_PORT)
        self.declare_parameter("quest_stream_enabled", False)
        self.declare_parameter("quest_stream_bind_host", DEFAULT_QUEST_STREAM_BIND_HOST)
        self.declare_parameter("quest_stream_port", DEFAULT_QUEST_STREAM_PORT)
        self.declare_parameter("watchdog_enabled", True)
        self.declare_parameter(
            "watchdog_failure_threshold", DEFAULT_WATCHDOG_FAILURE_THRESHOLD
        )
        self.declare_parameter(
            "watchdog_restart_cooldown_s", DEFAULT_WATCHDOG_RESTART_COOLDOWN_S
        )
        self.declare_parameter("browser_ui_enabled", True)
        self.declare_parameter("browser_ui_host", DEFAULT_BROWSER_UI_HOST)
        self.declare_parameter("browser_ui_port", DEFAULT_BROWSER_UI_PORT)
        self.declare_parameter("browser_ui_web_root", "")
        self.declare_parameter("status_topic", "/zed/status")

        self.jetson_host = str(self.get_parameter("jetson_host").value)
        self.jetson_user = str(self.get_parameter("jetson_user").value)
        self.webrtc_port = int(self.get_parameter("webrtc_port").value)
        self.start_remote_pipeline = _bool_value(
            self.get_parameter("start_remote_pipeline").value
        )
        self.video_width = int(self.get_parameter("video_width").value)
        self.video_height = int(self.get_parameter("video_height").value)
        self.video_fps = int(self.get_parameter("video_fps").value)
        self.video_bitrate = int(self.get_parameter("video_bitrate").value)
        self.video_layout = str(self.get_parameter("video_layout").value).strip()
        self.monitor_stream_enabled = _bool_value(
            self.get_parameter("monitor_stream_enabled").value
        )
        self.monitor_stream_transport = str(
            self.get_parameter("monitor_stream_transport").value
        ).strip()
        self.monitor_stream_host = str(
            self.get_parameter("monitor_stream_host").value
        ).strip()
        self.monitor_stream_port = int(self.get_parameter("monitor_stream_port").value)
        self.inference_stream_enabled = _bool_value(
            self.get_parameter("inference_stream_enabled").value
        )
        self.inference_stream_host = str(
            self.get_parameter("inference_stream_host").value
        ).strip()
        self.inference_stream_port = int(
            self.get_parameter("inference_stream_port").value
        )
        self.quest_stream_enabled = _bool_value(
            self.get_parameter("quest_stream_enabled").value
        )
        self.quest_stream_bind_host = str(
            self.get_parameter("quest_stream_bind_host").value
        ).strip()
        self.quest_stream_port = int(self.get_parameter("quest_stream_port").value)
        self.watchdog_enabled = _bool_value(
            self.get_parameter("watchdog_enabled").value
        )
        self.watchdog_failure_threshold = int(
            self.get_parameter("watchdog_failure_threshold").value
        )
        self.watchdog_restart_cooldown_s = float(
            self.get_parameter("watchdog_restart_cooldown_s").value
        )
        self.browser_ui_enabled = _bool_value(
            self.get_parameter("browser_ui_enabled").value
        )
        self.browser_ui_host = str(self.get_parameter("browser_ui_host").value).strip()
        self.browser_ui_port = int(self.get_parameter("browser_ui_port").value)
        self.browser_ui_web_root = str(self.get_parameter("browser_ui_web_root").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("video_width and video_height must be positive")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if self.video_bitrate <= 0:
            raise ValueError("video_bitrate must be positive")
        if self.video_layout not in VIDEO_LAYOUTS:
            raise ValueError(
                f"video_layout must be one of {sorted(VIDEO_LAYOUTS)}, "
                f"got {self.video_layout!r}"
            )
        if self.monitor_stream_enabled and not self.monitor_stream_host:
            raise ValueError(
                "monitor_stream_host must be non-empty when monitor stream is enabled"
            )
        if self.monitor_stream_transport != "rtp":
            raise ValueError("monitor_stream_transport must remain 'rtp' for recording")
        if self.monitor_stream_port <= 0 or self.monitor_stream_port > 65535:
            raise ValueError("monitor_stream_port must be in [1, 65535]")
        if self.inference_stream_enabled and not self.inference_stream_host:
            raise ValueError(
                "inference_stream_host must be non-empty when inference stream is enabled"
            )
        if self.inference_stream_port <= 0 or self.inference_stream_port > 65535:
            raise ValueError("inference_stream_port must be in [1, 65535]")
        if self.quest_stream_enabled and not self.quest_stream_bind_host:
            raise ValueError(
                "quest_stream_bind_host must be non-empty when Quest stream is enabled"
            )
        if self.quest_stream_port <= 0 or self.quest_stream_port > 65535:
            raise ValueError("quest_stream_port must be in [1, 65535]")
        if self.watchdog_failure_threshold <= 0:
            raise ValueError("watchdog_failure_threshold must be positive")
        if self.watchdog_restart_cooldown_s <= 0.0:
            raise ValueError("watchdog_restart_cooldown_s must be positive")
        if self.browser_ui_enabled and not self.browser_ui_host:
            raise ValueError(
                "browser_ui_host must be non-empty when browser UI is enabled"
            )
        if self.browser_ui_port <= 0 or self.browser_ui_port > 65535:
            raise ValueError("browser_ui_port must be in [1, 65535]")

        self.remote_config = RemoteConfig(
            host=self.jetson_host,
            user=self.jetson_user,
            webrtc_port=self.webrtc_port,
            root=str(self.get_parameter("jetson_webrtc_root").value),
            width=self.video_width,
            height=self.video_height,
            fps=self.video_fps,
            bitrate=self.video_bitrate,
            video_layout=self.video_layout,
            monitor_stream_enabled=self.monitor_stream_enabled,
            monitor_stream_host=self.monitor_stream_host,
            monitor_stream_port=self.monitor_stream_port,
            inference_stream_enabled=self.inference_stream_enabled,
            inference_stream_host=self.inference_stream_host,
            inference_stream_port=self.inference_stream_port,
            quest_stream_enabled=self.quest_stream_enabled,
            quest_stream_bind_host=self.quest_stream_bind_host,
            quest_stream_port=self.quest_stream_port,
        )
        self.remote_process = RemoteProcess()
        self.lock = threading.Lock()
        self.signal_status = SignalStatus()
        self.startup_action = "not_started"
        self.encoder = "none"
        self.pipeline_alive = False
        self.last_error = ""
        self.last_warning = ""
        self.signal_failure_count = 0
        self.watchdog_restart_in_progress = False
        self.last_watchdog_restart_at = 0.0
        self.last_watchdog_restart_reason = ""
        self.browser_ui: BrowserUiServer | None = None
        self.browser_ui_error = ""

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        if self.browser_ui_enabled:
            self._start_browser_ui()

        if self.start_remote_pipeline:
            self._start_remote_pipeline()
        else:
            self.startup_action = "remote_start_disabled"

        self.create_timer(1.0, self._poll_signal_status)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            "zed source ready; "
            f"webrtc=webrtc://{self.jetson_host}:{self.webrtc_port}; "
            f"monitor_rtp={self.monitor_stream_host}:{self.monitor_stream_port}; "
            f"inference_rtp={self.inference_stream_host}:{self.inference_stream_port}; "
            f"quest_tcp={self.quest_stream_bind_host}:{self.quest_stream_port} "
            f"enabled={self.quest_stream_enabled}; "
            "egoview=/egoview; "
            f"video={self.video_width}x{self.video_height}@{self.video_fps} "
            f"layout={self.video_layout} "
            f"bitrate={self.video_bitrate}"
        )

    def _start_browser_ui(self) -> None:
        try:
            self.browser_ui = BrowserUiServer(
                host=self.browser_ui_host,
                port=self.browser_ui_port,
                web_root=self.browser_ui_web_root,
                status_provider=self.status_payload,
            )
            self.browser_ui.start()
            self.browser_ui_error = ""
        except Exception as exc:
            self.browser_ui = None
            self.browser_ui_error = str(exc)
            self.get_logger().error(f"ZED EgoView browser UI failed to start: {exc}")

    def _start_remote_pipeline(self) -> bool:
        try:
            self.startup_action = "hardware_start_command_sent"
            self.encoder = "hardware_h264_attempting"
            self.last_error = ""
            stop_result = stop_video_processes(self.remote_config)
            if stop_result.returncode != 0:
                self.last_warning = (
                    "remote ZED cleanup failed: "
                    f"stdout={stop_result.stdout.strip()} "
                    f"stderr={stop_result.stderr.strip()}"
                )
                self.get_logger().warning(self.last_warning)
            check_hardware(self.remote_config)
            start_remote(
                self.remote_config,
                hardware_command(self.remote_config),
                self.remote_process,
            )
            time.sleep(0.3)
            returncode = self.remote_process.returncode()
            if returncode is not None:
                raise RuntimeError(
                    f"remote ZED pipeline exited during startup: {returncode}"
                )
            signal = asyncio.run(self._wait_for_signal(max_wait_s=6.0))
            if signal.has_producer:
                with self.lock:
                    self.signal_status = signal
                    self.pipeline_alive = True
                    self.startup_action = "hardware_ready"
                    self.encoder = "hardware_h264"
                return True
            with self.lock:
                self.signal_status = signal
                self.pipeline_alive = False
                self.startup_action = "hardware_no_producer"
                self.encoder = "hardware_h264"
                self.last_error = (
                    signal.last_error or "hardware WebRTC producer not found"
                )
            return False
        except Exception as exc:
            with self.lock:
                self.pipeline_alive = False
                self.startup_action = "start_exception"
                self.encoder = "error"
                self.last_error = str(exc)
            self.get_logger().error(f"ZED pipeline startup failed: {exc}")
            return False

    async def _wait_for_signal(self, max_wait_s: float) -> SignalStatus:
        deadline = time.monotonic() + max_wait_s
        last = SignalStatus(last_message="starting")
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            last = await check_webrtc_signal(self.jetson_host, self.webrtc_port)
            if last.has_producer:
                return last
        return last

    def _poll_signal_status(self) -> None:
        remote_returncode = (
            self.remote_process.returncode() if self.start_remote_pipeline else None
        )
        try:
            signal = asyncio.run(
                check_webrtc_signal(self.jetson_host, self.webrtc_port)
            )
        except Exception as exc:
            signal = SignalStatus(
                connected=False,
                producer_count=0,
                last_message=type(exc).__name__,
                last_error=str(exc),
            )
        if not signal.has_producer and remote_returncode is not None:
            signal.last_message = "remote_process_exited"
            signal.last_error = (
                f"remote ZED pipeline exited with code {remote_returncode}"
            )
        restart_reason = ""
        with self.lock:
            self.signal_status = signal
            self.pipeline_alive = signal.has_producer
            if signal.has_producer:
                self.signal_failure_count = 0
                if self.encoder != "error":
                    self.last_error = ""
            else:
                self.signal_failure_count += 1
                if signal.last_error:
                    self.last_error = signal.last_error
                elif self.encoder != "error":
                    self.last_error = "WebRTC producer not found"
                restart_reason = self._watchdog_restart_reason_locked(signal)
        if restart_reason:
            self._trigger_watchdog_restart(restart_reason)

    def _watchdog_restart_reason_locked(self, signal: SignalStatus) -> str:
        if not self.watchdog_enabled or not self.start_remote_pipeline:
            return ""
        if self.watchdog_restart_in_progress:
            return ""
        if self.signal_failure_count < self.watchdog_failure_threshold:
            return ""
        now = time.monotonic()
        if now - self.last_watchdog_restart_at < self.watchdog_restart_cooldown_s:
            return ""
        detail = signal.last_error or signal.last_message or "producer missing"
        return (
            f"no WebRTC producer after {self.signal_failure_count} checks; "
            f"last={detail}"
        )

    def _trigger_watchdog_restart(self, reason: str) -> None:
        with self.lock:
            if self.watchdog_restart_in_progress:
                return
            self.watchdog_restart_in_progress = True
            self.last_watchdog_restart_at = time.monotonic()
            self.last_watchdog_restart_reason = reason
            self.startup_action = "watchdog_restart_command_sent"
            self.last_warning = reason
        self.get_logger().warning(f"ZED watchdog restarting remote pipeline: {reason}")
        threading.Thread(
            target=self._watchdog_restart_remote_pipeline, daemon=True
        ).start()

    def _watchdog_restart_remote_pipeline(self) -> None:
        try:
            self._start_remote_pipeline()
        finally:
            with self.lock:
                self.watchdog_restart_in_progress = False
                if self.pipeline_alive:
                    self.signal_failure_count = 0

    def status_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            signal = self.signal_status
            pipeline_alive = self.pipeline_alive
            startup_action = self.startup_action
            encoder = self.encoder
            last_error = self.last_error
            last_warning = self.last_warning
        return {
            "ok": True,
            "node": "zed",
            "scope": "source_only",
            "sample_lifecycle_owner": "monitor",
            "pipeline_alive": pipeline_alive,
            "encoder": encoder,
            "startup_action": startup_action,
            "jetson_host": self.jetson_host,
            "webrtc_output": f"webrtc://{self.jetson_host}:{self.webrtc_port}",
            "video": {
                "width": self.video_width,
                "height": self.video_height,
                "encoded_height": (
                    self.video_height * 2
                    if self.video_layout == "top-bottom"
                    else self.video_height
                ),
                "layout": self.video_layout,
                "fps": self.video_fps,
                "bitrate": self.video_bitrate,
                "codec": "h264",
                "monitor_stream": {
                    "enabled": self.monitor_stream_enabled,
                    "transport": "rtp/udp",
                    "host": self.monitor_stream_host,
                    "port": self.monitor_stream_port,
                    "payload_type": RTP_PAYLOAD_TYPE,
                    "clock_rate": 90000,
                },
                "inference_stream": {
                    "enabled": self.inference_stream_enabled,
                    "transport": "rtp/udp",
                    "host": self.inference_stream_host,
                    "port": self.inference_stream_port,
                    "payload_type": RTP_PAYLOAD_TYPE,
                    "clock_rate": 90000,
                },
                "quest_stream": {
                    "enabled": self.quest_stream_enabled,
                    "transport": "mpegts/tcp",
                    "bind_host": self.quest_stream_bind_host,
                    "port": self.quest_stream_port,
                    "client_url": (
                        f"tcp://{self.jetson_host}:{self.quest_stream_port}"
                    ),
                },
            },
            "webrtc": {
                "signal_connected": signal.connected,
                "producer_count": signal.producer_count,
                "producer_ids": signal.producer_ids,
                "check_latency_ms": signal.latency_ms,
                "last_message": signal.last_message,
                "last_error": signal.last_error,
            },
            "watchdog": {
                "enabled": self.watchdog_enabled,
                "failure_count": self.signal_failure_count,
                "failure_threshold": self.watchdog_failure_threshold,
                "restart_in_progress": self.watchdog_restart_in_progress,
                "restart_cooldown_s": self.watchdog_restart_cooldown_s,
                "last_restart_age_ms": _age_ms(
                    self.last_watchdog_restart_at or None, now
                ),
                "last_restart_reason": self.last_watchdog_restart_reason,
            },
            "remote_process": {
                "returncode": self.remote_process.returncode(),
            },
            "browser_ui": self.browser_ui.status_payload()
            if self.browser_ui is not None
            else {
                "enabled": self.browser_ui_enabled,
                "running": False,
                "host": self.browser_ui_host,
                "port": self.browser_ui_port,
                "web_root": self.browser_ui_web_root,
                "egoview_url": "/egoview",
                "internal_url": f"http://{self.browser_ui_host}:{self.browser_ui_port}/egoview",
                "nginx_path": "/egoview",
                "last_error": self.browser_ui_error,
            },
            "last_warning": last_warning,
            "last_error": last_error,
        }

    def _publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(
            self.status_payload(), separators=(",", ":"), ensure_ascii=True
        )
        self.status_pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            if self.browser_ui is not None:
                self.browser_ui.close()
            if self.start_remote_pipeline:
                result = stop_video_processes(self.remote_config)
                if result.returncode != 0:
                    self.get_logger().warning(
                        "remote ZED cleanup on shutdown failed: "
                        f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
                    )
            self.remote_process.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: ZedNode | None = None
    try:
        node = ZedNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError as exc:
        if "rcl node's context is invalid" not in str(exc):
            raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
