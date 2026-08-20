"""Quest WebVR receiver with ROS TF/Joy output and source freshness tracking."""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import rclpy
import websockets
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from websockets.exceptions import ConnectionClosed

from quest_node.webvr_protocol import (
    POSE_NAMES,
    HandExecutionGate,
    Pose,
    Quaternion,
    Vector3,
    WebVRProtocolError,
    WebVRSample,
    calibration_from_sample,
    flatten_joy_buttons,
    hand_execution_is_ready,
    hand_position_states,
    parse_webvr_message,
    pose_status,
    validate_zero_pose_sample,
    vr_pose_to_ros,
)
from quest_node.webvr_security import (
    WebVRSecurityError,
    authenticate_first_message,
    generate_access_token,
    normalize_public_web_url,
    validate_access_token,
    validate_secure_same_origin,
)
from quest_node.webvr_state import (
    ReceiverSnapshot,
    WebVRReceiverState,
    calibration_is_stale,
)


CALIBRATE_BUTTON = 4
DECALIBRATE_BUTTON = 5


class _QuestRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: object,
        runtime_config: dict[str, object],
        directory: str,
        **kwargs: object,
    ) -> None:
        self.runtime_config = runtime_config
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/runtime-config.json":
            payload = json.dumps(
                self.runtime_config,
                separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


class WebVRHTTPServer:
    def __init__(
        self,
        host: str,
        port: int,
        web_root: Path,
        *,
        runtime_config: dict[str, object],
    ) -> None:
        handler = functools.partial(
            _QuestRequestHandler,
            runtime_config=runtime_config,
            directory=str(web_root),
        )
        self._server = http.server.ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="quest_webvr_http",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if self._started:
            self._server.shutdown()
            self._thread.join(timeout=3.0)
            self._started = False
        self._server.server_close()


class WebVRSocketServer:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        access_token: str,
        authentication_timeout: float,
        on_connection: Callable[[bool], None],
        on_sample: Callable[[WebVRSample, float], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.host = host
        self.port = port
        self.access_token = access_token
        self.authentication_timeout = authentication_timeout
        self.on_connection = on_connection
        self.on_sample = on_sample
        self.on_error = on_error
        self._clients: set[object] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._start_error: BaseException | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._thread_main,
            name="quest_webvr_websocket",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise RuntimeError("timed out starting Quest WebSocket server")
        if self._start_error is not None:
            raise RuntimeError(
                f"failed to start Quest WebSocket server: {self._start_error}"
            ) from self._start_error

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as exc:  # noqa: BLE001 - propagate thread startup errors.
            self._start_error = exc
            self._started.set()

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=64 * 1024,
            max_queue=4,
            ping_interval=10.0,
            ping_timeout=10.0,
        ):
            self._started.set()
            await self._stop_event.wait()

    @staticmethod
    def _request_headers(websocket: object) -> object:
        request = getattr(websocket, "request", None)
        headers = getattr(request, "headers", None)
        if headers is not None:
            return headers
        return getattr(websocket, "request_headers", {})

    @staticmethod
    async def _reject_client(
        websocket: object,
        *,
        error: str,
        message: str,
    ) -> None:
        try:
            await websocket.send(json.dumps({"error": error, "message": message}))
            await websocket.close(code=1008, reason=message[:120])
        except ConnectionClosed:
            pass

    async def _handle_client(self, websocket: object, _path: object = None) -> None:
        try:
            validate_secure_same_origin(self._request_headers(websocket))
        except WebVRSecurityError as exc:
            self.on_error(str(exc))
            await self._reject_client(
                websocket,
                error="ORIGIN_REJECTED",
                message="Quest WebVR requires the HTTPS page on this PND.",
            )
            return

        try:
            first_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=self.authentication_timeout,
            )
        except asyncio.TimeoutError:
            self.on_error("Quest WebSocket authentication timed out")
            await self._reject_client(
                websocket,
                error="AUTH_TIMEOUT",
                message="Quest access-token authentication timed out.",
            )
            return
        except ConnectionClosed:
            return

        try:
            authenticate_first_message(first_message, self.access_token)
        except WebVRSecurityError as exc:
            self.on_error(str(exc))
            await self._reject_client(
                websocket,
                error="AUTH_FAILED",
                message="Quest access token is invalid.",
            )
            return

        if self._clients:
            await self._reject_client(
                websocket,
                error="VR_HEADSET_LIMIT",
                message="Only one VR headset connection is allowed.",
            )
            return

        self._clients.add(websocket)
        self.on_connection(True)
        try:
            await websocket.send(json.dumps({"type": "auth_ok"}))
            async for message in websocket:
                received_at = time.monotonic()
                try:
                    sample = parse_webvr_message(message)
                except WebVRProtocolError as exc:
                    self.on_error(str(exc))
                    continue
                self.on_sample(sample, received_at)
        except ConnectionClosed:
            pass
        except Exception as exc:  # noqa: BLE001 - isolate a browser connection.
            self.on_error(f"WebSocket client error: {exc}")
        finally:
            self._clients.discard(websocket)
            self.on_connection(False)

    async def _broadcast_event(self, payload: dict[str, object]) -> None:
        message = json.dumps(payload, separators=(",", ":"))
        for websocket in tuple(self._clients):
            try:
                await websocket.send(message)
            except ConnectionClosed:
                pass

    def send_event(self, payload: dict[str, object]) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_event(payload),
            self._loop,
        )

    def stop(self) -> None:
        if (
            self._loop is not None
            and not self._loop.is_closed()
            and self._stop_event is not None
        ):
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class QuestWebVRNode(Node):
    def __init__(self) -> None:
        super().__init__("quest_webvr")
        self.declare_parameter("http_host", "127.0.0.1")
        self.declare_parameter("http_port", 8443)
        self.declare_parameter("websocket_host", "127.0.0.1")
        self.declare_parameter("websocket_port", 8442)
        self.declare_parameter("access_token", "")
        self.declare_parameter("authentication_timeout", 3.0)
        self.declare_parameter("public_web_url", "https://10.10.20.127/webvr/")
        self.declare_parameter("web_root", "")
        self.declare_parameter("video_layout", "mono")
        self.declare_parameter("video_swap_eyes", False)
        self.declare_parameter("video_enabled", True)
        self.declare_parameter("turn_enabled", True)
        self.declare_parameter("turn_host", "10.10.20.127")
        self.declare_parameter("turn_port", 3478)
        self.declare_parameter("turn_username", "quest-video")
        self.declare_parameter("turn_password", "")
        self.declare_parameter("turn_relay_min_port", 49160)
        self.declare_parameter("turn_relay_max_port", 49200)
        self.declare_parameter("joy_topic", "/_quest/joy")
        self.declare_parameter("status_topic", "/quest/webvr_status")
        self.declare_parameter("calibration_service", "/quest/calibrate")
        self.declare_parameter(
            "retarget_warm_start_service",
            "",
        )
        self.declare_parameter("poll_rate", 100.0)
        self.declare_parameter("status_rate", 2.0)
        self.declare_parameter("input_timeout", 0.2)
        self.declare_parameter("calibration_stale_timeout", 1.0)
        self.declare_parameter("position_jump_threshold", 0.15)
        self.declare_parameter("tracking_recovery_frames", 3)
        self.declare_parameter("tracking_recovery_motion_threshold", 0.03)
        self.declare_parameter("robot_arm_length", 0.53)
        self.declare_parameter("calibration_mode", "arms_forward")

        self.http_host = str(self.get_parameter("http_host").value)
        self.http_port = int(self.get_parameter("http_port").value)
        self.websocket_host = str(self.get_parameter("websocket_host").value)
        self.websocket_port = int(self.get_parameter("websocket_port").value)
        configured_access_token = str(self.get_parameter("access_token").value).strip()
        self.access_token = validate_access_token(
            configured_access_token or generate_access_token()
        )
        self.authentication_timeout = float(
            self.get_parameter("authentication_timeout").value
        )
        self.public_web_url = normalize_public_web_url(
            str(self.get_parameter("public_web_url").value)
        )
        self.video_layout = str(self.get_parameter("video_layout").value).strip()
        self.video_swap_eyes = bool(self.get_parameter("video_swap_eyes").value)
        self.video_enabled = bool(self.get_parameter("video_enabled").value)
        if self.video_layout not in {"mono", "top-bottom"}:
            raise ValueError("video_layout must be 'mono' or 'top-bottom'")
        self.turn_enabled = bool(self.get_parameter("turn_enabled").value)
        self.turn_host = str(self.get_parameter("turn_host").value).strip()
        self.turn_port = int(self.get_parameter("turn_port").value)
        self.turn_username = str(self.get_parameter("turn_username").value).strip()
        configured_turn_password = str(
            self.get_parameter("turn_password").value
        ).strip()
        self.turn_password = configured_turn_password or generate_access_token()
        self.turn_relay_min_port = int(self.get_parameter("turn_relay_min_port").value)
        self.turn_relay_max_port = int(self.get_parameter("turn_relay_max_port").value)
        self.input_timeout = float(self.get_parameter("input_timeout").value)
        self.calibration_stale_timeout = float(
            self.get_parameter("calibration_stale_timeout").value
        )
        self.position_jump_threshold = float(
            self.get_parameter("position_jump_threshold").value
        )
        self.tracking_recovery_frames = int(
            self.get_parameter("tracking_recovery_frames").value
        )
        self.tracking_recovery_motion_threshold = float(
            self.get_parameter("tracking_recovery_motion_threshold").value
        )
        self.robot_arm_length = float(self.get_parameter("robot_arm_length").value)
        self.calibration_mode = str(
            self.get_parameter("calibration_mode").value
        ).strip()
        if self.calibration_mode not in {"arms_forward", "zero_pose"}:
            raise ValueError("calibration_mode must be arms_forward or zero_pose")
        poll_rate = float(self.get_parameter("poll_rate").value)
        status_rate = float(self.get_parameter("status_rate").value)
        if (
            self.input_timeout <= 0.0
            or self.calibration_stale_timeout < self.input_timeout
            or self.position_jump_threshold <= 0.0
            or self.tracking_recovery_frames <= 0
            or self.tracking_recovery_motion_threshold <= 0.0
            or self.authentication_timeout <= 0.0
            or poll_rate <= 0.0
            or status_rate <= 0.0
        ):
            raise ValueError(
                "authentication_timeout, input_timeout, poll_rate and status_rate "
                "must be positive, and position_jump_threshold must be positive, "
                "tracking_recovery_frames must be positive, and "
                "tracking_recovery_motion_threshold must be positive, and "
                "calibration_stale_timeout must be at least input_timeout"
            )
        if self.turn_enabled and (
            not self.turn_host
            or not self.turn_username
            or self.turn_port <= 0
            or self.turn_port > 65535
            or self.turn_relay_min_port <= 0
            or self.turn_relay_max_port > 65535
            or self.turn_relay_min_port > self.turn_relay_max_port
        ):
            raise ValueError("invalid Quest TURN relay configuration")

        web_root_value = str(self.get_parameter("web_root").value).strip()
        self.web_root = (
            Path(web_root_value).expanduser()
            if web_root_value
            else Path(get_package_share_directory("quest_node")) / "web"
        )
        if not (self.web_root / "index.html").is_file():
            raise FileNotFoundError(
                f"Quest WebVR index not found under {self.web_root}"
            )

        self._receiver_state = WebVRReceiverState()
        self._processed_connection_version = 0
        self._processed_disconnect_epoch = 0
        self._processed_sequence = 0
        self._last_published_receive_time: float | None = None
        self._last_browser_timestamp_ms: float | None = None
        self._tracking_quality_ready = False
        self._hand_position_states = {
            "LeftHand": "Unavailable",
            "RightHand": "Unavailable",
        }
        self._hand_execution_states = {
            "LeftHand": "Normal",
            "RightHand": "Normal",
        }
        self._hand_execution_updated_at: dict[str, float | None] = {
            "LeftHand": None,
            "RightHand": None,
        }
        self._hand_gates = {
            "LeftHand": HandExecutionGate("LeftHand"),
            "RightHand": HandExecutionGate("RightHand"),
        }
        self._execution_poses: dict[str, Pose] = {}
        self._published_frames = 0
        self._last_error = ""
        self._calibrated = False
        self._calibration_requested = False
        self._calibrate_button_pressed = False
        self._decalibrate_button_pressed = False
        self._robot_scale = 1.0
        self._position_rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self._position_offsets = {name: Vector3(0.0, 0.0, 0.0) for name in POSE_NAMES}
        self._quaternion_offsets = {
            name: Quaternion(0.0, 0.0, 0.0, 1.0) for name in POSE_NAMES
        }
        self._servers_stopped = False
        self._turn_process: subprocess.Popen[bytes] | None = None
        self._turn_log_path = Path("/tmp/quest-turn.log")

        self._tf_broadcaster = TransformBroadcaster(self)
        self._joy_publisher = self.create_publisher(
            Joy,
            str(self.get_parameter("joy_topic").value),
            10,
        )
        self._status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self._calibration_service = self.create_service(
            Trigger,
            str(self.get_parameter("calibration_service").value),
            self._request_calibration,
        )
        warm_start_service = str(
            self.get_parameter("retarget_warm_start_service").value
        ).strip()
        self._retarget_warm_start_client = (
            self.create_client(Trigger, warm_start_service)
            if warm_start_service
            else None
        )

        ice_servers = []
        if self.turn_enabled:
            ice_servers.append(
                {
                    "urls": [f"turn:{self.turn_host}:{self.turn_port}?transport=udp"],
                    "username": self.turn_username,
                    "credential": self.turn_password,
                }
            )
        runtime_config = {
            "accessToken": self.access_token,
            "iceServers": ice_servers,
            "iceTransportPolicy": "relay" if self.turn_enabled else "all",
            "videoEnabled": self.video_enabled,
            "videoLayout": self.video_layout,
            "videoSwapEyes": self.video_swap_eyes,
        }
        self._http_server = WebVRHTTPServer(
            self.http_host,
            self.http_port,
            self.web_root,
            runtime_config=runtime_config,
        )
        self._socket_server = WebVRSocketServer(
            self.websocket_host,
            self.websocket_port,
            access_token=self.access_token,
            authentication_timeout=self.authentication_timeout,
            on_connection=self._on_connection,
            on_sample=self._on_sample,
            on_error=self._on_protocol_error,
        )
        try:
            self._start_turn_relay()
            self._http_server.start()
            self._socket_server.start()
        except Exception:
            self._stop_servers()
            raise

        self.create_timer(1.0 / poll_rate, self._poll_receiver)
        self.create_timer(1.0 / status_rate, self._publish_periodic_status)
        self.get_logger().info(
            "Quest WebVR ready: "
            f"http={self.http_host}:{self.http_port}, "
            f"websocket={self.websocket_host}:{self.websocket_port}, "
            f"web_root={self.web_root}, input_timeout={self.input_timeout:.3f}s, "
            f"position_jump_threshold={self.position_jump_threshold:.3f}m, "
            f"tracking_recovery_frames={self.tracking_recovery_frames}"
            f", tracking_recovery_motion_threshold="
            f"{self.tracking_recovery_motion_threshold:.3f}m"
        )
        self.get_logger().info(f"Quest access URL: {self.public_web_url}")

    def _start_turn_relay(self) -> None:
        if not self.turn_enabled:
            return
        executable = shutil.which("turnserver")
        if executable is None:
            raise RuntimeError("coturn is required for Quest ZED video relay")
        try:
            self._turn_log_path.unlink(missing_ok=True)
        except OSError:
            pass
        pid = os.getpid()
        command = [
            executable,
            "-n",
            f"--listening-ip={self.turn_host}",
            f"--relay-ip={self.turn_host}",
            f"--listening-port={self.turn_port}",
            f"--min-port={self.turn_relay_min_port}",
            f"--max-port={self.turn_relay_max_port}",
            "--fingerprint",
            "--lt-cred-mech",
            "--realm=quest.local",
            f"--user={self.turn_username}:{self.turn_password}",
            "--no-tls",
            "--no-dtls",
            "--no-cli",
            "--no-multicast-peers",
            f"--pidfile=/tmp/quest-turn-{pid}.pid",
            f"--db=/tmp/quest-turn-{pid}.sqlite",
            f"--log-file={self._turn_log_path}",
            "--simple-log",
            "--no-stdout-log",
        ]
        self._turn_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        time.sleep(0.25)
        if self._turn_process.poll() is not None:
            detail = ""
            try:
                detail = self._turn_log_path.read_text(encoding="utf-8")[-1000:]
            except OSError:
                pass
            self._turn_process = None
            raise RuntimeError(f"Quest TURN relay failed to start: {detail}")
        self.get_logger().info(
            "Quest TURN relay ready: "
            f"turn={self.turn_host}:{self.turn_port}, "
            f"relay_ports={self.turn_relay_min_port}-{self.turn_relay_max_port}"
        )

    def _stop_turn_relay(self) -> None:
        process = self._turn_process
        self._turn_process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def _on_connection(self, connected: bool) -> None:
        self._receiver_state.set_connection(connected)

    def _on_sample(self, sample: WebVRSample, received_at: float) -> None:
        self._receiver_state.record_sample(sample, received_at)

    def _on_protocol_error(self, reason: str) -> None:
        self._receiver_state.record_error(reason)

    def _snapshot(self) -> ReceiverSnapshot:
        return self._receiver_state.snapshot()

    def _poll_receiver(self) -> None:
        snapshot = self._snapshot()
        disconnect_seen = snapshot.disconnect_epoch != self._processed_disconnect_epoch
        if disconnect_seen:
            self._processed_disconnect_epoch = snapshot.disconnect_epoch
            self._reset_calibration()
            self._reset_tracking_quality()
            self._reset_button_edges()
            self._last_published_receive_time = None
            self._last_browser_timestamp_ms = None
            self._last_error = "Quest browser connection changed; calibration reset"
            self._publish_decalibration_joy()
            if snapshot.connected:
                self.get_logger().warning(
                    "Quest browser reconnected; calibration reset, press A again"
                )
            else:
                self.get_logger().warning(
                    "Quest browser disconnected; command output blocked"
                )
            self._publish_status(
                "disconnect",
                snapshot=snapshot,
                tracking_fresh=False,
            )

        if snapshot.connection_version != self._processed_connection_version:
            self._processed_connection_version = snapshot.connection_version
            if snapshot.connected and not disconnect_seen:
                self.get_logger().info("Quest browser connected; press A to calibrate")

        if (
            not snapshot.connected
            or snapshot.sequence == self._processed_sequence
            or snapshot.sample is None
            or snapshot.received_at is None
        ):
            return

        age = time.monotonic() - snapshot.received_at
        if age > self.input_timeout:
            self._processed_sequence = snapshot.sequence
            self._last_error = f"dropped stale browser frame age={age:.3f}s"
            self._publish_status("frame", snapshot=snapshot, tracking_fresh=False)
            return
        processed = self._receiver_state.process_if_current(
            snapshot,
            lambda: self._process_sample(
                snapshot.sample,
                snapshot.received_at,
                snapshot,
            ),
        )
        if processed:
            self._processed_sequence = snapshot.sequence

    def _process_sample(
        self,
        sample: WebVRSample,
        received_at: float,
        snapshot: ReceiverSnapshot,
    ) -> None:
        self._last_error = ""
        joy = Joy()
        joy.axes = list(sample.joy_axes)
        joy.buttons = flatten_joy_buttons(sample.joy_buttons)

        calibration_requested = self._calibration_requested
        self._calibration_requested = False
        if calibration_requested:
            joy.buttons[CALIBRATE_BUTTON] = 1
        calibrate_pressed = sample.calibration_pressed or calibration_requested
        joy.buttons[CALIBRATE_BUTTON] = int(calibrate_pressed)
        decalibrate_pressed = joy.buttons[DECALIBRATE_BUTTON] == 1
        calibrate_edge = calibration_requested or (
            calibrate_pressed and not self._calibrate_button_pressed
        )
        decalibrate_edge = decalibrate_pressed and not self._decalibrate_button_pressed

        self._observe_hand_tracking(sample, calibrate_edge, received_at)

        if calibrate_edge:
            was_calibrated = self._calibrated
            try:
                calibration_sample = self._calibration_sample(sample)
                if self.calibration_mode == "zero_pose":
                    validate_zero_pose_sample(calibration_sample)
                calibration = (
                    calibration_from_sample(
                        calibration_sample,
                        robot_arm_length=self.robot_arm_length,
                    )
                    if self.calibration_mode == "arms_forward"
                    else None
                )
            except WebVRProtocolError as exc:
                joy.buttons[CALIBRATE_BUTTON] = 0
                self._last_error = str(exc)
                self.get_logger().warning(f"Quest calibration rejected: {exc}")
                self._send_calibration_event(
                    "rejected",
                    message=str(exc),
                )
            else:
                if calibration is None:
                    self._robot_scale = 1.0
                    self._position_rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
                    self._position_offsets = {
                        name: Vector3(0.0, 0.0, 0.0) for name in POSE_NAMES
                    }
                    self._quaternion_offsets = {
                        name: Quaternion(0.0, 0.0, 0.0, 1.0) for name in POSE_NAMES
                    }
                else:
                    self._robot_scale = calibration.scale
                    self._position_rotation = calibration.position_rotation
                    self._position_offsets = dict(calibration.position_offsets)
                    self._quaternion_offsets = dict(calibration.quaternion_offsets)
                self._calibrated = True
                self._last_error = ""
                self.get_logger().info(
                    f"Quest calibration accepted: mode={self.calibration_mode}, "
                    f"position_scale={self._robot_scale:.3f}"
                )
                self._send_calibration_event(
                    "calibrated",
                    robot_scale=self._robot_scale,
                    recalibrated=was_calibrated,
                )
                self._request_retarget_warm_start()

        if decalibrate_edge and self._calibrated:
            self._reset_calibration()
            self.get_logger().info("Quest calibration reset")
            self._send_calibration_event("reset", reason="button_b")
            self._publish_status(
                "decalibrate",
                snapshot=snapshot,
                tracking_fresh=True,
            )

        self._calibrate_button_pressed = calibrate_pressed
        self._decalibrate_button_pressed = decalibrate_pressed

        self._last_published_receive_time = received_at
        self._last_browser_timestamp_ms = sample.timestamp_ms
        self._published_frames += 1
        self._publish_status(
            "frame",
            snapshot=snapshot,
            tracking_fresh=self._tracking_quality_ready,
        )
        stamp = self.get_clock().now().to_msg()
        joy.header.stamp = stamp
        self._joy_publisher.publish(joy)
        transforms: list[TransformStamped] = []
        for name in POSE_NAMES:
            raw_pose = vr_pose_to_ros(sample.poses[name])
            raw_transform = TransformStamped()
            raw_transform.header.stamp = stamp
            raw_transform.header.frame_id = "world"
            raw_transform.child_frame_id = f"QuestRaw{name}"
            raw_transform.transform.translation.x = raw_pose.position.x
            raw_transform.transform.translation.y = raw_pose.position.y
            raw_transform.transform.translation.z = raw_pose.position.z
            raw_transform.transform.rotation.x = raw_pose.quaternion.x
            raw_transform.transform.rotation.y = raw_pose.quaternion.y
            raw_transform.transform.rotation.z = raw_pose.quaternion.z
            raw_transform.transform.rotation.w = raw_pose.quaternion.w
            transforms.append(raw_transform)

        if not self._tracking_quality_ready:
            self._tf_broadcaster.sendTransform(transforms)
            return

        for name in POSE_NAMES:
            source_pose = (
                sample.poses[name]
                if name == "Head"
                else self._execution_poses.get(name)
            )
            if source_pose is None:
                continue
            execution_pose = vr_pose_to_ros(source_pose)
            execution_transform = TransformStamped()
            execution_transform.header.stamp = stamp
            execution_transform.header.frame_id = "world"
            execution_transform.child_frame_id = f"QuestExecution{name}"
            execution_transform.transform.translation.x = execution_pose.position.x
            execution_transform.transform.translation.y = execution_pose.position.y
            execution_transform.transform.translation.z = execution_pose.position.z
            execution_transform.transform.rotation.x = execution_pose.quaternion.x
            execution_transform.transform.rotation.y = execution_pose.quaternion.y
            execution_transform.transform.rotation.z = execution_pose.quaternion.z
            execution_transform.transform.rotation.w = execution_pose.quaternion.w
            transforms.append(execution_transform)

            pose = vr_pose_to_ros(
                source_pose,
                scale=self._robot_scale,
                position_rotation=self._position_rotation,
                position_offset=self._position_offsets[name],
                quaternion_offset=self._quaternion_offsets[name],
            )
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "world"
            transform.child_frame_id = (
                name if self._calibrated else f"{name}_uncalibrated"
            )
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation.x = pose.quaternion.x
            transform.transform.rotation.y = pose.quaternion.y
            transform.transform.rotation.z = pose.quaternion.z
            transform.transform.rotation.w = pose.quaternion.w
            transforms.append(transform)
        self._tf_broadcaster.sendTransform(transforms)

    def _request_calibration(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        snapshot = self._snapshot()
        if (
            snapshot.sample is None
            or not self._tracking_fresh(snapshot, time.monotonic())
            or not self._execution_poses
        ):
            response.success = False
            response.message = (
                "Quest tracking is not connected, fresh, or has no Normal hand pose"
            )
            return response
        self._calibration_requested = True
        response.success = True
        response.message = "Quest calibration requested for the next tracking frame"
        return response

    def _reset_calibration(self) -> None:
        self._calibrated = False
        self._robot_scale = 1.0
        self._position_rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self._position_offsets = {name: Vector3(0.0, 0.0, 0.0) for name in POSE_NAMES}
        self._quaternion_offsets = {
            name: Quaternion(0.0, 0.0, 0.0, 1.0) for name in POSE_NAMES
        }

    def _reset_tracking_quality(self) -> None:
        self._tracking_quality_ready = False
        self._hand_position_states = {
            "LeftHand": "Unavailable",
            "RightHand": "Unavailable",
        }
        self._hand_execution_states = {
            "LeftHand": "Normal",
            "RightHand": "Normal",
        }
        self._hand_execution_updated_at = {
            "LeftHand": None,
            "RightHand": None,
        }
        self._hand_gates = {
            "LeftHand": HandExecutionGate("LeftHand"),
            "RightHand": HandExecutionGate("RightHand"),
        }
        self._execution_poses = {}

    def _observe_hand_tracking(
        self,
        sample: WebVRSample,
        calibration_command: bool,
        received_at: float,
    ) -> None:
        self._hand_position_states = hand_position_states(sample)
        self._execution_poses = {}
        for name in ("LeftHand", "RightHand"):
            current = sample.poses[name]
            gate = self._hand_gates[name]
            execution_pose = gate.observe(
                current,
                calibration_command=calibration_command,
                jump_threshold=self.position_jump_threshold,
                recovery_frame_count=self.tracking_recovery_frames,
                recovery_motion_threshold=self.tracking_recovery_motion_threshold,
            )
            self._hand_execution_states[name] = gate.state
            if execution_pose is not None:
                self._execution_poses[name] = execution_pose
            if gate.ready:
                self._hand_execution_updated_at[name] = received_at

        self._tracking_quality_ready = hand_execution_is_ready(
            self._hand_gates,
            self._execution_poses,
        )
        errors = [gate.error for gate in self._hand_gates.values() if gate.error]
        self._last_error = "; ".join(errors)

    def _calibration_sample(self, sample: WebVRSample) -> WebVRSample:
        poses = dict(sample.poses)
        for name in ("LeftHand", "RightHand"):
            pose = self._execution_poses.get(name)
            if pose is not None:
                poses[name] = pose
        return WebVRSample(
            timestamp_ms=sample.timestamp_ms,
            poses=poses,
            joy_axes=sample.joy_axes,
            joy_buttons=sample.joy_buttons,
            calibration_pressed=sample.calibration_pressed,
            source_sequence=sample.source_sequence,
            source_monotonic_ms=sample.source_monotonic_ms,
        )

    def _request_retarget_warm_start(self) -> None:
        client = self._retarget_warm_start_client
        if client is None:
            return
        if not client.service_is_ready():
            self.get_logger().warning(
                "Retarget warm-start service is not ready; calibration remains active"
            )
            return
        client.call_async(Trigger.Request())

    def _reset_button_edges(self) -> None:
        self._calibrate_button_pressed = False
        self._decalibrate_button_pressed = False

    def _send_calibration_event(
        self,
        state: str,
        **details: object,
    ) -> None:
        self._socket_server.send_event(
            {
                "type": "calibration",
                "state": state,
                "calibrated": self._calibrated,
                **details,
            }
        )

    def _publish_decalibration_joy(self) -> None:
        joy = Joy()
        joy.header.stamp = self.get_clock().now().to_msg()
        joy.axes = [0.0] * 8
        joy.buttons = [0] * 12
        joy.buttons[DECALIBRATE_BUTTON] = 1
        self._joy_publisher.publish(joy)

    def _tracking_fresh(self, snapshot: ReceiverSnapshot, now: float) -> bool:
        return (
            snapshot.connected
            and self._tracking_quality_ready
            and self._last_published_receive_time is not None
            and now - self._last_published_receive_time <= self.input_timeout
        )

    def _publish_periodic_status(self) -> None:
        snapshot = self._snapshot()
        now = time.monotonic()
        if calibration_is_stale(
            calibrated=self._calibrated,
            connected=snapshot.connected,
            last_frame_time=self._last_published_receive_time,
            now=now,
            timeout=self.calibration_stale_timeout,
        ):
            self._reset_calibration()
            self._reset_button_edges()
            self._last_error = "Quest tracking stopped; calibration reset"
            self._publish_decalibration_joy()
            self._send_calibration_event("reset", reason="tracking_stale")
            self.get_logger().warning(
                "Quest tracking stopped; calibration reset, wake both "
                "controllers and press A again"
            )
            self._publish_status(
                "tracking_stale",
                snapshot=snapshot,
                tracking_fresh=False,
            )
            return
        self._publish_status("status", snapshot=snapshot)

    def _publish_status(
        self,
        event: str,
        *,
        snapshot: ReceiverSnapshot | None = None,
        tracking_fresh: bool | None = None,
    ) -> None:
        state = snapshot or self._snapshot()
        now = time.monotonic()
        fresh = (
            self._tracking_fresh(state, now)
            if tracking_fresh is None
            else tracking_fresh
        )
        receive_age = (
            None
            if self._last_published_receive_time is None
            else max(0.0, now - self._last_published_receive_time)
        )
        execution_ages = {
            name: (
                None
                if updated_at is None
                else max(0.0, now - updated_at) * 1000.0
            )
            for name, updated_at in self._hand_execution_updated_at.items()
        }
        hand_gates = {
            name: {
                "state": gate.state,
                "ready": gate.ready,
                "error": gate.error,
                "jump_distance_m": gate.jump_distance,
                "recovery_frames": gate.recovery_frames,
                "execution_age_ms": execution_ages[name],
            }
            for name, gate in self._hand_gates.items()
        }
        sample = state.sample
        raw_poses = (
            {} if sample is None else {
                name: pose_status(sample.poses[name]) for name in POSE_NAMES
            }
        )
        execution_poses = {
            name: pose_status(pose) for name, pose in self._execution_poses.items()
        }
        message = String()
        message.data = json.dumps(
            {
                "protocol_version": 1,
                "event": event,
                "connected": state.connected,
                "calibrated": self._calibrated,
                "tracking_fresh": fresh,
                "hand_position_tracking": self._hand_position_states,
                "hand_execution": self._hand_execution_states,
                "hand_gates": hand_gates,
                "raw_poses": raw_poses,
                "execution_poses": execution_poses,
                "calibration_ready": self._tracking_quality_ready,
                "received_sequence": state.sequence,
                "sequence": state.sequence,
                "source_sequence": (
                    None if sample is None else sample.source_sequence
                ),
                "source_sequence_gaps": state.source_sequence_gaps,
                "source_sequence_resets": state.source_sequence_resets,
                "receive_age_ms": None if receive_age is None else receive_age * 1000.0,
                "browser_timestamp_ms": self._last_browser_timestamp_ms,
                "source_monotonic_ms": (
                    None if sample is None else sample.source_monotonic_ms
                ),
                "published_frames": self._published_frames,
                "invalid_frames": state.invalid_frames,
                "robot_scale": self._robot_scale,
                "last_error": self._last_error or state.last_error,
                "web_url": self.public_web_url,
            },
            separators=(",", ":"),
        )
        self._status_publisher.publish(message)

    def _stop_servers(self) -> None:
        if self._servers_stopped:
            return
        self._servers_stopped = True
        if hasattr(self, "_socket_server"):
            self._socket_server.stop()
        if hasattr(self, "_http_server"):
            self._http_server.stop()
        self._stop_turn_relay()

    def destroy_node(self) -> bool:
        self._stop_servers()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = QuestWebVRNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
