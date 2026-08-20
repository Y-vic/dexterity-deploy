from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from ws_msgs.msg import Status

from deploy_common.protocol import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    configure_tcp,
    recv_frame,
)


FrameHandler = Callable[[Any, int], None]


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def age_ms(stamp: float | None, now: float | None = None) -> float | None:
    if stamp is None:
        return None
    if now is None:
        now = time.monotonic()
    return round((now - stamp) * 1000.0, 1)


def make_status_msg(node_name: str, ok: bool, payload: dict[str, Any], stamp: Any) -> Status:
    msg = Status()
    msg.header.stamp = stamp
    msg.node = node_name
    msg.ok = bool(ok)
    msg.payload_json = compact_json(payload)
    return msg


class TcpFrameServer:
    def __init__(
        self,
        *,
        name: str,
        host: str,
        port: int,
        expected_frame_type: int,
        handler: FrameHandler,
        logger: Any,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self.name = name
        self.host = host
        self.port = int(port)
        self.expected_frame_type = int(expected_frame_type)
        self.handler = handler
        self.logger = logger
        self.max_payload_bytes = int(max_payload_bytes)

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._server_sock: socket.socket | None = None
        self._client_socks: set[socket.socket] = set()
        self._client_threads: list[threading.Thread] = []

        self._listening = False
        self._accepted_connections = 0
        self._closed_connections = 0
        self._received_frames = 0
        self._published_frames = 0
        self._ignored_frames = 0
        self._error_frames = 0
        self._last_accept_time: float | None = None
        self._last_receive_time: float | None = None
        self._last_publish_time: float | None = None
        self._last_error = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._serve,
            name=f"{self.name}-tcp-server",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._lock:
            server_sock = self._server_sock
            client_socks = list(self._client_socks)
        self._close_socket(server_sock)
        for client_sock in client_socks:
            self._close_socket(client_sock)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        for thread in list(self._client_threads):
            if thread is not threading.current_thread():
                thread.join(timeout=0.5)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            return {
                "role": "tcp_server",
                "host": self.host,
                "port": self.port,
                "expected_frame_type": self.expected_frame_type,
                "listening": self._listening,
                "connected_clients": len(self._client_socks),
                "accepted_connections": self._accepted_connections,
                "closed_connections": self._closed_connections,
                "received_frames": self._received_frames,
                "published_frames": self._published_frames,
                "ignored_frames": self._ignored_frames,
                "error_frames": self._error_frames,
                "last_accept_age_ms": age_ms(self._last_accept_time, now),
                "last_receive_age_ms": age_ms(self._last_receive_time, now),
                "last_publish_age_ms": age_ms(self._last_publish_time, now),
                "last_error": self._last_error,
            }

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._listening

    def _serve(self) -> None:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen()
            sock.settimeout(0.2)
            with self._lock:
                self._server_sock = sock
                self._listening = True
                self._last_error = ""
            self.logger.info(f"{self.name} listening on tcp://{self.host}:{self.port}")

            while not self._stop_event.is_set():
                try:
                    client_sock, addr = sock.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self._stop_event.is_set():
                        self._record_error(f"accept failed: {exc}", frame_error=False)
                    break
                try:
                    configure_tcp(client_sock)
                except OSError as exc:
                    self._record_error(
                        f"{self._addr_text(addr)}: configure failed: {exc}",
                        frame_error=False,
                    )
                    self._close_socket(client_sock)
                    continue
                with self._lock:
                    self._client_socks.add(client_sock)
                    self._accepted_connections += 1
                    self._last_accept_time = time.monotonic()
                thread = threading.Thread(
                    target=self._client_loop,
                    args=(client_sock, addr),
                    name=f"{self.name}-tcp-client",
                    daemon=True,
                )
                self._client_threads.append(thread)
                thread.start()
        except OSError as exc:
            self._record_error(f"listen failed: {exc}", frame_error=False)
            self.logger.error(f"{self.name} listen failed: {exc}")
        finally:
            with self._lock:
                self._listening = False
                self._server_sock = None
            self._close_socket(sock)

    def _client_loop(self, client_sock: socket.socket, addr: Any) -> None:
        addr_text = self._addr_text(addr)
        try:
            while not self._stop_event.is_set():
                try:
                    frame = recv_frame(client_sock, self.max_payload_bytes)
                except EOFError:
                    break
                except OSError as exc:
                    if not self._stop_event.is_set():
                        self._record_error(f"{addr_text}: receive failed: {exc}")
                    break
                except Exception as exc:  # noqa: BLE001 - bad frame closes this client.
                    self._record_error(f"{addr_text}: bad frame: {exc}")
                    break

                self._record_received()
                if int(frame.frame_type) != self.expected_frame_type:
                    self._record_ignored(
                        f"{addr_text}: unexpected frame type {frame.frame_type}, "
                        f"expected {self.expected_frame_type}"
                    )
                    continue

                recv_time_ns = time.time_ns()
                try:
                    self.handler(frame, recv_time_ns)
                except Exception as exc:  # noqa: BLE001 - keep server alive.
                    self._record_error(f"{addr_text}: frame handling failed: {exc}")
                    continue
                self._record_published()
        finally:
            with self._lock:
                self._client_socks.discard(client_sock)
                self._closed_connections += 1
            self._close_socket(client_sock)

    def _record_received(self) -> None:
        with self._lock:
            self._received_frames += 1
            self._last_receive_time = time.monotonic()

    def _record_published(self) -> None:
        with self._lock:
            self._published_frames += 1
            self._last_publish_time = time.monotonic()
            self._last_error = ""

    def _record_ignored(self, message: str) -> None:
        with self._lock:
            self._ignored_frames += 1
            self._last_error = message

    def _record_error(self, message: str, *, frame_error: bool = True) -> None:
        with self._lock:
            if frame_error:
                self._error_frames += 1
            self._last_error = message

    @staticmethod
    def _addr_text(addr: Any) -> str:
        if isinstance(addr, tuple) and len(addr) >= 2:
            return f"{addr[0]}:{addr[1]}"
        return str(addr)

    @staticmethod
    def _close_socket(sock: socket.socket | None) -> None:
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
