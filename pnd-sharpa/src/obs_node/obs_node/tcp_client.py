from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from deploy_common.protocol import configure_tcp, send_frame


@dataclass
class SenderStatus:
    connected: bool
    connect_attempts: int
    sent_frames: int
    dropped_frames: int
    last_connect_time: float | None
    last_send_time: float | None
    last_error: str


class TcpFrameSender:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = 0.2,
        socket_timeout_s: float = 0.2,
        reconnect_initial_s: float = 0.2,
        reconnect_max_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.socket_timeout_s = float(socket_timeout_s)
        self.reconnect_initial_s = float(reconnect_initial_s)
        self.reconnect_max_s = float(reconnect_max_s)
        self.sock: socket.socket | None = None
        self.next_connect_at = 0.0
        self.reconnect_delay_s = self.reconnect_initial_s
        self.connect_attempts = 0
        self.sent_frames = 0
        self.dropped_frames = 0
        self.last_connect_time: float | None = None
        self.last_send_time: float | None = None
        self.last_error = ""

    @property
    def connected(self) -> bool:
        return self.sock is not None

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def status(self) -> SenderStatus:
        return SenderStatus(
            connected=self.connected,
            connect_attempts=self.connect_attempts,
            sent_frames=self.sent_frames,
            dropped_frames=self.dropped_frames,
            last_connect_time=self.last_connect_time,
            last_send_time=self.last_send_time,
            last_error=self.last_error,
        )

    def send(
        self,
        frame_type: int,
        payload: bytes,
        seq: int,
        stamp_ns: int | None = None,
        flags: int = 0,
    ) -> bool:
        if self.sock is None and not self._connect_if_due():
            self.dropped_frames += 1
            return False
        assert self.sock is not None
        try:
            send_frame(self.sock, frame_type, payload, seq, stamp_ns, flags)
        except OSError as exc:
            self.last_error = str(exc)
            self.close()
            self._schedule_reconnect()
            self.dropped_frames += 1
            return False
        self.sent_frames += 1
        self.last_send_time = time.monotonic()
        self.last_error = ""
        return True

    def _connect_if_due(self) -> bool:
        now = time.monotonic()
        if now < self.next_connect_at:
            return False
        self.connect_attempts += 1
        try:
            sock = socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout_s,
            )
            configure_tcp(sock, timeout_s=self.socket_timeout_s)
        except OSError as exc:
            self.last_error = str(exc)
            self._schedule_reconnect()
            return False
        self.sock = sock
        self.reconnect_delay_s = self.reconnect_initial_s
        self.last_connect_time = time.monotonic()
        self.last_error = ""
        return True

    def _schedule_reconnect(self) -> None:
        self.next_connect_at = time.monotonic() + self.reconnect_delay_s
        self.reconnect_delay_s = min(
            self.reconnect_max_s,
            max(self.reconnect_initial_s, self.reconnect_delay_s * 1.7),
        )
