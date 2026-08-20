import base64
import hashlib
import shlex
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from sharpa_interface.server.transport import (
    SharpaV3PolicyClient,
    SimpleWebSocket,
    SshCommandStream,
)


class FakeHttpStream:
    def __init__(self, response: bytes):
        self.response = bytearray(response)
        self.sent = bytearray()
        self.closed = False

    def settimeout(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s

    def sendall(self, payload: bytes) -> None:
        self.sent.extend(payload)

    def recv(self, nbytes: int) -> bytes:
        payload = bytes(self.response[:nbytes])
        del self.response[:nbytes]
        return payload

    def close(self) -> None:
        self.closed = True


def test_ssh_relay_health_watchdog_is_scoped_to_healthz() -> None:
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()

    with patch(
        "sharpa_interface.server.transport.subprocess.Popen",
        return_value=process,
    ) as popen:
        SshCommandStream(
            "BAAI2",
            "127.0.0.1",
            5500,
            5.0,
            health_check_path="/healthz",
        )

    command = popen.call_args.args[0]
    relay_script = shlex.split(command[-1])[2]
    assert "GET /healthz HTTP/1.1" in relay_script
    assert 'wait -n "$reader_pid" "$writer_pid" "$watcher_pid"' in relay_script
    assert "ServerAliveInterval=5" in command
    assert "ServerAliveCountMax=1" in command


@pytest.mark.parametrize("path", ["healthz", "/health z", "/health;rm"])
def test_ssh_relay_rejects_unsafe_health_paths(path: str) -> None:
    with pytest.raises(ValueError, match="invalid health check path"):
        SshCommandStream(
            "BAAI2",
            "127.0.0.1",
            5500,
            5.0,
            health_check_path=path,
        )


def test_ssh_relay_close_is_idempotent_and_kills_a_stuck_process() -> None:
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired("ssh", 2.0),
        subprocess.TimeoutExpired("ssh", 2.0),
        0,
    ]
    stream = object.__new__(SshCommandStream)
    stream.process = process

    stream.close()
    stream.close()

    process.stdin.close.assert_called_once_with()
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()


def test_websocket_passes_health_watchdog_to_ssh_stream() -> None:
    websocket_key = base64.b64encode(b"x" * 16).decode("ascii")
    accept = base64.b64encode(
        hashlib.sha1(
            (
                websocket_key
                + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            ).encode("ascii")
        ).digest()
    ).decode("ascii")
    stream = FakeHttpStream(
        b"HTTP/1.1 101 Switching Protocols\r\n"
        + f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode()
    )

    with patch(
        "sharpa_interface.server.transport.SshCommandStream",
        return_value=stream,
    ) as ssh, patch(
        "sharpa_interface.server.transport.os.urandom",
        side_effect=lambda size: b"x" * size,
    ):
        websocket = SimpleWebSocket(
            "ws://127.0.0.1:5500/infer",
            timeout_s=5.0,
            ssh_host="BAAI2",
            ssh_remote_host="127.0.0.1",
            ssh_remote_port=5500,
            health_check_path="/healthz",
        )

    ssh.assert_called_once_with(
        "BAAI2",
        "127.0.0.1",
        5500,
        5.0,
        health_check_path="/healthz",
    )
    websocket.close()


def test_v3_client_fetches_metadata_before_opening_websocket() -> None:
    metadata = {"schema": "sharpa_policy_server.v3"}
    websocket = MagicMock()
    with patch(
        "sharpa_interface.server.transport._request_msgpack_http",
        return_value=metadata,
    ), patch(
        "sharpa_interface.server.transport.SimpleWebSocket",
        return_value=websocket,
    ) as websocket_type:
        client = SharpaV3PolicyClient(
            "http://policy.example:5500/ignored",
            ssh_host="BAAI2",
            ssh_remote_host="127.0.0.1",
            ssh_remote_port=5500,
        )

    websocket_type.assert_called_once_with(
        "ws://policy.example:5500/infer",
        timeout_s=60.0,
        ssh_host="BAAI2",
        ssh_remote_host="127.0.0.1",
        ssh_remote_port=5500,
        health_check_path="/healthz",
        max_message_size=64 * 1024 * 1024,
    )
    websocket.recv.assert_not_called()
    assert client.metadata == metadata
