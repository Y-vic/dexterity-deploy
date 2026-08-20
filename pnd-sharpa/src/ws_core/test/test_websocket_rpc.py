import base64
import hashlib
import shlex
import subprocess

import pytest
from unittest.mock import MagicMock, patch

from ws_core.msgpack_numpy import packb
from ws_core.websocket_rpc import (
    PolicyHttpError,
    SharpaV3PolicyClient,
    SimpleWebSocket,
    SshCommandStream,
    _request_msgpack_http,
)


class FakeHttpStream:
    def __init__(self, response: bytes):
        self.response = bytearray(response)
        self.sent = bytearray()
        self.timeouts = []
        self.closed = False

    def settimeout(self, timeout_s):
        self.timeouts.append(timeout_s)

    def sendall(self, payload):
        self.sent.extend(payload)

    def recv(self, nbytes):
        if not self.response:
            return b""
        payload = bytes(self.response[:nbytes])
        del self.response[:nbytes]
        return payload

    def close(self):
        self.closed = True


def test_ssh_command_stream_uses_clean_system_environment(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/conda/lib:/cuda/lib")
    monkeypatch.setenv("LD_PRELOAD", "/conda/lib/libcrypto.so.3")
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()

    with patch("ws_core.websocket_rpc.subprocess.Popen", return_value=process) as popen:
        SshCommandStream("BAAI2", "127.0.0.1", 5501, 5.0)

    command = popen.call_args.args[0]
    environment = popen.call_args.kwargs["env"]
    assert command[0] == "/usr/bin/ssh"
    assert command[-2] == "BAAI2"
    relay_command = shlex.split(command[-1])
    assert relay_command[:2] == ["bash", "-c"]
    relay_script = relay_command[2]
    assert "exec 3<>/dev/tcp/127.0.0.1/5501" in relay_script
    assert "exec 4<&0" in relay_script
    assert "trap cleanup EXIT HUP INT TERM" in relay_script
    assert "cat <&3 & reader_pid=$!" in relay_script
    assert "cat <&4 >&3 & writer_pid=$!" in relay_script
    assert 'wait -n "$reader_pid" "$writer_pid"' in relay_script
    assert 'wait -n "$reader_pid" "$writer_pid" "$watcher_pid"' not in relay_script
    assert "cat <&3 & cat >&3; wait" not in relay_script
    assert "ServerAliveInterval=5" in command
    assert "ServerAliveCountMax=1" in command
    assert "LD_LIBRARY_PATH" not in environment
    assert "LD_PRELOAD" not in environment
    assert environment["PATH"] == (
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )


def test_ssh_command_stream_health_watchdog_is_scoped_to_path():
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()

    with patch("ws_core.websocket_rpc.subprocess.Popen", return_value=process) as popen:
        SshCommandStream(
            "BAAI2",
            "127.0.0.1",
            5500,
            5.0,
            health_check_path="/healthz",
        )

    relay_command = shlex.split(popen.call_args.args[0][-1])
    relay_script = relay_command[2]
    assert "watch_server()" in relay_script
    assert "watch_server() { trap - EXIT HUP INT TERM;" in relay_script
    assert "exec 3>&- 4<&-; while :; do" in relay_script
    assert "GET /healthz HTTP/1.1" in relay_script
    assert "exec 5<>/dev/tcp/127.0.0.1/5500" in relay_script
    assert "exec 5>&- 5<&-; sleep 1; continue;" in relay_script
    assert 'wait -n "$reader_pid" "$writer_pid" "$watcher_pid"' in relay_script


@pytest.mark.parametrize("path", ["healthz", "/health z", "/health;rm -rf"])
def test_ssh_command_stream_rejects_invalid_health_path(path):
    with pytest.raises(ValueError, match="invalid health check path"):
        SshCommandStream("BAAI2", "127.0.0.1", 5500, 5.0, health_check_path=path)


def test_ssh_command_stream_close_allows_graceful_relay_exit():
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.poll.return_value = None
    stream = object.__new__(SshCommandStream)
    stream.process = process

    stream.close()
    stream.close()

    process.stdin.close.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2.0)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    process.stdout.close.assert_called_once_with()
    process.stderr.close.assert_called_once_with()


def test_ssh_command_stream_close_kills_stuck_relay():
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

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()


def test_simple_websocket_passes_health_watchdog_to_ssh_stream():
    websocket_key = base64.b64encode(b"x" * 16).decode("ascii")
    accept = base64.b64encode(
        hashlib.sha1(
            (
                websocket_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            ).encode("ascii")
        ).digest()
    ).decode("ascii")
    stream = FakeHttpStream(
        b"HTTP/1.1 101 Switching Protocols\r\n"
        + f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode()
    )

    with patch(
        "ws_core.websocket_rpc.SshCommandStream", return_value=stream
    ) as ssh, patch(
        "ws_core.websocket_rpc.os.urandom", side_effect=lambda size: b"x" * size
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


def test_v3_client_fetches_metadata_before_ws_without_waiting_for_app_message():
    metadata = {"schema": "sharpa_policy_server.v3", "metadata_format": {}}
    websocket = MagicMock()
    with patch(
        "ws_core.websocket_rpc._request_msgpack_http", return_value=metadata
    ) as request, patch(
        "ws_core.websocket_rpc.SimpleWebSocket", return_value=websocket
    ) as websocket_type:
        client = SharpaV3PolicyClient(
            "http://policy.example:5500/ignored?discarded=true",
            timeout_s=3.0,
            ssh_host="BAAI2",
            ssh_remote_host="127.0.0.1",
            ssh_remote_port=5500,
        )

    request.assert_called_once_with(
        "http://policy.example:5500/metadata",
        "GET",
        timeout_s=3.0,
        ssh_host="BAAI2",
        ssh_remote_host="127.0.0.1",
        ssh_remote_port=5500,
        max_body_size=64 * 1024 * 1024,
    )
    websocket_type.assert_called_once_with(
        "ws://policy.example:5500/infer",
        timeout_s=3.0,
        ssh_host="BAAI2",
        ssh_remote_host="127.0.0.1",
        ssh_remote_port=5500,
        health_check_path="/healthz",
        max_message_size=64 * 1024 * 1024,
    )
    websocket.recv.assert_not_called()
    assert client.metadata == metadata


def test_v3_infer_and_reset_use_binary_msgpack():
    metadata = {"schema": "sharpa_policy_server.v3"}
    reset_response = {
        "schema": "sharpa_policy_reset.v1",
        "session_id": "episode-2",
        "request_id": 7,
        "reset": True,
    }
    websocket = MagicMock()
    websocket.recv.return_value = packb({"schema": "sharpa_policy_action.v4"})
    with patch(
        "ws_core.websocket_rpc._request_msgpack_http",
        side_effect=[metadata, reset_response],
    ) as request, patch(
        "ws_core.websocket_rpc.SimpleWebSocket", return_value=websocket
    ):
        client = SharpaV3PolicyClient("ws://127.0.0.1:5500")
        result = client.infer({"schema": "sharpa_policy_observation.v3"})
        reset = client.reset("episode-2", request_id=7)

    websocket.send_binary.assert_called_once_with(
        packb({"schema": "sharpa_policy_observation.v3"})
    )
    assert result.payload == {"schema": "sharpa_policy_action.v4"}
    assert reset == reset_response
    assert request.call_args_list[1].args[:3] == (
        "http://127.0.0.1:5500/reset",
        "POST",
        {"session_id": "episode-2", "request_id": 7},
    )


def test_msgpack_http_request_uses_direct_socket_and_content_length():
    payload = {"schema": "sharpa_policy_server.v3", "value": 1}
    encoded = packb(payload)
    stream = FakeHttpStream(
        b"HTTP/1.1 200 OK\r\n"
        + f"Content-Length: {len(encoded)}\r\nContent-Type: application/msgpack\r\n\r\n".encode()
        + encoded
    )
    with patch(
        "ws_core.websocket_rpc.socket.create_connection", return_value=stream
    ) as create_connection:
        result = _request_msgpack_http(
            "http://policy.example:5500/metadata",
            "GET",
            timeout_s=2.0,
            ssh_host="",
            ssh_remote_host="",
            ssh_remote_port=0,
        )

    create_connection.assert_called_once_with(("policy.example", 5500), timeout=2.0)
    assert result == payload
    assert b"GET /metadata HTTP/1.1\r\n" in stream.sent
    assert b"Connection: close\r\n" in stream.sent
    assert stream.closed


def test_msgpack_http_request_uses_ssh_stream_for_reset_post():
    response = {"schema": "sharpa_policy_reset.v1", "reset": True}
    encoded = packb(response)
    stream = FakeHttpStream(
        b"HTTP/1.1 200 OK\r\n"
        + f"Content-Length: {len(encoded)}\r\n\r\n".encode()
        + encoded
    )
    reset_payload = {"session_id": "episode-2", "request_id": None}
    with patch("ws_core.websocket_rpc.SshCommandStream", return_value=stream) as ssh:
        result = _request_msgpack_http(
            "http://127.0.0.1:5500/reset",
            "POST",
            reset_payload,
            timeout_s=2.0,
            ssh_host="BAAI2",
            ssh_remote_host="127.0.0.1",
            ssh_remote_port=5500,
        )

    ssh.assert_called_once_with("BAAI2", "127.0.0.1", 5500, 2.0)
    assert result == response
    assert b"POST /reset HTTP/1.1\r\n" in stream.sent
    assert b"Content-Type: application/msgpack\r\n" in stream.sent
    assert packb(reset_payload) == bytes(stream.sent).split(b"\r\n\r\n", 1)[1]
    assert stream.closed


def test_msgpack_http_request_exposes_msgpack_error_payload():
    payload = {
        "schema": "sharpa_policy_error.v1",
        "error": {"code": "INVALID_RESET"},
    }
    encoded = packb(payload)
    stream = FakeHttpStream(
        b"HTTP/1.1 400 Bad Request\r\n"
        + f"Content-Length: {len(encoded)}\r\n\r\n".encode()
        + encoded
    )
    with patch("ws_core.websocket_rpc.socket.create_connection", return_value=stream):
        with pytest.raises(PolicyHttpError) as error:
            _request_msgpack_http(
                "http://127.0.0.1:5500/reset",
                "POST",
                {"session_id": ""},
                timeout_s=2.0,
                ssh_host="",
                ssh_remote_host="",
                ssh_remote_port=0,
            )

    assert error.value.status_code == 400
    assert error.value.payload == payload
