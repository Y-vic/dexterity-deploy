"""Small synchronous websocket msgpack RPC client for policy providers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import select
import shlex
import socket
import ssl
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ws_core.msgpack_numpy import packb, unpackb


GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
DEFAULT_MAX_HTTP_BODY_SIZE = 64 * 1024 * 1024
DEFAULT_MAX_WEBSOCKET_MESSAGE_SIZE = 64 * 1024 * 1024


def _validate_health_check_path(path: str | None) -> str | None:
    if path is None:
        return None
    value = str(path).strip()
    if not re.fullmatch(r"/[A-Za-z0-9._~/-]*", value):
        raise ValueError(f"invalid health check path: {path!r}")
    return value


def _build_ssh_relay_script(
    remote_host: str,
    remote_port: int,
    health_check_path: str | None,
) -> str:
    script = (
        f"exec 3<>/dev/tcp/{remote_host}/{int(remote_port)} || exit $?; "
        "exec 4<&0; "
        "reader_pid=; writer_pid=; watcher_pid=; "
        "cleanup() { "
        "trap - EXIT HUP INT TERM; "
        "for pid in \"$reader_pid\" \"$writer_pid\" \"$watcher_pid\"; do "
        "if [ -n \"$pid\" ]; then "
        "kill \"$pid\" 2>/dev/null || true; fi; "
        "done; "
        "for pid in \"$reader_pid\" \"$writer_pid\" \"$watcher_pid\"; do "
        "if [ -n \"$pid\" ]; then "
        "wait \"$pid\" 2>/dev/null || true; fi; "
        "done; "
        "exec 3>&- 4<&-; "
        "}; "
        "trap cleanup EXIT HUP INT TERM; "
        "cat <&3 & reader_pid=$!; "
        "cat <&4 >&3 & writer_pid=$!; "
    )
    if health_check_path is not None:
        script += (
            "watch_server() { "
            "trap - EXIT HUP INT TERM; "
            "exec 3>&- 4<&-; "
            "while :; do "
            f"exec 5<>/dev/tcp/{remote_host}/{int(remote_port)} || exit 0; "
            f"printf 'GET {health_check_path} HTTP/1.1\\r\\n"
            f"Host: {remote_host}\\r\\nConnection: close\\r\\n\\r\\n' >&5 "
            "|| { exec 5>&- 5<&-; exit 0; }; "
            "IFS= read -r -t 2 status <&5 "
            "|| { exec 5>&- 5<&-; sleep 1; continue; }; "
            "case \"$status\" in "
            "HTTP/1.[01]\\ 2*) ;; "
            "*) exec 5>&- 5<&-; exit 0 ;; "
            "esac; "
            "exec 5>&- 5<&-; "
            "sleep 1; "
            "done; "
            "}; "
            "watch_server & watcher_pid=$!; "
        )
    wait_command = (
        'wait -n "$reader_pid" "$writer_pid" "$watcher_pid"'
        if health_check_path is not None
        else 'wait -n "$reader_pid" "$writer_pid"'
    )
    return script + wait_command


class WebSocketProtocolError(RuntimeError):
    pass


class PolicyHttpError(RuntimeError):
    def __init__(
        self,
        method: str,
        url: str,
        status_code: int,
        reason: str,
        payload: Any,
    ) -> None:
        self.method = method
        self.url = url
        self.status_code = status_code
        self.reason = reason
        self.payload = payload
        super().__init__(
            f"policy HTTP {method} {url} failed with "
            f"{status_code} {reason}".rstrip()
        )


@dataclass
class PolicyRpcResult:
    payload: dict[str, Any]
    latency_s: float


class SshCommandStream:
    def __init__(
        self,
        ssh_host: str,
        remote_host: str,
        remote_port: int,
        timeout_s: float,
        *,
        health_check_path: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", remote_host):
            raise ValueError(f"invalid SSH remote host: {remote_host!r}")
        if not 1 <= int(remote_port) <= 65535:
            raise ValueError(f"invalid SSH remote port: {remote_port}")
        health_check_path = _validate_health_check_path(health_check_path)
        script = _build_ssh_relay_script(
            remote_host,
            int(remote_port),
            health_check_path,
        )
        remote_command = "bash -c " + shlex.quote(script)
        ssh_executable = "/usr/bin/ssh"
        if not os.path.isfile(ssh_executable):
            ssh_executable = "ssh"
        ssh_environment = os.environ.copy()
        # The workstation runs inside Conda, whose OpenSSL can be newer than the
        # system OpenSSH ABI.  Do not let dynamic-library overrides leak into
        # the system ssh child process.
        ssh_environment.pop("LD_LIBRARY_PATH", None)
        ssh_environment.pop("LD_PRELOAD", None)
        ssh_environment["PATH"] = (
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        self.timeout_s = float(timeout_s)
        self._close_lock = threading.Lock()
        self.process = subprocess.Popen(
            [
                ssh_executable,
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=5",
                "-o",
                "ServerAliveCountMax=1",
                ssh_host,
                remote_command,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=ssh_environment,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self.close()
            raise ConnectionError("failed to open SSH command stream")

    def settimeout(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)

    def sendall(self, payload: bytes) -> None:
        process = self.process
        if process is None:
            raise ConnectionError("SSH command stream is closed")
        stdin = process.stdin
        if stdin is None:
            raise ConnectionError("SSH command stream is closed")
        view = memoryview(payload)
        deadline = time.monotonic() + self.timeout_s
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("timed out writing SSH command stream")
            _, writable, _ = select.select([], [stdin.fileno()], [], remaining)
            if not writable:
                raise TimeoutError("timed out writing SSH command stream")
            try:
                written = os.write(stdin.fileno(), view)
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise ConnectionError(self._failure_detail(process)) from exc
            if written <= 0:
                raise ConnectionError(self._failure_detail(process))
            view = view[written:]

    def recv(self, nbytes: int) -> bytes:
        process = self.process
        if process is None:
            raise ConnectionError("SSH command stream is closed")
        stdout = process.stdout
        if stdout is None:
            raise ConnectionError("SSH command stream is closed")
        try:
            readable, _, _ = select.select(
                [stdout.fileno()],
                [],
                [],
                self.timeout_s,
            )
        except (OSError, ValueError) as exc:
            raise ConnectionError(self._failure_detail(process)) from exc
        if not readable:
            raise TimeoutError("timed out reading SSH command stream")
        try:
            data = os.read(stdout.fileno(), nbytes)
        except (OSError, ValueError) as exc:
            raise ConnectionError(self._failure_detail(process)) from exc
        if not data:
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            raise ConnectionError(self._failure_detail(process))
        return data

    def close(self) -> None:
        close_lock = getattr(self, "_close_lock", None)
        if close_lock is None:
            close_lock = threading.Lock()
            self._close_lock = close_lock
        with close_lock:
            process = getattr(self, "process", None)
            if process is None:
                return
            self.process = None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    @staticmethod
    def _failure_detail(process: subprocess.Popen[bytes]) -> str:
        return_code = process.poll()
        detail = ""
        if return_code is not None and process.stderr is not None:
            try:
                detail = process.stderr.read(4096).decode(
                    "utf-8", errors="replace"
                ).strip()
            except (OSError, ValueError):
                pass
        suffix = f": {detail}" if detail else ""
        return f"SSH command stream exited with code {return_code}{suffix}"


class SimpleWebSocket:
    def __init__(
        self,
        url: str,
        timeout_s: float = 30.0,
        *,
        ssh_host: str = "",
        ssh_remote_host: str = "",
        ssh_remote_port: int = 0,
        health_check_path: str | None = None,
        max_message_size: int | None = None,
    ) -> None:
        self.url = url
        self.timeout_s = float(timeout_s)
        self.ssh_host = str(ssh_host).strip()
        self.ssh_remote_host = str(ssh_remote_host).strip()
        self.ssh_remote_port = int(ssh_remote_port)
        self.health_check_path = health_check_path
        self.max_message_size = (
            None if max_message_size is None else int(max_message_size)
        )
        if self.max_message_size is not None and self.max_message_size < 1:
            raise ValueError("max_message_size must be positive or None")
        self._close_lock = threading.Lock()
        self.sock: socket.socket | ssl.SSLSocket | SshCommandStream | None = None
        self._recv_buffer = bytearray()
        self._connect()

    def close(self) -> None:
        with self._close_lock:
            sock = self.sock
            if sock is None:
                return
            self.sock = None
        if isinstance(sock, (socket.socket, ssl.SSLSocket)):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            sock.close()
        except Exception:
            pass

    def send_binary(self, payload: bytes) -> None:
        if (
            self.max_message_size is not None
            and len(payload) > self.max_message_size
        ):
            raise WebSocketProtocolError("websocket message is too large")
        self._send_frame(payload, opcode=0x2)

    def recv(self) -> bytes | str:
        fragments: list[bytes] = []
        opcode0: int | None = None
        while True:
            fin, opcode, payload = self._recv_frame()
            if opcode == 0x8:
                raise ConnectionError("websocket closed by peer")
            if opcode == 0x9:
                self._send_frame(payload, opcode=0xA)
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                opcode0 = opcode
                fragments = [payload]
            elif opcode == 0x0:
                if opcode0 is None:
                    raise WebSocketProtocolError("continuation without initial frame")
                fragments.append(payload)
            else:
                raise WebSocketProtocolError(f"unsupported websocket opcode {opcode}")
            if self.max_message_size is not None and sum(
                len(fragment) for fragment in fragments
            ) > self.max_message_size:
                raise WebSocketProtocolError("websocket message is too large")
            if fin:
                data = b"".join(fragments)
                if opcode0 == 0x1:
                    return data.decode("utf-8")
                return data

    def _connect(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"unsupported websocket scheme: {parsed.scheme!r}")
        host = parsed.hostname
        if not host:
            raise ValueError(f"websocket URL has no host: {self.url}")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        if self.ssh_host:
            if parsed.scheme == "wss":
                raise ValueError("wss over SSH command stream is not supported")
            raw_sock: socket.socket | SshCommandStream = SshCommandStream(
                self.ssh_host,
                self.ssh_remote_host or host,
                self.ssh_remote_port or port,
                self.timeout_s,
                health_check_path=self.health_check_path,
            )
        else:
            raw_sock = socket.create_connection((host, port), timeout=self.timeout_s)
        raw_sock.settimeout(self.timeout_s)
        if parsed.scheme == "wss":
            context = ssl.create_default_context()
            sock: socket.socket | ssl.SSLSocket = context.wrap_socket(
                raw_sock,
                server_hostname=host,
            )
        else:
            sock = raw_sock

        try:
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            host_header = host if parsed.port is None else f"{host}:{port}"
            request = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host_header}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                "\r\n"
            )
            sock.sendall(request.encode("ascii"))
            status_line, headers, leftover = self._read_http_response(sock)
            if " 101 " not in status_line:
                raise ConnectionError(f"websocket upgrade failed: {status_line}")
            accept_expected = base64.b64encode(
                hashlib.sha1((key + GUID).encode("ascii")).digest()
            ).decode("ascii")
            if headers.get("sec-websocket-accept", "") != accept_expected:
                raise WebSocketProtocolError("invalid Sec-WebSocket-Accept")
        except Exception:
            sock.close()
            raise
        self._recv_buffer.extend(leftover)
        self.sock = sock

    @staticmethod
    def _read_http_response(
        sock: socket.socket | ssl.SSLSocket | SshCommandStream,
    ) -> tuple[str, dict[str, str], bytes]:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("socket closed during websocket handshake")
            data.extend(chunk)
            if len(data) > 65536:
                raise WebSocketProtocolError("websocket handshake response too large")
        header_bytes, leftover = bytes(data).split(b"\r\n\r\n", 1)
        header_text = header_bytes.decode("iso-8859-1")
        lines = header_text.split("\r\n")
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        return lines[0], headers, leftover

    def _send_frame(self, payload: bytes, opcode: int) -> None:
        sock = self.sock
        if sock is None:
            raise ConnectionError("websocket is closed")
        mask = os.urandom(4)
        length = len(payload)
        first = 0x80 | opcode
        if length < 126:
            header = struct.pack("!BB", first, 0x80 | length)
        elif length < (1 << 16):
            header = struct.pack("!BBH", first, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", first, 0x80 | 127, length)
        masked = bytes(byte ^ mask[idx % 4] for idx, byte in enumerate(payload))
        sock.sendall(header + mask + masked)

    def _recv_exact(self, nbytes: int) -> bytes:
        sock = self.sock
        if sock is None:
            raise ConnectionError("websocket is closed")
        data = bytearray()
        if self._recv_buffer:
            take = min(nbytes, len(self._recv_buffer))
            data.extend(self._recv_buffer[:take])
            del self._recv_buffer[:take]
        while len(data) < nbytes:
            chunk = sock.recv(nbytes - len(data))
            if not chunk:
                raise ConnectionError("socket closed while reading websocket frame")
            data.extend(chunk)
        return bytes(data)

    def _recv_frame(self) -> tuple[bool, int, bytes]:
        first, second = self._recv_exact(2)
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        if self.max_message_size is not None and length > self.max_message_size:
            raise WebSocketProtocolError("websocket frame is too large")
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[idx % 4] for idx, byte in enumerate(payload))
        return fin, opcode, payload


def _policy_endpoint_urls(url: str) -> tuple[str, str, str]:
    parsed = urlparse(str(url).strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https", "ws", "wss"):
        raise ValueError(f"unsupported policy URL scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"policy URL has no host: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("policy URL must not contain user information")
    port = parsed.port
    authority_host = f"[{host}]" if ":" in host else host
    authority = authority_host if port is None else f"{authority_host}:{port}"
    http_scheme = "https" if scheme in ("https", "wss") else "http"
    websocket_scheme = "wss" if scheme in ("https", "wss") else "ws"
    return (
        f"{http_scheme}://{authority}/metadata",
        f"{http_scheme}://{authority}/reset",
        f"{websocket_scheme}://{authority}/infer",
    )


def _http_host_header(parsed: Any) -> str:
    host = parsed.hostname
    if not host:
        raise ValueError(f"HTTP URL has no host: {parsed.geturl()}")
    host = f"[{host}]" if ":" in host else host
    return host if parsed.port is None else f"{host}:{parsed.port}"


def _open_http_stream(
    url: str,
    timeout_s: float,
    *,
    ssh_host: str,
    ssh_remote_host: str,
    ssh_remote_port: int,
) -> tuple[Any, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported HTTP scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"HTTP URL has no host: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if ssh_host:
        if parsed.scheme == "https":
            raise ValueError("https over SSH command stream is not supported")
        stream: Any = SshCommandStream(
            ssh_host,
            ssh_remote_host or host,
            ssh_remote_port or port,
            timeout_s,
        )
    else:
        raw_stream = socket.create_connection((host, port), timeout=timeout_s)
        raw_stream.settimeout(timeout_s)
        if parsed.scheme == "https":
            context = ssl.create_default_context()
            stream = context.wrap_socket(raw_stream, server_hostname=host)
        else:
            stream = raw_stream
    stream.settimeout(timeout_s)
    return parsed, stream


def _read_chunked_http_body(
    stream: Any,
    initial: bytes,
    max_body_size: int,
) -> bytes:
    buffered = bytearray(initial)
    body = bytearray()

    def read_more() -> None:
        chunk = stream.recv(4096)
        if not chunk:
            raise ConnectionError("socket closed during chunked HTTP response")
        buffered.extend(chunk)

    def read_line() -> bytes:
        while b"\r\n" not in buffered:
            read_more()
        line, remaining = bytes(buffered).split(b"\r\n", 1)
        buffered.clear()
        buffered.extend(remaining)
        return line

    while True:
        size_line = read_line()
        try:
            chunk_size = int(size_line.split(b";", 1)[0], 16)
        except ValueError as exc:
            raise ConnectionError("invalid HTTP chunk size") from exc
        if chunk_size < 0:
            raise ConnectionError("invalid negative HTTP chunk size")
        if chunk_size == 0:
            while read_line():
                pass
            return bytes(body)
        if len(body) + chunk_size > max_body_size:
            raise ConnectionError("policy HTTP response body is too large")
        while len(buffered) < chunk_size + 2:
            read_more()
        body.extend(buffered[:chunk_size])
        if buffered[chunk_size : chunk_size + 2] != b"\r\n":
            raise ConnectionError("invalid HTTP chunk terminator")
        del buffered[: chunk_size + 2]


def _read_http_body(
    stream: Any,
    headers: dict[str, str],
    initial: bytes,
    max_body_size: int,
) -> bytes:
    transfer_encoding = headers.get("transfer-encoding", "").lower()
    if "chunked" in transfer_encoding:
        return _read_chunked_http_body(stream, initial, max_body_size)

    raw_content_length = headers.get("content-length")
    if raw_content_length is None:
        raise ConnectionError("policy HTTP response has no body length")
    try:
        content_length = int(raw_content_length)
    except ValueError as exc:
        raise ConnectionError("invalid HTTP Content-Length") from exc
    if content_length < 0:
        raise ConnectionError("invalid negative HTTP Content-Length")
    if content_length > max_body_size:
        raise ConnectionError("policy HTTP response body is too large")
    body = bytearray(initial[:content_length])
    while len(body) < content_length:
        chunk = stream.recv(content_length - len(body))
        if not chunk:
            raise ConnectionError("socket closed during HTTP response body")
        body.extend(chunk)
    return bytes(body)


def _request_msgpack_http(
    url: str,
    method: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float,
    ssh_host: str,
    ssh_remote_host: str,
    ssh_remote_port: int,
    max_body_size: int = DEFAULT_MAX_HTTP_BODY_SIZE,
) -> Any:
    method = method.upper()
    if method not in ("GET", "POST"):
        raise ValueError(f"unsupported policy HTTP method: {method!r}")
    parsed, stream = _open_http_stream(
        url,
        timeout_s,
        ssh_host=ssh_host,
        ssh_remote_host=ssh_remote_host,
        ssh_remote_port=ssh_remote_port,
    )
    request_body = b"" if payload is None else packb(payload)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    request_headers = [
        f"{method} {target} HTTP/1.1",
        f"Host: {_http_host_header(parsed)}",
        "Accept: application/msgpack",
        "Connection: close",
    ]
    if method == "POST":
        request_headers.extend(
            [
                "Content-Type: application/msgpack",
                f"Content-Length: {len(request_body)}",
            ]
        )
    request = ("\r\n".join(request_headers) + "\r\n\r\n").encode("ascii")
    try:
        stream.sendall(request + request_body)
        status_line, headers, initial = SimpleWebSocket._read_http_response(stream)
        status_parts = status_line.split(" ", 2)
        if len(status_parts) < 2 or not status_parts[0].startswith("HTTP/"):
            raise ConnectionError(f"invalid HTTP status line: {status_line!r}")
        try:
            status_code = int(status_parts[1])
        except ValueError as exc:
            raise ConnectionError(f"invalid HTTP status line: {status_line!r}") from exc
        reason = status_parts[2] if len(status_parts) == 3 else ""
        raw_response = _read_http_body(stream, headers, initial, max_body_size)
        try:
            response = unpackb(raw_response)
        except Exception as exc:
            raise ConnectionError("policy HTTP response is not valid msgpack") from exc
        if not 200 <= status_code < 300:
            raise PolicyHttpError(method, url, status_code, reason, response)
        return response
    finally:
        stream.close()


class MsgpackPolicyWsClient:
    def __init__(
        self,
        url: str,
        timeout_s: float = 60.0,
        *,
        expect_initial_message: bool = True,
        ssh_host: str = "",
        ssh_remote_host: str = "",
        ssh_remote_port: int = 0,
    ) -> None:
        self.url = url
        self.timeout_s = float(timeout_s)
        self.ws = SimpleWebSocket(
            url,
            timeout_s=timeout_s,
            ssh_host=ssh_host,
            ssh_remote_host=ssh_remote_host,
            ssh_remote_port=ssh_remote_port,
        )
        self.metadata: dict[str, Any] = {}
        try:
            if expect_initial_message:
                initial = self._recv_message()
                self.metadata = (
                    initial
                    if isinstance(initial, dict)
                    else {"initial_message": str(initial)}
                )
        except Exception:
            self.ws.close()
            raise

    def close(self) -> None:
        self.ws.close()

    def infer(self, payload: dict[str, Any]) -> PolicyRpcResult:
        start = time.monotonic()
        self.ws.send_binary(packb(payload))
        response = self._recv_payload()
        return PolicyRpcResult(payload=response, latency_s=time.monotonic() - start)

    def reset(self, session_id: str, schema: str) -> Any | None:
        payload = {
            "schema": schema,
            "endpoint": "reset",
            "session_id": session_id,
        }
        self.ws.send_binary(packb(payload))
        return self._recv_message()

    def _recv_payload(self) -> dict[str, Any]:
        payload = self._recv_message()
        if isinstance(payload, str):
            raise RuntimeError(f"policy server returned text: {payload}")
        if not isinstance(payload, dict):
            raise TypeError(f"policy response is {type(payload).__name__}, expected dict")
        return payload

    def _recv_message(self) -> Any:
        raw = self.ws.recv()
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return unpackb(raw)


class SharpaV3PolicyClient:
    """Client for the fixed SharpA policy server v3 transport.

    Unlike :class:`MsgpackPolicyWsClient`, the v3 WebSocket does not send an
    application-level metadata message after the upgrade.  Metadata is fetched
    explicitly over HTTP before the WebSocket is opened.
    """

    def __init__(
        self,
        url: str,
        timeout_s: float = 60.0,
        *,
        ssh_host: str = "",
        ssh_remote_host: str = "",
        ssh_remote_port: int = 0,
        max_http_body_size: int = DEFAULT_MAX_HTTP_BODY_SIZE,
        max_message_size: int = DEFAULT_MAX_WEBSOCKET_MESSAGE_SIZE,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.ssh_host = str(ssh_host).strip()
        self.ssh_remote_host = str(ssh_remote_host).strip()
        self.ssh_remote_port = int(ssh_remote_port)
        self.max_http_body_size = int(max_http_body_size)
        self.max_message_size = int(max_message_size)
        if self.max_http_body_size < 1:
            raise ValueError("max_http_body_size must be positive")
        if self.max_message_size < 1:
            raise ValueError("max_message_size must be positive")
        self.metadata_url, self.reset_url, self.infer_url = _policy_endpoint_urls(url)
        self.metadata: dict[str, Any]
        self.ws: SimpleWebSocket | None = None
        metadata = _request_msgpack_http(
            self.metadata_url,
            "GET",
            timeout_s=self.timeout_s,
            ssh_host=self.ssh_host,
            ssh_remote_host=self.ssh_remote_host,
            ssh_remote_port=self.ssh_remote_port,
            max_body_size=self.max_http_body_size,
        )
        if not isinstance(metadata, dict):
            raise TypeError(
                f"policy metadata is {type(metadata).__name__}, expected dict"
            )
        self.metadata = metadata
        try:
            self.ws = SimpleWebSocket(
                self.infer_url,
                timeout_s=self.timeout_s,
                ssh_host=self.ssh_host,
                ssh_remote_host=self.ssh_remote_host,
                ssh_remote_port=self.ssh_remote_port,
                health_check_path="/healthz" if self.ssh_host else None,
                max_message_size=self.max_message_size,
            )
        except Exception:
            self.ws = None
            raise

    def close(self) -> None:
        websocket = self.ws
        if websocket is None:
            return
        self.ws = None
        websocket.close()

    def infer(self, payload: dict[str, Any]) -> PolicyRpcResult:
        websocket = self.ws
        if websocket is None:
            raise ConnectionError("policy websocket is closed")
        start = time.monotonic()
        websocket.send_binary(packb(payload))
        response = self._recv_payload()
        return PolicyRpcResult(payload=response, latency_s=time.monotonic() - start)

    def reset(self, session_id: str, request_id: int | None = None) -> dict[str, Any]:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a nonempty string")
        if request_id is not None and (
            isinstance(request_id, bool) or not isinstance(request_id, int)
        ):
            raise ValueError("request_id must be an integer or None")
        response = _request_msgpack_http(
            self.reset_url,
            "POST",
            {"session_id": session_id, "request_id": request_id},
            timeout_s=self.timeout_s,
            ssh_host=self.ssh_host,
            ssh_remote_host=self.ssh_remote_host,
            ssh_remote_port=self.ssh_remote_port,
            max_body_size=self.max_http_body_size,
        )
        if not isinstance(response, dict):
            raise TypeError(
                f"policy reset response is {type(response).__name__}, expected dict"
            )
        return response

    def _recv_payload(self) -> dict[str, Any]:
        websocket = self.ws
        if websocket is None:
            raise ConnectionError("policy websocket is closed")
        raw = websocket.recv()
        if isinstance(raw, str):
            raise RuntimeError("policy v3 server returned a text websocket message")
        payload = unpackb(raw)
        if not isinstance(payload, dict):
            raise TypeError(
                f"policy response is {type(payload).__name__}, expected dict"
            )
        return payload


SharpaV3PolicyWsClient = SharpaV3PolicyClient
