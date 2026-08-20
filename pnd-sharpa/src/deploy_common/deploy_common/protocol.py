from __future__ import annotations

import json
import socket
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Any


MAGIC = b"PND1"
VERSION = 1
FRAME_TYPE_ACTION = 1
FRAME_TYPE_OBS_STATE = 2
FRAME_TYPE_TACTILE_BULK = 3
HEADER_STRUCT = struct.Struct("!4sBBHQqII")
HEADER_SIZE = HEADER_STRUCT.size
DEFAULT_MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class Frame:
    frame_type: int
    flags: int
    seq: int
    stamp_ns: int
    payload: bytes


def now_ns() -> int:
    return time.time_ns()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def json_from_bytes(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON frame payload must be an object")
    return value


def pack_frame(
    frame_type: int,
    payload: bytes,
    seq: int,
    stamp_ns: int | None = None,
    flags: int = 0,
) -> bytes:
    if stamp_ns is None:
        stamp_ns = now_ns()
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        int(frame_type) & 0xFF,
        int(flags) & 0xFFFF,
        int(seq) & 0xFFFFFFFFFFFFFFFF,
        int(stamp_ns),
        len(payload),
        checksum,
    )
    return header + payload


def send_frame(
    sock: socket.socket,
    frame_type: int,
    payload: bytes,
    seq: int,
    stamp_ns: int | None = None,
    flags: int = 0,
) -> None:
    sock.sendall(pack_frame(frame_type, payload, seq, stamp_ns, flags))


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(
    sock: socket.socket,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> Frame:
    header = recv_exact(sock, HEADER_SIZE)
    magic, version, frame_type, flags, seq, stamp_ns, payload_len, checksum = (
        HEADER_STRUCT.unpack(header)
    )
    if magic != MAGIC:
        raise ValueError(f"bad frame magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported frame version: {version}")
    if payload_len > max_payload_bytes:
        raise ValueError(f"payload too large: {payload_len} > {max_payload_bytes}")
    payload = recv_exact(sock, payload_len)
    actual = zlib.crc32(payload) & 0xFFFFFFFF
    if actual != checksum:
        raise ValueError(f"bad frame crc32: {actual} != {checksum}")
    return Frame(frame_type, flags, seq, stamp_ns, payload)


def configure_tcp(sock: socket.socket, timeout_s: float | None = None) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if timeout_s is not None:
        sock.settimeout(timeout_s)


def pack_tactile_bulk_payload(metadata: dict[str, Any], raw: bytes) -> bytes:
    meta = json_bytes(metadata)
    return struct.pack("!I", len(meta)) + meta + raw


def unpack_tactile_bulk_payload(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if len(payload) < 4:
        raise ValueError("tactile bulk payload is missing metadata length")
    (meta_len,) = struct.unpack("!I", payload[:4])
    if meta_len > len(payload) - 4:
        raise ValueError("tactile bulk metadata length exceeds payload")
    metadata = json_from_bytes(payload[4 : 4 + meta_len])
    raw = payload[4 + meta_len :]
    return metadata, raw
