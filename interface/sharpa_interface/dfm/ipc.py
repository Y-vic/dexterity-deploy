"""Bounded framed MessagePack for the isolated native-client interpreter."""

from __future__ import annotations

import os
import select
import struct
import time

from sharpa_interface.server.serialization import MAX_MESSAGE_SIZE, packb, unpackb


def write_frame(stream, value):
    encoded = packb(value)
    frame = memoryview(struct.pack("!I", len(encoded)) + encoded)
    while frame:
        written = os.write(stream.fileno(), frame)
        if written <= 0:
            raise BrokenPipeError("Native worker IPC write failed")
        frame = frame[written:]


def read_frame(stream, *, timeout_s=None):
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    def read(size):
        result = bytearray()
        while len(result) < size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([stream.fileno()], [], [], remaining)[0]:
                    raise TimeoutError("Native client worker response timed out; reset/reconnect required")
            chunk = os.read(stream.fileno(), size - len(result))
            if not chunk:
                raise EOFError("Native client worker closed its response stream")
            result.extend(chunk)
        return bytes(result)

    size = struct.unpack("!I", read(4))[0]
    if not 0 < size <= MAX_MESSAGE_SIZE:
        raise ValueError("Native worker frame exceeds the bounded message contract")
    return unpackb(read(size))
