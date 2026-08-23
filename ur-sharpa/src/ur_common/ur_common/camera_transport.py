"""Local latest-frame transport for high-rate camera payloads."""

from __future__ import annotations

import mmap
import os
import re
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


CAMERA_NAMES = ("ego", "left_wrist", "right_wrist")
DEFAULT_PREFIX = "ur_sharpa_camera_v1"
_MAGIC = b"USCAMV1\0"
_HEADER = struct.Struct("<8sQQIB3x")


@dataclass(frozen=True)
class SharedCameraSet:
    encoding: str
    ego_timestamp_ns: int
    ego: bytes
    ego_valid: bool
    left_wrist_timestamp_ns: int
    left_wrist: bytes
    left_wrist_valid: bool
    right_wrist_timestamp_ns: int
    right_wrist: bytes
    right_wrist_valid: bool


def _safe_prefix(prefix: str) -> str:
    value = str(prefix).strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("camera shared-memory prefix contains unsafe characters")
    return value


class SharedCameraWriter:
    def __init__(self, prefix=DEFAULT_PREFIX, max_payload_bytes=4 * 1024 * 1024,
                 directory="/dev/shm"):
        self.prefix = _safe_prefix(prefix)
        self.max_payload_bytes = int(max_payload_bytes)
        if self.max_payload_bytes <= 0:
            raise ValueError("camera shared-memory payload size must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._maps = {}
        self._sequences = {name: 0 for name in CAMERA_NAMES}
        size = _HEADER.size + self.max_payload_bytes
        for name in CAMERA_NAMES:
            path = self.directory / f"{self.prefix}_{name}"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.prefix}_{name}.", dir=self.directory
            )
            temporary_path = Path(temporary_name)
            buffer = None
            try:
                os.ftruncate(descriptor, size)
                buffer = mmap.mmap(descriptor, size, access=mmap.ACCESS_WRITE)
                _HEADER.pack_into(buffer, 0, _MAGIC, 0, 0, 0, False)
                metadata = os.fstat(descriptor)
                os.replace(temporary_path, path)
            except Exception:
                if buffer is not None:
                    buffer.close()
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                raise
            finally:
                os.close(descriptor)
            self._maps[name] = (
                path, buffer, (metadata.st_dev, metadata.st_ino)
            )

    def write(self, name: str, timestamp_ns: int, jpeg: bytes, valid=True) -> None:
        if name not in self._maps:
            raise KeyError(name)
        data = bytes(jpeg) if valid else b""
        timestamp = int(timestamp_ns) if valid else 0
        valid = bool(valid and timestamp > 0 and data)
        if not valid:
            data = b""
            timestamp = 0
        if len(data) > self.max_payload_bytes:
            raise ValueError(f"{name} JPEG exceeds shared-memory capacity")
        sequence = self._sequences[name] + 2
        buffer = self._maps[name][1]
        _HEADER.pack_into(buffer, 0, _MAGIC, sequence - 1, timestamp, len(data), valid)
        if data:
            buffer[_HEADER.size:_HEADER.size + len(data)] = data
        _HEADER.pack_into(buffer, 0, _MAGIC, sequence, timestamp, len(data), valid)
        self._sequences[name] = sequence

    def close(self) -> None:
        for path, buffer, identity in self._maps.values():
            try:
                buffer.close()
            finally:
                try:
                    metadata = path.stat()
                    if (metadata.st_dev, metadata.st_ino) == identity:
                        path.unlink()
                except FileNotFoundError:
                    pass
        self._maps.clear()


class SharedCameraReader:
    def __init__(self, prefix=DEFAULT_PREFIX, directory="/dev/shm"):
        self.prefix = _safe_prefix(prefix)
        self.directory = Path(directory)
        self._maps = {}
        self._identities = {}
        self._cache = {}

    def _close_map(self, name):
        descriptor = self._maps.pop(name, None)
        self._identities.pop(name, None)
        self._cache.pop(name, None)
        if descriptor is not None:
            os.close(descriptor)

    def _open(self, name):
        path = self.directory / f"{self.prefix}_{name}"
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            metadata = os.fstat(descriptor)
            if metadata.st_size < _HEADER.size:
                os.close(descriptor)
                return None
        except Exception:
            os.close(descriptor)
            raise
        self._maps[name] = descriptor
        self._identities[name] = (metadata.st_dev, metadata.st_ino)
        self._cache.pop(name, None)
        return descriptor

    def _read(self, name):
        path = self.directory / f"{self.prefix}_{name}"
        try:
            metadata = path.stat()
        except FileNotFoundError:
            self._close_map(name)
            return None
        if self._identities.get(name) != (metadata.st_dev, metadata.st_ino):
            self._close_map(name)
        descriptor = self._maps.get(name)
        if descriptor is None:
            descriptor = self._open(name)
        if descriptor is None:
            return None
        for _ in range(4):
            header = os.pread(descriptor, _HEADER.size, 0)
            if len(header) != _HEADER.size:
                time.sleep(0)
                continue
            magic, sequence, timestamp, size, valid = _HEADER.unpack(header)
            cached = self._cache.get(name)
            if magic != _MAGIC or sequence & 1:
                time.sleep(0)
                continue
            cache_key = (sequence, timestamp, size, bool(valid))
            if cached is not None and cached[0] == cache_key:
                return cached[1]
            try:
                available = os.fstat(descriptor).st_size - _HEADER.size
            except OSError:
                return None
            if size > available:
                return None
            data = os.pread(descriptor, size, _HEADER.size) if valid else b""
            if valid and len(data) != size:
                time.sleep(0)
                continue
            second_header = os.pread(descriptor, _HEADER.size, 0)
            if len(second_header) != _HEADER.size:
                time.sleep(0)
                continue
            second = _HEADER.unpack(second_header)
            if second[0] == _MAGIC and second[1] == sequence and not sequence & 1:
                value = (int(timestamp), data, bool(valid and timestamp > 0 and data))
                self._cache[name] = (cache_key, value)
                return value
        return None

    def read(self):
        values = {name: self._read(name) for name in CAMERA_NAMES}
        if not any(value is not None for value in values.values()):
            return None
        values = {name: value or (0, b"", False) for name, value in values.items()}
        return SharedCameraSet(
            encoding="jpeg",
            ego_timestamp_ns=values["ego"][0], ego=values["ego"][1], ego_valid=values["ego"][2],
            left_wrist_timestamp_ns=values["left_wrist"][0], left_wrist=values["left_wrist"][1], left_wrist_valid=values["left_wrist"][2],
            right_wrist_timestamp_ns=values["right_wrist"][0], right_wrist=values["right_wrist"][1], right_wrist_valid=values["right_wrist"][2],
        )

    def close(self) -> None:
        for name in tuple(self._maps):
            self._close_map(name)
        self._cache.clear()
