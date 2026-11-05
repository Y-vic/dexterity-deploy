"""Local loss-resistant transport for high-rate camera payloads.

Each camera owns a small shared-memory ring. Latest-frame consumers such as
Monitor can keep using :meth:`SharedCameraReader.read`, while recording drains
every retained frame with :meth:`SharedCameraReader.read_pending`.
"""

from __future__ import annotations

import mmap
import os
import re
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .raw_camera import RAW_ENCODING, complete_bgr


CAMERA_NAMES = ("ego", "left_wrist", "right_wrist")
DEFAULT_PREFIX = "ur_sharpa_camera_v1"
DEFAULT_SLOT_COUNT = 64
_MAGIC = b"USCAMV2\0"
_FILE_HEADER = struct.Struct("<8sIIQ")
_TAIL = struct.Struct("<Q")
_TAIL_OFFSET = _FILE_HEADER.size - _TAIL.size
_SLOT_HEADER = struct.Struct("<QQIB3x")


def _complete_jpeg(data: bytes) -> bool:
    """Cheaply reject empty/truncated shared-memory snapshots."""
    return len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


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
                 directory="/dev/shm", slot_count=DEFAULT_SLOT_COUNT, reserve_space=False,
                 *, names=CAMERA_NAMES):
        self.names = tuple(_safe_prefix(name) for name in names)
        self.prefix = _safe_prefix(prefix)
        self.max_payload_bytes = int(max_payload_bytes)
        if self.max_payload_bytes <= 0:
            raise ValueError("camera shared-memory payload size must be positive")
        self.slot_count = int(slot_count)
        if self.slot_count <= 1:
            raise ValueError("camera shared-memory slot count must be greater than one")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._maps = {}
        self._frame_ids = {name: 0 for name in self.names}
        self._slot_size = _SLOT_HEADER.size + self.max_payload_bytes
        size = _FILE_HEADER.size + self.slot_count * self._slot_size
        for name in self.names:
            path = self.directory / f"{self.prefix}_{name}"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.prefix}_{name}.", dir=self.directory
            )
            temporary_path = Path(temporary_name)
            buffer = None
            try:
                os.ftruncate(descriptor, size)
                if reserve_space:
                    # Reserve tmpfs pages now: ENOSPC is catchable here, whereas
                    # touching an unbacked sparse mmap page can kill us (SIGBUS).
                    os.posix_fallocate(descriptor, 0, size)
                buffer = mmap.mmap(descriptor, size, access=mmap.ACCESS_WRITE)
                _FILE_HEADER.pack_into(
                    buffer,
                    0,
                    _MAGIC,
                    self.slot_count,
                    self.max_payload_bytes,
                    0,
                )
                metadata = os.fstat(descriptor)
                os.replace(temporary_path, path)
            except Exception:
                if buffer is not None:
                    buffer.close()
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                self.close()
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
            raise ValueError(f"{name} frame exceeds shared-memory capacity")
        frame_id = self._frame_ids[name] + 1
        sequence = frame_id * 2
        slot_index = (frame_id - 1) % self.slot_count
        slot_offset = _FILE_HEADER.size + slot_index * self._slot_size
        buffer = self._maps[name][1]
        _SLOT_HEADER.pack_into(
            buffer, slot_offset, sequence - 1, timestamp, len(data), valid
        )
        if data:
            payload_offset = slot_offset + _SLOT_HEADER.size
            buffer[payload_offset:payload_offset + len(data)] = data
        _SLOT_HEADER.pack_into(
            buffer, slot_offset, sequence, timestamp, len(data), valid
        )
        # Do not pack_into the live header: struct clears the destination before
        # filling its fields, exposing zero/partial metadata to other processes.
        # Keep the configuration immutable and publish only the prepared tail.
        buffer[_TAIL_OFFSET:_FILE_HEADER.size] = _TAIL.pack(sequence)
        self._frame_ids[name] = frame_id

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
    def __init__(self, prefix=DEFAULT_PREFIX, directory="/dev/shm", encoding="jpeg",
                 *, names=CAMERA_NAMES, validate_payload=None):
        if encoding not in ("jpeg", RAW_ENCODING) and validate_payload is None:
            raise ValueError("unsupported shared camera encoding")
        self.names = tuple(_safe_prefix(name) for name in names)
        self._validate_payload = validate_payload or (complete_bgr if encoding == RAW_ENCODING else _complete_jpeg)
        self.encoding = encoding
        self.prefix = _safe_prefix(prefix)
        self.directory = Path(directory)
        self._maps = {}
        self._identities = {}
        self._configs = {}
        self._cache = {}
        self._cursors = {}
        self._pending_values = {
            name: (0, b"", False) for name in self.names
        }
        self.dropped_frames = {name: 0 for name in self.names}

    def _close_map(self, name):
        descriptor = self._maps.pop(name, None)
        self._identities.pop(name, None)
        self._configs.pop(name, None)
        self._cache.pop(name, None)
        self._cursors.pop(name, None)
        self._pending_values[name] = (0, b"", False)
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
            header = os.pread(descriptor, _FILE_HEADER.size, 0)
            if len(header) != _FILE_HEADER.size:
                os.close(descriptor)
                return None
            magic, slot_count, max_payload_bytes, _ = _FILE_HEADER.unpack(header)
            expected_size = (
                _FILE_HEADER.size
                + int(slot_count) * (_SLOT_HEADER.size + int(max_payload_bytes))
            )
            if (
                magic != _MAGIC
                or int(slot_count) <= 1
                or int(max_payload_bytes) <= 0
                or metadata.st_size < expected_size
            ):
                os.close(descriptor)
                return None
        except Exception:
            os.close(descriptor)
            raise
        self._maps[name] = descriptor
        self._identities[name] = (metadata.st_dev, metadata.st_ino)
        self._configs[name] = (int(slot_count), int(max_payload_bytes))
        self._cache.pop(name, None)
        return descriptor

    def _metadata(self, name):
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
        header = os.pread(descriptor, _FILE_HEADER.size, 0)
        if len(header) != _FILE_HEADER.size:
            return None
        magic, slot_count, max_payload_bytes, latest_sequence = _FILE_HEADER.unpack(
            header
        )
        if magic != _MAGIC or latest_sequence & 1:
            return None
        config = self._configs.get(name)
        if config != (int(slot_count), int(max_payload_bytes)):
            self._close_map(name)
            return None
        return descriptor, config, int(latest_sequence)

    @staticmethod
    def _slot_offset(frame_id, slot_count, max_payload_bytes):
        slot_size = _SLOT_HEADER.size + max_payload_bytes
        return _FILE_HEADER.size + ((frame_id - 1) % slot_count) * slot_size

    def _read_frame(self, descriptor, config, frame_id):
        slot_count, max_payload_bytes = config
        expected_sequence = int(frame_id) * 2
        slot_offset = self._slot_offset(
            frame_id, slot_count, max_payload_bytes
        )
        for _ in range(4):
            header = os.pread(descriptor, _SLOT_HEADER.size, slot_offset)
            if len(header) != _SLOT_HEADER.size:
                time.sleep(0)
                continue
            sequence, timestamp, size, valid = _SLOT_HEADER.unpack(header)
            if sequence != expected_sequence or sequence & 1:
                time.sleep(0)
                continue
            if size > max_payload_bytes:
                return None
            payload_offset = slot_offset + _SLOT_HEADER.size
            data = os.pread(descriptor, size, payload_offset) if valid else b""
            if valid and (len(data) != size or not self._validate_payload(data)):
                time.sleep(0)
                continue
            second_header = os.pread(
                descriptor, _SLOT_HEADER.size, slot_offset
            )
            if len(second_header) != _SLOT_HEADER.size:
                time.sleep(0)
                continue
            second = _SLOT_HEADER.unpack(second_header)
            if second[0] == expected_sequence and not expected_sequence & 1:
                return int(timestamp), data, bool(valid and timestamp > 0 and data)
        return None

    def _read_latest(self, name):
        for attempt in range(4):
            if attempt:
                time.sleep(0)
            # Refresh the tail as well as the slot: the ring may have wrapped,
            # or a legacy writer may have been partway through a header update.
            metadata = self._metadata(name)
            if metadata is None:
                continue
            descriptor, config, latest_sequence = metadata
            if latest_sequence == 0:
                continue
            cached = self._cache.get(name)
            if cached is not None and cached[0] == latest_sequence:
                return cached[1]
            value = self._read_frame(descriptor, config, latest_sequence // 2)
            if value is not None:
                self._cache[name] = (latest_sequence, value)
                # An explicitly invalid frame is authoritative; never replace
                # it with a cached valid image or change a source timestamp.
                return value
        return (0, b"", False) if metadata is not None else None

    def _make_set(self, values):
        values = {
            name: values.get(name) or (0, b"", False)
            for name in CAMERA_NAMES
        }
        return SharedCameraSet(
            encoding=self.encoding,
            ego_timestamp_ns=values["ego"][0],
            ego=values["ego"][1],
            ego_valid=values["ego"][2],
            left_wrist_timestamp_ns=values["left_wrist"][0],
            left_wrist=values["left_wrist"][1],
            left_wrist_valid=values["left_wrist"][2],
            right_wrist_timestamp_ns=values["right_wrist"][0],
            right_wrist=values["right_wrist"][1],
            right_wrist_valid=values["right_wrist"][2],
        )

    def read(self):
        values = {name: self._read_latest(name) for name in CAMERA_NAMES}
        if not any(value is not None for value in values.values()):
            return None
        return self._make_set(values)

    def read_entries(self):
        """Drain validated (timestamp, stream name, value) entries without copying views."""
        events = []
        for name in self.names:
            metadata = self._metadata(name)
            if metadata is None:
                continue
            descriptor, config, latest_sequence = metadata
            latest_frame_id = latest_sequence // 2
            if latest_frame_id <= 0:
                continue
            slot_count, _ = config
            oldest_frame_id = max(1, latest_frame_id - slot_count + 1)
            previous = self._cursors.get(name, oldest_frame_id - 1)
            if previous < oldest_frame_id - 1:
                self.dropped_frames[name] += oldest_frame_id - previous - 1
                previous = oldest_frame_id - 1
            for frame_id in range(previous + 1, latest_frame_id + 1):
                value = self._read_frame(descriptor, config, frame_id)
                if value is None:
                    self.dropped_frames[name] += 1
                    continue
                events.append((int(value[0]), name, value))
            self._cursors[name] = latest_frame_id
        if not events:
            return []
        events.sort(key=lambda event: (event[0], self.names.index(event[1])))
        return events

    def read_pending(self):
        """Return every unread ring entry, with other views filled from latest."""
        events = self.read_entries()
        frames = []
        for _, name, value in events:
            self._pending_values[name] = value
            frames.append(self._make_set(dict(self._pending_values)))
        return frames

    def close(self) -> None:
        for name in tuple(self._maps):
            self._close_map(name)
        self._cache.clear()
        self._cursors.clear()
        self._pending_values = {
            name: (0, b"", False) for name in self.names
        }
