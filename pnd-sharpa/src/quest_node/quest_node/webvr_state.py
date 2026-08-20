"""Thread-safe receive state for the Quest WebVR socket."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from quest_node.webvr_protocol import WebVRSample


@dataclass(frozen=True)
class ReceiverSnapshot:
    connected: bool
    connection_version: int
    disconnect_epoch: int
    sequence: int
    sample: WebVRSample | None
    received_at: float | None
    invalid_frames: int
    source_sequence_gaps: int
    source_sequence_resets: int
    last_error: str


def calibration_is_stale(
    *,
    calibrated: bool,
    connected: bool,
    last_frame_time: float | None,
    now: float,
    timeout: float,
) -> bool:
    return (
        calibrated
        and connected
        and last_frame_time is not None
        and float(now) - last_frame_time > timeout
    )


class WebVRReceiverState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connected = False
        self._connection_version = 0
        self._disconnect_epoch = 0
        self._sequence = 0
        self._latest_sample: WebVRSample | None = None
        self._latest_received_at: float | None = None
        self._invalid_frames = 0
        self._last_source_sequence: int | None = None
        self._source_sequence_gaps = 0
        self._source_sequence_resets = 0
        self._last_error = ""

    def set_connection(self, connected: bool) -> None:
        with self._lock:
            if not connected:
                self._disconnect_epoch += 1
                self._latest_sample = None
                self._latest_received_at = None
                self._last_source_sequence = None
            if self._connected != connected:
                self._connected = connected
                self._connection_version += 1

    def record_sample(self, sample: WebVRSample, received_at: float) -> None:
        with self._lock:
            source_sequence = sample.source_sequence
            if source_sequence is not None:
                if self._last_source_sequence is not None:
                    if source_sequence > self._last_source_sequence + 1:
                        self._source_sequence_gaps += (
                            source_sequence - self._last_source_sequence - 1
                        )
                    elif source_sequence <= self._last_source_sequence:
                        self._source_sequence_resets += 1
                self._last_source_sequence = source_sequence
            self._sequence += 1
            self._latest_sample = sample
            self._latest_received_at = received_at
            self._last_error = ""

    def record_error(self, reason: str) -> None:
        with self._lock:
            self._invalid_frames += 1
            self._last_error = reason[:300]

    def snapshot(self) -> ReceiverSnapshot:
        with self._lock:
            return ReceiverSnapshot(
                connected=self._connected,
                connection_version=self._connection_version,
                disconnect_epoch=self._disconnect_epoch,
                sequence=self._sequence,
                sample=self._latest_sample,
                received_at=self._latest_received_at,
                invalid_frames=self._invalid_frames,
                source_sequence_gaps=self._source_sequence_gaps,
                source_sequence_resets=self._source_sequence_resets,
                last_error=self._last_error,
            )

    def process_if_current(
        self,
        snapshot: ReceiverSnapshot,
        callback: Callable[[], None],
    ) -> bool:
        with self._lock:
            if (
                not self._connected
                or self._connection_version != snapshot.connection_version
                or self._disconnect_epoch != snapshot.disconnect_epoch
                or self._sequence != snapshot.sequence
                or self._latest_sample is not snapshot.sample
                or self._latest_received_at != snapshot.received_at
            ):
                return False
            callback()
            return True
