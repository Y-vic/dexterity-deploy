"""Isolate the ZED SDK's blocking, GIL-holding calls from ROS and wrist capture."""

from __future__ import annotations

import importlib
import multiprocessing
import time
import numpy as np

from .drivers import (
    HardwareConnectionError,
    HardwareDependencyError,
    ZedEgoCamera,
    ensure_optional_runtime_paths,
)


def _capture_worker(connection, stop, factory, options, storage, slot_bytes, slots):
    """Pipe carries descriptors only; acknowledged slots cannot be overwritten."""
    camera = None
    available = list(range(slots))
    outstanding = set()
    shared = np.frombuffer(storage, dtype=np.uint8).reshape(slots, slot_bytes)
    try:
        camera = factory(**options)
        camera.connect()
        connection.send(("ready", None))
        while not stop.is_set():
            while connection.poll(0 if available else .01):
                slot = connection.recv()
                if slot not in outstanding:
                    raise RuntimeError("invalid ZED shared-frame acknowledgement")
                outstanding.remove(slot)
                available.append(slot)
            if not available:
                continue
            try:
                frame = camera._grab_frame()
                pixels, stamp, started, ended = frame
                if pixels is None:
                    connection.send(("frame", frame))
                    stop.wait(.001)
                    continue
                pixels = np.asarray(pixels)
                if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
                        or min(pixels.shape) <= 0 or pixels.nbytes > slot_bytes):
                    raise ValueError("invalid or oversized ZED BGR frame")
                slot = available.pop()
                shared[slot, :pixels.nbytes] = pixels.reshape(-1)
                outstanding.add(slot)
                connection.send(("shared_frame", (slot, pixels.shape, stamp, started, ended)))
            except Exception as exc:
                connection.send(("error", str(exc)))
                stop.wait(0.01)
                continue
    except (BrokenPipeError, EOFError):
        pass
    except Exception as exc:
        try:
            kind = "dependency_error" if isinstance(exc, HardwareDependencyError) else "error"
            connection.send((kind, str(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if camera is not None:
            camera.close()
        connection.close()


class IsolatedZedEgoCamera(ZedEgoCamera):
    """Run only SDK capture in a spawned process; retain existing frame queues.

    JPEG, raw recording, diagnostics and ROS publication stay in the parent.
    The child returns the original IMAGE timestamp and monotonic grab timings.
    No CUDA context is inherited and no timestamps are replaced on receipt.
    """

    _camera_factory = ZedEgoCamera
    _connect_timeout_s = 25.0

    def __init__(self, **options):
        super().__init__(**options)
        self._capture_options = options
        self._capture_process = None
        self._capture_connection = None
        self._capture_stop = None
        self._capture_storage = None
        self._capture_slot_bytes = 8 * 1024 * 1024  # Includes ZED HD2K BGR.
        self._capture_slots = 2

    def connect(self):
        if self._capture_process is not None:
            raise RuntimeError("close isolated camera before reconnecting")
        # Keep SDK enumeration available for camera_node's connection diagnosis;
        # only the child creates/opens a Camera and its CUDA context.
        ensure_optional_runtime_paths()
        try:
            self.sl = importlib.import_module("pyzed.sl")
        except (ImportError, OSError) as exc:
            raise HardwareDependencyError("pyzed is unavailable; install the ZED SDK Python API") from exc
        self._stop.clear()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._capture_connection = parent
        self._capture_stop = context.Event()
        self._capture_storage = context.RawArray("B", self._capture_slot_bytes * self._capture_slots)
        process = context.Process(
            target=_capture_worker,
            args=(child, self._capture_stop, self._camera_factory, self._capture_options,
                  self._capture_storage, self._capture_slot_bytes, self._capture_slots),
            name="zed-sdk-capture",
            daemon=True,
        )
        self._capture_process = process
        try:
            process.start()
            child.close()
            kind, value = self._receive(self._connect_timeout_s)
            if kind == "dependency_error":
                raise HardwareDependencyError(value)
            if kind != "ready":
                raise HardwareConnectionError(f"ZED capture process failed to open: {value}")
        except Exception:
            child.close()
            self.close()
            raise

    def _receive(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while not self._stop.is_set():
            connection = self._capture_connection
            if connection is None:
                break
            try:
                if connection.poll(0.05):
                    return connection.recv()
            except (EOFError, OSError) as exc:
                raise HardwareConnectionError("ZED capture process disconnected") from exc
            if not self._capture_process.is_alive():
                raise HardwareConnectionError("ZED capture process exited")
            if time.monotonic() >= deadline:
                raise HardwareConnectionError("ZED capture process timed out")
        raise HardwareConnectionError("ZED capture process stopped")

    def _grab_frame(self):
        kind, value = self._receive(5.0)
        if kind == "shared_frame":
            slot, shape, stamp, started, ended = value
            size = int(np.prod(shape))
            if (slot not in range(self._capture_slots) or len(shape) != 3 or shape[2] != 3
                    or min(shape) <= 0 or size > self._capture_slot_bytes):
                raise HardwareConnectionError("invalid ZED shared frame descriptor")
            # Own the pixels before returning the credit: no torn/reused images.
            image = np.frombuffer(self._capture_storage, dtype=np.uint8,
                                  count=size, offset=slot * self._capture_slot_bytes).reshape(shape).copy()
            self._capture_connection.send(slot)
            return image, stamp, started, ended
        if kind != "frame":
            raise HardwareConnectionError(f"ZED capture failed: {value}")
        return value

    def diagnostics(self, *, include_frame_timings=False):
        result = super().diagnostics(include_frame_timings=include_frame_timings)
        process = self._capture_process
        result.update(capture_process_pid=process.pid if process is not None else None,
                      capture_process_alive=bool(process is not None and process.is_alive()),
                      capture_transport="acknowledged_shared_memory")
        return result

    def close(self):
        self._stop.set()
        if self._capture_stop is not None:
            self._capture_stop.set()
        process = self._capture_process
        if process is not None and process.pid is not None:
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
        # Closing the child first also unblocks a parent recv of a partial frame.
        super().close()
        if self._capture_connection is not None:
            self._capture_connection.close()
        if process is not None and process.pid is not None:
            process.close()
        self._capture_connection = self._capture_process = self._capture_stop = None
        self._capture_storage = None
