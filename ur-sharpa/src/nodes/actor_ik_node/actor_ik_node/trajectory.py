"""Actor trajectory serialization and bounded Deploy execution tracing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any

import numpy as np


TRACE_SCHEMA = "sharpa.actor_ik.execution_trace.v2"
LEGACY_PREVIEW_SCHEMA = "sharpa.actor_ik.preview_trace.v1"
TRAJECTORY_SCHEMA = "sharpa.actor_ik.trajectory.v1"
DEFAULT_RETENTION_DAYS = 7
DEFAULT_MAX_RUNS = 100
DEFAULT_MAX_TOTAL_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_EXTERNAL_EVENT_QUEUE_CAPACITY = 64


def _json_default(value: Any) -> Any:
    """Preserve numeric ROS/NumPy values in the existing JSON list schema."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_trajectory_payload(
    action: Any,
    plan: Any,
    *,
    provider: str,
    dry_run: bool,
    trace_dir: str = "",
) -> dict[str, Any]:
    """Build the canonical chunk published after Actor planning succeeds."""
    left = np.asarray(plan.left_joint, dtype=np.float32)
    right = np.asarray(plan.right_joint, dtype=np.float32)
    hand = np.asarray(plan.hand_joint, dtype=np.float32)
    length = int(action.execute_length)
    if left.shape != (length, 6) or right.shape != (length, 6):
        raise ValueError("Actor arm plan must have shape (execute_length, 6)")
    if hand.shape != (length, 44):
        raise ValueError("Actor hand plan must have shape (execute_length, 44)")
    if not all(np.isfinite(values).all() for values in (left, right, hand)):
        raise ValueError("Actor trajectory contains non-finite joints")

    def max_step(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(values.astype(np.float64), axis=0))))

    diagnostics = {
        "schema": TRAJECTORY_SCHEMA,
        "left_arm_max_step_rad": max_step(left),
        "right_arm_max_step_rad": max_step(right),
        "hand_max_step_rad": max_step(hand),
        "left_eef_present": bool(action.left_eef),
        "right_eef_present": bool(action.right_eef),
    }
    return {
        "provider": str(provider),
        "dry_run": bool(dry_run),
        "trace_dir": str(trace_dir),
        "session_id": str(action.session_id),
        "request_id": int(action.request_id),
        "action_id": str(action.action_id),
        "revision": int(action.revision),
        "timestamp_ns": int(action.timestamp_ns),
        "frequency_hz": float(action.frequency_hz),
        "action_length": int(action.action_length),
        "execute_start": int(action.execute_start),
        "execute_length": length,
        "rtc": int(getattr(action, "rtc", 0)),
        "eef_frame": str(action.eef_frame),
        "left_eef": [float(value) for value in action.left_eef],
        "right_eef": [float(value) for value in action.right_eef],
        "left_arm_joint": left.reshape(-1).tolist(),
        "right_arm_joint": right.reshape(-1).tolist(),
        "hand_joint": hand.reshape(-1).tolist(),
        "ik_applied": bool(action.left_eef and action.right_eef),
        "diagnostics": diagnostics,
    }


class ExecutionTraceWriter:
    """Durable live/preview Actor journal with bounded on-disk retention."""

    def __init__(
        self,
        root: str | Path,
        *,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_runs: int = DEFAULT_MAX_RUNS,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        external_event_queue_capacity: int = DEFAULT_EXTERNAL_EVENT_QUEUE_CAPACITY,
    ) -> None:
        self._lock = threading.RLock()
        self.root = Path(root).expanduser().resolve()
        self.retention_days = int(retention_days)
        self.max_runs = int(max_runs)
        self.max_total_bytes = int(max_total_bytes)
        self.external_event_queue_capacity = int(external_event_queue_capacity)
        if self.retention_days <= 0:
            raise ValueError("retention_days must be positive")
        if self.max_runs <= 0:
            raise ValueError("max_runs must be positive")
        if self.max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be positive")
        if self.external_event_queue_capacity <= 0:
            raise ValueError("external_event_queue_capacity must be positive")
        self.run_dir: Path | None = None
        self.started_at_ns = 0
        self.trajectory_count = 0
        self.event_count = 0
        self.failure_count = 0
        self.ack_late_warning_count = 0
        self.late_ack_count = 0
        self.max_ack_delivery_latency_ms = 0.0
        self.mode = ""
        self.provider = ""
        self._external_queue: queue.Queue[dict[str, Any]] | None = None
        self._external_stop: threading.Event | None = None
        self._external_worker: threading.Thread | None = None
        self._external_accepting = False
        self._external_stats: dict[str, Any] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover_abandoned_runs()
        self.cleanup()

    @property
    def active(self) -> bool:
        with self._lock:
            return self.run_dir is not None

    @property
    def path(self) -> str:
        with self._lock:
            return str(self.run_dir) if self.run_dir is not None else ""

    def start(self, *, mode: str = "preview", provider: str = "policy") -> str:
        with self._lock:
            if self.run_dir is not None:
                return str(self.run_dir)
            self.cleanup()
            stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            suffix = f"{time.time_ns() % 1_000_000_000:09d}"
            self.run_dir = self.root / f"{stamp}.{suffix}"
            self.run_dir.mkdir(parents=True, exist_ok=False)
            self.started_at_ns = time.time_ns()
            self.trajectory_count = 0
            self.event_count = 0
            self.failure_count = 0
            self.ack_late_warning_count = 0
            self.late_ack_count = 0
            self.max_ack_delivery_latency_ms = 0.0
            self._external_stats = {
                "attempted_count": 0,
                "enqueued_count": 0,
                "written_count": 0,
                "dropped_count": 0,
                "write_error_count": 0,
                "last_write_error": "",
            }
            self.mode = str(mode)
            self.provider = str(provider)
            self._write_json(
                "manifest.json",
                {
                    "schema": TRACE_SCHEMA,
                    "created_at_ns": self.started_at_ns,
                    "state": "active",
                    "mode": self.mode,
                    "provider": self.provider,
                    "trajectory_source": "/actor_ik/trajectory",
                    "actor_event_journal": "events.jsonl",
                    "node_event_journal": "node_events.jsonl",
                    "coordinate_frame": "canonical",
                    "retention": {
                        "days": self.retention_days,
                        "max_runs": self.max_runs,
                        "max_total_bytes": self.max_total_bytes,
                    },
                },
            )
            self.event("execution_started", mode=self.mode, provider=self.provider)
            self._start_external_worker()
            return str(self.run_dir)

    def trajectory(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if self.run_dir is None:
                return
            self._append_jsonl("trajectories.jsonl", payload)
            self.trajectory_count += 1

    def event(self, event: str, **fields: Any) -> None:
        with self._lock:
            if self.run_dir is None:
                return
            record = {
                "event": str(event),
                "timestamp_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                **fields,
            }
            self.record(record)

    def record(self, record: dict[str, Any]) -> None:
        """Append an already structured event from another Deploy node."""
        with self._lock:
            if self.run_dir is None:
                return
            value = dict(record)
            value.setdefault("timestamp_ns", time.time_ns())
            value.setdefault("monotonic_ns", time.monotonic_ns())
            self._append_jsonl("events.jsonl", value)
            self.event_count += 1
            if value.get("success") is False:
                self.failure_count += 1
            event = str(value.get("event") or "")
            if event == "hardware_command_ack_late_warning":
                self.ack_late_warning_count += 1
            elif event == "late_hardware_command_ack":
                self.late_ack_count += 1
            try:
                latency_ms = float(value.get("ack_delivery_latency_ms", 0.0))
            except (TypeError, ValueError):
                latency_ms = 0.0
            self.max_ack_delivery_latency_ms = max(
                self.max_ack_delivery_latency_ms, latency_ms
            )

    def record_external(self, record: dict[str, Any]) -> bool:
        """Queue a non-critical cross-node event without blocking Actor control.

        External events are diagnostic samples.  Their queue is intentionally
        bounded: when storage cannot keep up, new samples are dropped instead
        of allowing the ROS subscription backlog to delay hardware ACKs.
        Actor-owned ACK warnings, late-ACK and failure events still use
        ``event`` and are written to the primary journal.
        """
        external_queue = self._external_queue
        if not self._external_accepting or external_queue is None:
            return False
        value = dict(record)
        value.setdefault("timestamp_ns", time.time_ns())
        value.setdefault("monotonic_ns", time.monotonic_ns())
        self._external_stats["attempted_count"] += 1
        try:
            external_queue.put_nowait(value)
        except queue.Full:
            self._external_stats["dropped_count"] += 1
            return False
        self._external_stats["enqueued_count"] += 1
        return True

    def _start_external_worker(self) -> None:
        self._external_queue = queue.Queue(
            maxsize=self.external_event_queue_capacity
        )
        self._external_stop = threading.Event()
        self._external_accepting = True
        run_dir = self.run_dir
        assert run_dir is not None
        self._external_worker = threading.Thread(
            target=self._external_worker_loop,
            args=(
                run_dir,
                self._external_queue,
                self._external_stop,
                self._external_stats,
            ),
            name="actor-execution-trace",
            daemon=True,
        )
        self._external_worker.start()

    def _external_worker_loop(
        self,
        run_dir: Path,
        external_queue: queue.Queue[dict[str, Any]],
        stop_event: threading.Event,
        stats: dict[str, Any],
    ) -> None:
        while not stop_event.is_set():
            try:
                value = external_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if stop_event.is_set():
                    continue
                self._append_jsonl_at(run_dir / "node_events.jsonl", value)
                stats["written_count"] += 1
            except Exception as exc:
                stats["write_error_count"] += 1
                stats["last_write_error"] = str(exc)
            finally:
                external_queue.task_done()

    def _stop_external_worker(self) -> tuple[int, bool]:
        self._external_accepting = False
        external_queue = self._external_queue
        stop_event = self._external_stop
        worker = self._external_worker
        if stop_event is not None:
            stop_event.set()
        if worker is not None:
            worker.join(timeout=0.25)
        worker_alive = bool(worker is not None and worker.is_alive())
        pending = external_queue.qsize() if external_queue is not None else 0
        self._external_stats["dropped_count"] = (
            int(self._external_stats.get("dropped_count", 0)) + pending
        )
        self._external_queue = None
        self._external_stop = None
        self._external_worker = None
        return pending, worker_alive

    def stop(
        self,
        reason: str = "execution_stopped",
        *,
        final_state: dict[str, Any] | None = None,
    ) -> str:
        pending_external, worker_shutdown_timed_out = self._stop_external_worker()
        with self._lock:
            if self.run_dir is None:
                return ""
            path = str(self.run_dir)
            self.event(reason)
            finished_at_ns = time.time_ns()
            if final_state is not None:
                self._write_json("final_state.json", final_state)
            self._write_json(
                "summary.json",
                {
                    "schema": TRACE_SCHEMA,
                    "started_at_ns": self.started_at_ns,
                    "finished_at_ns": finished_at_ns,
                    "mode": self.mode,
                    "provider": self.provider,
                    "trajectory_count": self.trajectory_count,
                    "event_count": self.event_count,
                    "failure_count": self.failure_count,
                    "ack_diagnostics": {
                        "late_warning_count": self.ack_late_warning_count,
                        "late_ack_count": self.late_ack_count,
                        "max_delivery_latency_ms": self.max_ack_delivery_latency_ms,
                    },
                    "external_trace_diagnostics": {
                        "queue_capacity": self.external_event_queue_capacity,
                        "attempted_count": int(
                            self._external_stats.get("attempted_count", 0)
                        ),
                        "enqueued_count": int(
                            self._external_stats.get("enqueued_count", 0)
                        ),
                        "written_count": int(
                            self._external_stats.get("written_count", 0)
                        ),
                        "dropped_count": int(
                            self._external_stats.get("dropped_count", 0)
                        ),
                        "pending_at_stop": pending_external,
                        "write_error_count": int(
                            self._external_stats.get("write_error_count", 0)
                        ),
                        "last_write_error": str(
                            self._external_stats.get("last_write_error", "")
                        ),
                        "worker_shutdown_timed_out": worker_shutdown_timed_out,
                    },
                    "reason": str(reason),
                },
            )
            manifest = self._read_json(self.run_dir / "manifest.json") or {}
            manifest.update(state="complete", finished_at_ns=finished_at_ns)
            self._write_json("manifest.json", manifest)
            self.run_dir = None
            self.mode = ""
            self.provider = ""
            self.cleanup()
            return path

    def cleanup(self, *, now_ns: int | None = None) -> list[str]:
        """Delete only recognized, completed trace dirs that exceed retention."""
        with self._lock:
            return self._cleanup_locked(now_ns=now_ns)

    def _cleanup_locked(self, *, now_ns: int | None = None) -> list[str]:
        now_ns = time.time_ns() if now_ns is None else int(now_ns)
        cutoff_ns = now_ns - self.retention_days * 24 * 60 * 60 * 1_000_000_000
        runs = self._completed_runs()
        removed: list[str] = []

        def remove(run: dict[str, Any]) -> None:
            path = run["path"]
            allowed_parents = {self.root, self.root / "legacy_previews"}
            if (
                path == self.run_dir
                or path.parent not in allowed_parents
                or path.is_symlink()
            ):
                return
            shutil.rmtree(path)
            removed.append(str(path))

        for run in list(runs):
            if run["created_at_ns"] < cutoff_ns:
                remove(run)
        runs = [run for run in self._completed_runs() if str(run["path"]) not in removed]
        while len(runs) > self.max_runs:
            run = runs.pop(0)
            remove(run)
        total_bytes = sum(run["bytes"] for run in runs)
        while runs and total_bytes > self.max_total_bytes:
            run = runs.pop(0)
            total_bytes -= run["bytes"]
            remove(run)
        return removed

    def _recover_abandoned_runs(self) -> None:
        """A new Actor process makes every older active manifest abandoned."""
        for path in self.root.iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            manifest_path = path / "manifest.json"
            manifest = self._read_json(manifest_path)
            if not manifest or manifest.get("schema") != TRACE_SCHEMA:
                continue
            if manifest.get("state") != "active":
                continue
            finished_at_ns = time.time_ns()
            manifest.update(state="abandoned", finished_at_ns=finished_at_ns)
            self._write_json_at(manifest_path, manifest)
            summary_path = path / "summary.json"
            if not summary_path.exists():
                self._write_json_at(
                    summary_path,
                    {
                        "schema": TRACE_SCHEMA,
                        "started_at_ns": int(manifest.get("created_at_ns") or 0),
                        "finished_at_ns": finished_at_ns,
                        "mode": str(manifest.get("mode") or ""),
                        "provider": str(manifest.get("provider") or ""),
                        "reason": "actor_process_restarted",
                    },
                )

    def _completed_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        paths = list(self.root.iterdir())
        legacy_root = self.root / "legacy_previews"
        if legacy_root.is_dir() and not legacy_root.is_symlink():
            paths.extend(legacy_root.iterdir())
        for path in paths:
            if not path.is_dir() or path.is_symlink() or path == self.run_dir:
                continue
            manifest = self._read_json(path / "manifest.json")
            if not manifest:
                continue
            schema = manifest.get("schema")
            completed = bool(
                schema == TRACE_SCHEMA
                and manifest.get("state") in {"complete", "abandoned"}
            )
            legacy_completed = bool(
                schema == LEGACY_PREVIEW_SCHEMA
                and path.parent == legacy_root
            )
            if not (completed or legacy_completed):
                continue
            runs.append(
                {
                    "path": path,
                    "created_at_ns": int(manifest.get("created_at_ns") or 0),
                    "bytes": sum(
                        item.stat().st_size
                        for item in path.rglob("*")
                        if item.is_file() and not item.is_symlink()
                    ),
                }
            )
        return sorted(runs, key=lambda run: (run["created_at_ns"], str(run["path"])))

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return value if isinstance(value, dict) else None

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        assert self.run_dir is not None
        self._write_json_at(self.run_dir / name, payload)

    @staticmethod
    def _write_json_at(path: Path, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, default=_json_default, ensure_ascii=False, separators=(",", ":"))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def _append_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        assert self.run_dir is not None
        self._append_jsonl_at(self.run_dir / name, payload)

    @staticmethod
    def _append_jsonl_at(path: Path, payload: dict[str, Any]) -> None:
        # Serialize before opening the journal so rejected values cannot leave
        # a partial record in front of the following protection event.
        encoded = json.dumps(payload, default=_json_default, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()


# Compatibility for callers/tests created when traces were preview-only.
PreviewTraceWriter = ExecutionTraceWriter
