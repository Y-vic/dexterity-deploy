"""Validated recordings and source-time resampling for replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_checksums(path: Path) -> None:
    checksum_path = path / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {checksum_path}") from exc
    for line in lines:
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line: {line!r}") from exc
        target = path / relative
        if not target.is_file():
            raise ValueError(f"checksum target is missing: {relative}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"checksum mismatch: {relative}")


def _verify_joint_quality(path: Path, joint_source: str) -> None:
    """Require the validation report entries used by joint-only replay."""
    validation = _load_json(path / "validation.json")
    if not bool(validation.get("available", False)):
        raise ValueError("sample validation report is unavailable")
    items = validation.get("items")
    if not isinstance(items, list):
        raise ValueError("sample validation items must be a list")
    by_key = {
        str(item.get("key")): item
        for item in items
        if isinstance(item, dict) and item.get("key")
    }
    required = ("arm_state", "hand_state" if joint_source == "measured" else "hand_action")
    for key in required:
        item = by_key.get(key)
        if item is None:
            raise ValueError(f"sample validation is missing required item: {key}")
        if not bool(item.get("ok", False)):
            details = ", ".join(str(value) for value in item.get("details") or [])
            suffix = f": {details}" if details else ""
            raise ValueError(f"sample validation failed for {key}{suffix}")


def _hold_invalid_rows(values: np.ndarray, valid: np.ndarray, label: str) -> tuple[np.ndarray, list[int]]:
    if valid.shape != (values.shape[0],):
        raise ValueError(f"{label} validity must have shape ({values.shape[0]},)")
    invalid_rows = np.flatnonzero(~valid).astype(int).tolist()
    if not invalid_rows:
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains non-finite values in valid rows")
        return np.ascontiguousarray(values, dtype=np.float32), []
    if not bool(valid[0]):
        raise ValueError(f"{label} first row is invalid and cannot be held")
    repaired = np.asarray(values, dtype=np.float32).copy()
    for row in range(repaired.shape[0]):
        if valid[row]:
            if not np.isfinite(repaired[row]).all():
                raise ValueError(f"{label} row {row} is marked valid but is non-finite")
        else:
            repaired[row] = repaired[row - 1]
    return np.ascontiguousarray(repaired), invalid_rows


def _resample_source_rows(
    values: np.ndarray,
    valid: np.ndarray,
    source_timestamp_ns: np.ndarray,
    target_timestamp_ns: np.ndarray,
    label: str,
) -> tuple[np.ndarray, list[int]]:
    _, invalid_rows = _hold_invalid_rows(values, valid, label)
    if source_timestamp_ns.shape != (values.shape[0],):
        raise ValueError(
            f"{label} source timestamps must have shape ({values.shape[0]},)"
        )
    if target_timestamp_ns.shape != (values.shape[0],):
        raise ValueError(
            f"{label} target timestamps must have shape ({values.shape[0]},)"
        )
    if np.any(np.diff(target_timestamp_ns) <= 0):
        raise ValueError(f"{label} target timestamps must be strictly increasing")

    valid_indices = np.flatnonzero(valid)
    source_times = source_timestamp_ns[valid_indices]
    if np.any(source_times <= 0):
        raise ValueError(f"{label} source timestamps must be positive")
    if np.any(np.diff(source_times) < 0):
        raise ValueError(f"{label} source timestamps must be nondecreasing")

    unique_indices: list[int] = []
    for index in valid_indices:
        index = int(index)
        is_new_timestamp = (
            not unique_indices
            or source_timestamp_ns[index] != source_timestamp_ns[unique_indices[-1]]
        )
        if is_new_timestamp:
            unique_indices.append(index)
        elif not np.array_equal(values[index], values[unique_indices[-1]]):
            raise ValueError(
                f"{label} has conflicting rows for one source timestamp"
            )

    source_indices = np.asarray(unique_indices, dtype=np.int64)
    origin_ns = min(
        int(target_timestamp_ns[0]), int(source_timestamp_ns[source_indices[0]])
    )
    target_seconds = (target_timestamp_ns - origin_ns).astype(np.float64) / 1.0e9
    source_seconds = (
        source_timestamp_ns[source_indices] - origin_ns
    ).astype(np.float64) / 1.0e9
    resampled = np.empty(values.shape, dtype=np.float32)
    for joint in range(values.shape[1]):
        resampled[:, joint] = np.interp(
            target_seconds,
            source_seconds,
            values[source_indices, joint],
        )
    return np.ascontiguousarray(resampled), invalid_rows


@dataclass(frozen=True)
class ReplaySample:
    path: Path
    name: str
    elapsed_ns: np.ndarray
    arm_joint: np.ndarray
    hand_joint: np.ndarray
    recorded_rate_hz: float
    repaired_hand_rows: tuple[int, ...]
    joint_source: str

    @property
    def row_count(self) -> int:
        return int(self.elapsed_ns.size)

    @property
    def duration_s(self) -> float:
        return float(self.elapsed_ns[-1]) / 1.0e9 if self.row_count else 0.0

    @classmethod
    def load(cls, sample_dir: str | Path, joint_source: str = "measured") -> "ReplaySample":
        if not str(sample_dir).strip():
            raise ValueError("select a Replay sample directory before loading")
        path = Path(sample_dir).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"sample directory not found: {path}")
        if not (path / "COMPLETE").is_file():
            raise ValueError(f"sample is not finalized (missing COMPLETE): {path}")
        manifest = _load_json(path / "manifest.json")
        schema = _load_json(path / "schema.json")
        if manifest.get("schema") != "ur_sharpa_sample.v1":
            raise ValueError(f"unsupported manifest schema: {manifest.get('schema')!r}")
        if manifest.get("state") != "complete":
            raise ValueError(f"sample manifest state is not complete: {manifest.get('state')!r}")
        if not bool((manifest.get("quality") or {}).get("ok", False)):
            details = []
            try:
                report = _load_json(path / "validation.json")
                failed = [item for item in report.get("items", []) if not item.get("ok")]
                # Show trajectory faults first; a success label is not capture quality.
                failed.sort(key=lambda item: item.get("key") not in {"arm_state", "hand_state", "hand_action"})
                for item in failed[:3]:
                    details.append(f"{item.get('label', item.get('key'))}: {item.get('hz', '?')} Hz，"
                                   f"缺失 {item.get('missing', '?')} 帧")
            except (OSError, ValueError, TypeError):
                pass
            reason = "sample manifest quality.ok is false"
            raise ValueError(reason + ("；" + "；".join(details) if details else ""))
        if schema.get("schema") != "ur_sharpa_sample.v1":
            raise ValueError(f"unsupported sample schema: {schema.get('schema')!r}")
        _verify_checksums(path)

        source = str(joint_source).strip().lower()
        if source not in {"measured", "commanded"}:
            raise ValueError("joint_source must be measured or commanded")
        _verify_joint_quality(path, source)
        try:
            with np.load(path / "timeline" / "clock.npz", allow_pickle=False) as clock:
                time_unix_ns = np.asarray(clock["time_unix_ns"], dtype=np.int64)
                elapsed_ns = np.asarray(clock["elapsed_ns"], dtype=np.int64)
                rate_hz = float(np.asarray(clock["sample_rate_hz"]).item())
            with np.load(path / "ur" / "physical_12.npz", allow_pickle=False) as arm:
                arm_joint = np.asarray(arm["q"], dtype=np.float32)
                arm_valid = np.asarray(arm["valid"], dtype=bool)
            with np.load(path / "sharpa" / "joints_44.npz", allow_pickle=False) as hand:
                hand_key = "q" if source == "measured" else "q_cmd"
                valid_key = "q_valid" if source == "measured" else "q_cmd_valid"
                timestamp_key = (
                    "q_timestamp_ns" if source == "measured" else "q_cmd_timestamp_ns"
                )
                hand_joint = np.asarray(hand[hand_key], dtype=np.float32)
                hand_valid = np.asarray(hand[valid_key], dtype=bool)
                hand_timestamp_ns = np.asarray(hand[timestamp_key], dtype=np.int64)
                hand_source = (
                    np.asarray(hand["q_cmd_source"], dtype=np.uint8)
                    if source == "commanded"
                    else None
                )
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            raise ValueError(f"failed to load replay arrays: {exc}") from exc

        if elapsed_ns.ndim != 1 or elapsed_ns.size < 2:
            raise ValueError(f"timeline elapsed_ns must have shape (T>=2,), got {elapsed_ns.shape}")
        rows = int(elapsed_ns.size)
        if time_unix_ns.shape != (rows,):
            raise ValueError(f"timeline time_unix_ns must have shape ({rows},)")
        if arm_joint.shape != (rows, 12):
            raise ValueError(f"arm q must have shape ({rows}, 12), got {arm_joint.shape}")
        if hand_joint.shape != (rows, 44):
            raise ValueError(f"hand joints must have shape ({rows}, 44), got {hand_joint.shape}")
        if source == "commanded":
            if hand_source is None or hand_source.shape != (rows,):
                raise ValueError(f"hand q_cmd_source must have shape ({rows},)")
            if np.any(hand_source[hand_valid] != 2):
                raise ValueError("commanded hand trajectory contains a non-teleop source")
        if arm_valid.shape != (rows,) or not np.all(arm_valid):
            invalid = np.flatnonzero(~arm_valid).astype(int).tolist()
            raise ValueError(f"arm trajectory contains invalid rows: {invalid[:20]}")
        if not np.isfinite(arm_joint).all():
            raise ValueError("arm trajectory contains non-finite values")
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("recorded sample rate must be finite and positive")
        normalized_elapsed = elapsed_ns - elapsed_ns[0]
        if normalized_elapsed[0] != 0 or np.any(np.diff(normalized_elapsed) <= 0):
            raise ValueError("timeline elapsed_ns must be strictly increasing")
        repaired_hand, repaired_rows = _resample_source_rows(
            hand_joint,
            hand_valid,
            hand_timestamp_ns,
            time_unix_ns,
            f"hand {source} trajectory",
        )
        arm_step = float(np.max(np.abs(np.diff(arm_joint, axis=0))))
        hand_step = float(np.max(np.abs(np.diff(repaired_hand, axis=0))))
        if arm_step > 0.25:
            raise ValueError(f"arm trajectory step {arm_step:.4f} rad exceeds 0.25 rad")
        if hand_step > 0.4:
            raise ValueError(f"hand trajectory step {hand_step:.4f} rad exceeds 0.4 rad")
        return cls(
            path=path,
            name=path.name,
            elapsed_ns=np.ascontiguousarray(normalized_elapsed),
            arm_joint=np.ascontiguousarray(arm_joint),
            hand_joint=repaired_hand,
            recorded_rate_hz=rate_hz,
            repaired_hand_rows=tuple(repaired_rows),
            joint_source=source,
        )
