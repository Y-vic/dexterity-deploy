"""Resolve a recorded episode's initial measured arm and hand pose for reset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from .protection_config import setting


def validate_target(target: dict) -> dict:
    """Check a resolved target again at the motion boundary."""
    if not isinstance(target, dict) or target.get("source") != "episode":
        raise ValueError("invalid episode reset target")
    result = {"source": "episode", "episode": str(target.get("episode", ""))}
    if not result["episode"]:
        raise ValueError("episode reset target has no source")
    for key, width, limit in (
        ("left_joint", 6, setting('ur', 'max_abs_joint_rad')),
        ("right_joint", 6, setting('ur', 'max_abs_joint_rad')),
        ("left_hand_joint", 22, setting('reset', 'max_abs_hand_joint_rad')),
        ("right_hand_joint", 22, setting('reset', 'max_abs_hand_joint_rad')),
    ):
        values = np.asarray(target.get(key), dtype=np.float64)
        if values.shape != (width,) or not np.isfinite(values).all() or np.any(np.abs(values) > limit):
            raise ValueError(f"invalid {key} in episode reset target")
        result[key] = values.tolist()
    return result


def episode_initial_target(root: str, episode: str) -> dict:
    """Return the synchronized first timeline row, refusing incomplete samples."""
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        raise ValueError("reference dataset root must be an absolute path")
    root_path = root_path.resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError("reference dataset root is not a directory")
    relative = Path(episode)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("invalid episode path")
    path = (root_path / relative).resolve(strict=True)
    if not path.is_relative_to(root_path) or path == root_path:
        raise ValueError("episode is outside the reference dataset")
    if not path.is_dir() or not (path / "COMPLETE").is_file() or (path / "ERROR").exists():
        raise ValueError("episode is not complete")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "ur_sharpa_sample.v1" or manifest.get("state") != "complete":
        raise ValueError("unsupported or incomplete episode manifest")
    if not (manifest.get("quality") or {}).get("ok", False):
        raise ValueError("episode quality check did not pass")

    def first_row(name: str, width: int, valid_name: str) -> np.ndarray:
        file = (path / name).resolve(strict=True)
        if not file.is_relative_to(path):
            raise ValueError("episode data file escapes the episode directory")
        with np.load(file, allow_pickle=False) as data:
            q = np.asarray(data["q"])
            valid = np.asarray(data[valid_name])
            if q.ndim != 2 or q.shape[1] != width or q.shape[0] == 0:
                raise ValueError(f"{name} has invalid joint shape")
            if valid.shape != (q.shape[0],) or not bool(valid[0]):
                raise ValueError(f"{name} initial row is not valid")
            row = np.asarray(q[0], dtype=np.float64)
            if not np.isfinite(row).all():
                raise ValueError(f"{name} initial row has non-finite joints")
            return row

    arm = first_row("ur/physical_12.npz", 12, "valid")
    hand = first_row("sharpa/joints_44.npz", 44, "q_valid")
    if (np.max(np.abs(arm)) > setting('ur', 'max_abs_joint_rad')
            or np.max(np.abs(hand)) > setting('reset', 'max_abs_hand_joint_rad')):
        raise ValueError("episode initial joints exceed hardware command limits")
    return validate_target({
        "source": "episode",
        "episode": str(path),
        "left_joint": arm[:6].tolist(),
        "right_joint": arm[6:].tolist(),
        "left_hand_joint": hand[:22].tolist(),
        "right_hand_joint": hand[22:].tolist(),
    })
