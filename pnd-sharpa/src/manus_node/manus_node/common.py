from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from std_msgs.msg import String


LEFT_JOINT_NAMES = [
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]
RIGHT_JOINT_NAMES = [name.replace("left_", "right_", 1) for name in LEFT_JOINT_NAMES]
SHARPA_JOINT_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def repo_root_from_cwd() -> str:
    env_root = os.environ.get("PND_WORKSPACE_DIR", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "external" / "sharpa_control" / "sdk").is_dir():
            return str(candidate)
    cwd = Path(os.getcwd()).resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "external" / "sharpa_control" / "sdk").is_dir():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find external/sharpa_control/sdk from PND_WORKSPACE_DIR or cwd"
    )


def sdk_path(*parts: str) -> str:
    return os.path.join(
        repo_root_from_cwd(), "external", "sharpa_control", "sdk", *parts
    )


def json_msg(payload: dict[str, Any]) -> String:
    msg = String()
    msg.data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return msg


def age_ms(timestamp: float | None, now: float) -> float | None:
    if timestamp is None:
        return None
    return round((now - timestamp) * 1000.0, 1)


def fixed_float_vector(values: Any, length: int) -> list[float]:
    output = [0.0] * length
    try:
        iterator = list(values)[:length]
    except TypeError:
        iterator = []
    for idx, value in enumerate(iterator):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        output[idx] = number if math.isfinite(number) else 0.0
    return output


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
