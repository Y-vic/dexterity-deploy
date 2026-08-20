from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


MODE_DAMPING = "damping"
MODE_ZERO = "zero"
MODE_TELEOP = "teleop"
MODE_UNSET_TELEOP = "unset_teleop"
STATE_DAMPING = MODE_DAMPING
STATE_D_SHARPA = "d_sharpa"
STATE_T_INIT = "t_init"
STATE_T_INIT_SHARPA = "t_init_sharpa"
STATE_T_ADAM = "t_adam"
STATE_T_ADAM_SHARPA = "t_adam_sharpa"
VALID_STATES = {
    STATE_DAMPING,
    STATE_D_SHARPA,
    STATE_T_INIT,
    STATE_T_INIT_SHARPA,
    STATE_T_ADAM,
    STATE_T_ADAM_SHARPA,
}
VALID_MODES = {
    MODE_DAMPING,
    MODE_ZERO,
    MODE_TELEOP,
    MODE_UNSET_TELEOP,
    *VALID_STATES,
}


MODE_ALIASES = {
    "": MODE_ZERO,
    "damping": MODE_DAMPING,
    "idle": MODE_DAMPING,
    "float": MODE_DAMPING,
    "floating": MODE_DAMPING,
    "no_command": MODE_DAMPING,
    "zero": MODE_ZERO,
    "standup": MODE_ZERO,
    "stand_up": MODE_ZERO,
    "teleop": MODE_TELEOP,
    "teleop_on": MODE_TELEOP,
    "on": MODE_TELEOP,
    "unset_teleop": MODE_UNSET_TELEOP,
    "teleop_off": MODE_UNSET_TELEOP,
    "off": MODE_UNSET_TELEOP,
    "d_sharpa": STATE_D_SHARPA,
    "t_init": STATE_T_INIT,
    "t_init_sharpa": STATE_T_INIT_SHARPA,
    "t_adam": STATE_T_ADAM,
    "t_adam_sharpa": STATE_T_ADAM_SHARPA,
}


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
    cwd = Path(os.getcwd()).resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "src" / "sharpa_control" / "sdk").is_dir():
            return str(candidate)
    return "/home/pnd-humanoid/Deploy"


def sdk_path(*parts: str) -> str:
    return os.path.join(repo_root_from_cwd(), "src", "sharpa_control", "sdk", *parts)


def normalize_mode(value: str) -> str | None:
    return MODE_ALIASES.get(normalize_status_value(value))


def normalize_status_value(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def parse_status_mode(data: str) -> tuple[str | None, dict[str, Any] | None]:
    raw = data.strip()
    if not raw:
        return MODE_ZERO, None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return normalize_mode(raw), None
    if isinstance(decoded, str):
        return normalize_mode(decoded), None
    if not isinstance(decoded, dict):
        return None, None
    payload = decoded
    status_value = payload.get("state", payload.get("mode", ""))
    mode = normalize_mode(str(status_value))
    return mode, payload


def transient_local_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
