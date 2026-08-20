from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


MODE_DAMPING = "damping"
MODE_ZERO = "zero"
MODE_TELEOP = "teleop"
MODE_UNSET_TELEOP = "unset_teleop"
VALID_MODES = {MODE_DAMPING, MODE_ZERO, MODE_TELEOP, MODE_UNSET_TELEOP}
SHARPA_ACTIVE_STATES = {"d_sharpa", "t_init_sharpa", "t_adam_sharpa"}
SHARPA_ZERO_STATES = {"t_init", "t_adam"}


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
}


@dataclass(frozen=True)
class ControlStatus:
    mode: str
    teleop_state: str
    sharpa_active: bool
    known: bool
    payload: dict[str, Any] | None


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
# Sharpa tactile SDK channel indices are pinky-first; ROS topic names remain thumb-first.
TACTILE_FINGER_NAMES_BY_CHANNEL = ("pinky", "ring", "middle", "index", "thumb")


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


def normalize_mode(value: str) -> str | None:
    return MODE_ALIASES.get(normalize_status_value(value))


def normalize_status_value(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _extract_status_value(payload: dict[str, Any]) -> str | None:
    for key in ("state", "mode"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def parse_control_status(data: str) -> ControlStatus:
    raw = data.strip()
    payload: dict[str, Any] | None = None
    status_value: str | None = raw
    if raw:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(decoded, dict):
                payload = decoded
                status_value = _extract_status_value(decoded)
            elif isinstance(decoded, str):
                status_value = decoded
            else:
                status_value = None
    else:
        status_value = MODE_ZERO

    if status_value is None:
        return ControlStatus(
            mode=MODE_DAMPING,
            teleop_state="unknown",
            sharpa_active=False,
            known=False,
            payload=payload,
        )

    teleop_state = normalize_status_value(status_value)
    direct_mode = normalize_mode(teleop_state)
    if teleop_state in SHARPA_ACTIVE_STATES:
        return ControlStatus(MODE_TELEOP, teleop_state, True, True, payload)
    if direct_mode == MODE_TELEOP:
        return ControlStatus(MODE_TELEOP, MODE_TELEOP, True, True, payload)
    if direct_mode in {MODE_DAMPING, MODE_ZERO, MODE_UNSET_TELEOP}:
        return ControlStatus(direct_mode, direct_mode, False, True, payload)
    if teleop_state in SHARPA_ZERO_STATES:
        return ControlStatus(MODE_ZERO, teleop_state, False, True, payload)
    return ControlStatus(MODE_DAMPING, teleop_state, False, False, payload)


def parse_status_mode(data: str) -> tuple[str | None, dict[str, Any] | None]:
    status = parse_control_status(data)
    return (status.mode if status.known else None), status.payload


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
