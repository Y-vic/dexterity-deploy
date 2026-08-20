"""Shared helpers for bias files and URDF limits."""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from typing import Any

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from bias_node.body_joints import (
    ADAM_COMMAND_JOINTS_19,
    DEFAULT_JOINT_VALUES,
    UPPER_BODY_EDITABLE_JOINTS,
    canonical_body_name,
)


DEFAULT_BIAS_PATH = "/home/pnd-humanoid/.adam/joint/bias_joints_set_with_init.json"


def default_bias_path() -> str:
    return os.path.expanduser(os.environ.get("BIAS_NODE_PATH", DEFAULT_BIAS_PATH))


def default_urdf_path() -> str:
    try:
        share = get_package_share_directory("adam_sharpa_description")
    except PackageNotFoundError:
        return ""
    return os.path.join(share, "urdf", "adam_pro_sharpa", "adam_pro_sharpa.urdf")


def body_display_name(joint_name: str) -> str:
    return joint_name.removeprefix("dof_pos/")


def load_joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    if not urdf_path or not os.path.exists(urdf_path):
        return {}

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if not name:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        except (KeyError, TypeError, ValueError):
            continue
        canonical = canonical_body_name(name)
        if canonical in ADAM_COMMAND_JOINTS_19 and math.isfinite(lower) and math.isfinite(upper):
            limits[canonical] = (lower, upper)
    return limits


def normalize_joint_set_map(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}

    if isinstance(raw, dict) and isinstance(raw.get("joints"), dict):
        raw = raw["joints"]
    elif isinstance(raw, dict) and isinstance(raw.get("positions"), dict):
        raw = raw["positions"]
    elif (
        isinstance(raw, dict)
        and isinstance(raw.get("names"), list)
        and isinstance(raw.get("positions"), list)
    ):
        if len(raw["names"]) != len(raw["positions"]):
            raise ValueError("joint names and positions length mismatch")
        raw = dict(zip(raw["names"], raw["positions"]))

    if not isinstance(raw, dict):
        raise ValueError("joint set must be a joint map or names/positions payload")

    joint_set: dict[str, float] = {}
    for name, value in raw.items():
        canonical = canonical_body_name(str(name))
        if canonical not in ADAM_COMMAND_JOINTS_19:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"non-finite position for {canonical}: {value!r}") from None
        if not math.isfinite(number):
            raise ValueError(f"non-finite position for {canonical}: {value!r}")
        joint_set[canonical] = number
    return joint_set


def load_bias_joint_sets_file(path: str) -> tuple[dict[str, float], dict[str, float]]:
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(expanded)
    with open(expanded, "r", encoding="utf-8") as infp:
        raw = json.load(infp)

    if not isinstance(raw, dict):
        raise ValueError("bias file must be a JSON object")
    if "bias_init" not in raw:
        raise ValueError("bias file must contain explicit bias_init")
    if "bias" not in raw and "joints" not in raw:
        raise ValueError("bias file must contain explicit bias or joints")

    bias = normalize_joint_set_map(raw["bias"] if "bias" in raw else raw["joints"])
    bias_init = normalize_joint_set_map(raw["bias_init"])
    return bias_init, bias


def write_bias_joint_sets_file(
    path: str,
    bias_init_positions: dict[str, float],
    bias_positions: dict[str, float],
    *,
    source: str,
) -> None:
    expanded = os.path.expanduser(path)
    directory = os.path.dirname(expanded)
    if directory:
        os.makedirs(directory, exist_ok=True)

    missing_bias_init = [
        name for name in UPPER_BODY_EDITABLE_JOINTS if name not in bias_init_positions
    ]
    missing_bias = [
        name for name in UPPER_BODY_EDITABLE_JOINTS if name not in bias_positions
    ]
    if missing_bias_init or missing_bias:
        raise ValueError(
            "cannot write incomplete bias file: "
            f"missing bias_init={missing_bias_init}, missing bias={missing_bias}"
        )

    ordered_bias_init = {
        name: float(bias_init_positions[name])
        for name in UPPER_BODY_EDITABLE_JOINTS
    }
    ordered_bias = {
        name: float(bias_positions[name])
        for name in UPPER_BODY_EDITABLE_JOINTS
    }
    for label, positions in (
        ("bias_init", ordered_bias_init),
        ("bias", ordered_bias),
    ):
        non_finite = [
            name for name, value in positions.items() if not math.isfinite(value)
        ]
        if non_finite:
            raise ValueError(f"cannot write non-finite {label} positions: {non_finite}")
    payload = {
        "version": 2,
        "kind": "bias",
        "source": source,
        "updated_at": time.time(),
        "names": list(UPPER_BODY_EDITABLE_JOINTS),
        "positions": [ordered_bias[name] for name in UPPER_BODY_EDITABLE_JOINTS],
        "bias_init": ordered_bias_init,
        "joints": ordered_bias,
    }
    temp_path = f"{expanded}.tmp"
    with open(temp_path, "w", encoding="utf-8") as outfp:
        json.dump(payload, outfp, indent=2, sort_keys=False)
        outfp.write("\n")
    os.replace(temp_path, expanded)


def complete_body_positions(partial: dict[str, float]) -> dict[str, float]:
    return {
        name: float(partial.get(name, DEFAULT_JOINT_VALUES[name]))
        for name in ADAM_COMMAND_JOINTS_19
    }


def validate_within_limits(
    positions: dict[str, float],
    limits: dict[str, tuple[float, float]],
    *,
    tolerance: float = 1e-9,
) -> None:
    for name, value in positions.items():
        limit = limits.get(name)
        if limit is None:
            continue
        lower, upper = limit
        if value < lower - tolerance or value > upper + tolerance:
            raise ValueError(
                f"{name}={value:.6f} outside URDF range [{lower:.6f}, {upper:.6f}]"
            )
