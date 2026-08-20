"""Convert PND tactile state and bulk frames to the GR00T MoT wire."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


FINGER_ORDER = (
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb",
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb",
)
IMAGE_SIZE = 64
FORCE_KEY = "observation/tactile_force_10x6"
FORCE_VALID_KEY = "observation/tactile_force_valid_10"
DEFORMATION_KEY = "observation/tactile_deformation_10x64x64"
DEFORMATION_VALID_KEY = "observation/tactile_deformation_valid_10"
ORDER_KEY = "observation/tactile_finger_order"


def build_mot_tactile_request(
    observation: Any,
    tactile_data: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields: dict[str, Any] = {}
    info = {
        "schema": "ws.groot_n17_mot_tactile.v1",
        "force_valid": 0,
        "deformation_valid": 0,
        "bulk_bytes": len(tactile_data),
    }
    if not isinstance(observation, dict):
        info["fallback"] = "observation_not_object"
        return fields, info

    force, force_valid = _force_frame(observation)
    deformation, deformation_valid = _deformation_frame(observation, tactile_data)
    if force is not None:
        fields[FORCE_KEY] = force
        fields[FORCE_VALID_KEY] = force_valid
        info["force_valid"] = int(force_valid.sum())
    if deformation is not None:
        fields[DEFORMATION_KEY] = deformation
        fields[DEFORMATION_VALID_KEY] = deformation_valid
        info["deformation_valid"] = int(deformation_valid.sum())
    if fields:
        fields[ORDER_KEY] = list(FINGER_ORDER)
        info["fallback"] = None
    else:
        info["fallback"] = "no_tactile_payload"
    return fields, info


def _force_frame(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        return None, None
    payload = _unwrap_json(robot_state.get("payload"))
    tactile = payload.get("tactile") if isinstance(payload, dict) else None
    if not isinstance(tactile, dict):
        return None, None
    order = tactile.get("order")
    force = tactile.get("force")
    torque = tactile.get("torque")
    valid = tactile.get("force_valid")
    if not all(isinstance(value, (list, tuple)) for value in (order, force, torque, valid)):
        return None, None
    if not (len(order) == len(force) == len(torque) == len(valid)):
        return None, None

    output = np.zeros((len(FINGER_ORDER), 6), dtype=np.float32)
    output_valid = np.zeros(len(FINGER_ORDER), dtype=bool)
    destination = {name: index for index, name in enumerate(FINGER_ORDER)}
    for source_index, raw_name in enumerate(order):
        target_index = destination.get(str(raw_name))
        if target_index is None:
            continue
        try:
            value = np.asarray(
                [*force[source_index], *torque[source_index]], dtype=np.float32
            )
        except (TypeError, ValueError):
            continue
        is_valid = bool(valid[source_index]) and value.shape == (6,)
        is_valid = is_valid and bool(np.all(np.isfinite(value)))
        if is_valid:
            output[target_index] = value
            output_valid[target_index] = True
    return output, output_valid


def _deformation_frame(
    observation: dict[str, Any], tactile_data: bytes
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    robot_tactile = observation.get("robot_tactile")
    if not isinstance(robot_tactile, dict):
        return None, None
    metadata = _unwrap_json(robot_tactile.get("metadata"))
    entries = metadata.get("entries") if isinstance(metadata, dict) else None
    if not isinstance(entries, list) or not tactile_data:
        return None, None

    output = np.zeros(
        (len(FINGER_ORDER), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8
    )
    output_valid = np.zeros(len(FINGER_ORDER), dtype=bool)
    destination = {name: index for index, name in enumerate(FINGER_ORDER)}
    raw = memoryview(tactile_data)
    for entry in entries:
        if not isinstance(entry, dict) or not bool(entry.get("valid")):
            continue
        name = f"{entry.get('side', '')}_{entry.get('finger', '')}"
        target_index = destination.get(name)
        if target_index is None:
            continue
        try:
            offset = int(entry["offset"])
            length = int(entry["length"])
            height = int(entry["height"])
            width = int(entry["width"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            offset < 0
            or length <= 0
            or height <= 0
            or width <= 0
            or length != height * width
            or offset + length > len(raw)
        ):
            continue
        image = np.frombuffer(raw[offset : offset + length], dtype=np.uint8).reshape(
            height, width
        )
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
            image = np.asarray(
                Image.fromarray(image, mode="L").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
        output[target_index] = image
        output_valid[target_index] = True
    return output, output_valid
def _unwrap_json(value: Any) -> Any:
    if isinstance(value, dict) and value.get("valid") is True and "json" in value:
        return value["json"]
    return value
