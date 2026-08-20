from __future__ import annotations

import json
import time
from array import array
from typing import Any

from ws_msgs.msg import Status


def now_ns() -> int:
    return time.time_ns()


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json_dumps(payload).encode("utf-8")


def json_or_raw(data: str) -> dict[str, Any]:
    if not data:
        return {"valid": False, "json": None, "raw": ""}
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError:
        return {"valid": False, "json": None, "raw": data}
    return {"valid": True, "json": decoded, "raw": data}


def header_stamp_ns(msg: Any) -> int | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def set_header(msg: Any, node_name: str, clock: Any) -> None:
    msg.header.stamp = clock.now().to_msg()
    msg.header.frame_id = node_name


def make_status(
    clock: Any,
    node_name: str,
    ok: bool,
    payload: dict[str, Any],
) -> Status:
    msg = Status()
    set_header(msg, node_name, clock)
    msg.node = node_name
    msg.ok = bool(ok)
    msg.payload_json = json_dumps(payload)
    return msg


def age_ms(stamp: float | None, now: float) -> float | None:
    if stamp is None:
        return None
    return round((now - stamp) * 1000.0, 1)


def uint8_array(data: Any | None = None) -> array:
    output = array("B")
    if data is None:
        return output
    output.extend(data)
    return output
