"""Small, dependency-free helpers for correlated Deploy execution events."""

from __future__ import annotations

import json
import time
from typing import Any


EVENT_SCHEMA = "sharpa.deploy.execution_event.v1"


def execution_event_json(node: str, event: str, **fields: Any) -> str:
    """Return one compact event that Actor can append to the active journal."""
    payload = {
        "schema": EVENT_SCHEMA,
        "node": str(node),
        "event": str(event),
        "timestamp_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
