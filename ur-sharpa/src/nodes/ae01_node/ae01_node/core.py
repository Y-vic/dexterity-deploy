"""ROS-independent AE01 ROBOT-frame validation and per-hand freshness."""
from __future__ import annotations

import math
import time

from sharpa_interface.sharpa import HANDS, JOINT_NAMES_PER_HAND, NUM_JOINTS_PER_HAND


def robot_joints(positions, names=(), side="left"):
    """SDK ROBOT uses radians and the frozen Wave order; RAW has a different order."""
    values = [float(v) for v in positions]
    if len(values) != NUM_JOINTS_PER_HAND or not all(map(math.isfinite, values)):
        raise ValueError("AE01 ROBOT requires 22 finite joint values in radians")
    if names:
        normalized = [str(n).strip().lower().removeprefix(side + "_") for n in names]
        expected = [n.lower() for n in JOINT_NAMES_PER_HAND]
        # SDK 1.7.5 labels ROBOT output joint_0..joint_21. Its indices use the
        # documented ROBOT/Wave order, which differs from the RAW indices.
        indexed = [f"joint_{i}" for i in range(NUM_JOINTS_PER_HAND)]
        if set(normalized) == set(indexed):
            expected = indexed
        if len(normalized) != len(values) or len(set(normalized)) != len(values) or set(normalized) != set(expected):
            raise ValueError(f"AE01 {side} ROBOT joint names do not match the Wave contract")
        by_name = dict(zip(normalized, values))
        values = [by_name[n] for n in expected]
    return values


class AE01Core:
    def __init__(self, timeout_s=0.25, clock=time.monotonic):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("AE01 data timeout must be positive and finite")
        self.timeout_s, self.clock = float(timeout_s), clock
        self.sides = {}
        self.error = ""
        self._last_packet = 0.0

    def observe(self, packet):
        """Worker and ROS share CLOCK_MONOTONIC; queued packets retain their age."""
        now = self.clock()
        emitted = float(packet["monotonic_s"])
        if not math.isfinite(emitted) or emitted > now + .01 or emitted < self._last_packet:
            raise ValueError("AE01 worker packet timestamp invalid or out of order")
        self._last_packet = emitted
        self.error = str(packet.get("error", ""))
        # Replace the complete snapshot so an offline/missing side cannot retain its target.
        sides = {}
        for side in HANDS:
            row = dict(packet.get("sides", {}).get(side, {}))
            try:
                for field in ("raw_seen_s", "robot_seen_s"):
                    seen = float(row.get(field, 0.0))
                    if not math.isfinite(seen) or seen < 0 or seen > emitted + .01:
                        raise ValueError(f"AE01 {side} invalid {field}")
                    row[field] = seen
                if row.get("q") is not None:
                    row["q"] = robot_joints(row["q"], row.get("names", ()), side)
            except (ValueError, TypeError, OverflowError) as exc:
                row = {"error": str(exc)}
            sides[side] = row
        self.sides = sides

    def valid(self, side, alive=True):
        row = self.sides.get(side, {})
        now = self.clock()
        return bool(alive and not self.error and row.get("online") and row.get("ready")
                    and not row.get("error") and row.get("q") is not None
                    and all(row.get(key, 0) > 0 and 0 <= now-row[key] < self.timeout_s
                            for key in ("raw_seen_s", "robot_seen_s")))

    def command(self, alive=True):
        enabled = [self.valid(side, alive) for side in HANDS]
        if not any(enabled):
            return None
        q = []
        for side, valid in zip(HANDS, enabled):
            q.extend(self.sides[side]["q"] if valid else [0.0]*NUM_JOINTS_PER_HAND)
        return q, *enabled

    def status(self, alive, started=False, error=""):
        now = self.clock()
        result = dict(provider="ae01", connected=False, valid=False, managed_by_ros=True,
                      hardware_started=started, sdk_alive=alive, error=error or self.error)
        errors = []
        for side in HANDS:
            row = self.sides.get(side, {})
            valid = self.valid(side, alive)
            result[side + "_valid"] = valid
            result[side + "_hz"] = row.get("hz", 0.0)
            result[side + "_robot_hz"] = row.get("robot_hz", 0.0)
            result[side + "_sn"] = row.get("sn", "")
            result[side + "_ready"] = bool(row.get("ready"))
            for key in ("raw", "robot"):
                seen = row.get(key + "_seen_s", 0)
                result[f"{side}_{key}_age_s"] = max(0.0, now-seen) if seen else None
                result[f"{side}_{key}_timestamp_ns"] = row.get(key + "_timestamp_ns", 0)
            result[side + "_fault_code"] = row.get("fault_code", 0)
            if not valid:
                reason = row.get("error") or ("等待重定向预热" if row.get("online") and not row.get("ready")
                                              else "无新鲜 RAW/ROBOT 数据")
                errors.append(f"{side}: {reason}")
        result["connected"] = result["valid"] = any(result[s + "_valid"] for s in HANDS)
        result["error"] = result["error"] or "; ".join(errors)
        return result
