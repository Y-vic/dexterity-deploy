"""Fingertip force return over the existing AE01 SDK connection."""
from __future__ import annotations

import json
import math
import os
import threading
import time

from sharpa_interface.sharpa import FINGERS, HANDS, TACTILE_FINGERS

FEEDBACK_HZ = 30.0  # Same force update cadence as the vendor WaveService.


def fingertip_forces(values, valid):
    """WrenchFrame: pinky->thumb, six axes. AE01: thumb->pinky, Fx/Fy/Fz."""
    if len(values) != 30 or len(valid) != 5:
        raise ValueError("tactile wrench requires 5 x 6 values and 5 validity flags")
    result = []
    for finger in FINGERS:
        index = TACTILE_FINGERS.index(finger)
        xyz = [float(v) for v in values[index*6:index*6+3]]
        result.append(xyz if valid[index] and all(map(math.isfinite, xyz)) else [0.0]*3)
    return result


def force_payload(side, forces, *, source=True):
    payload = dict(key="set_3d_force", device_side=side.upper(),
                   **{f"force_{axis}": [row[i] for row in forces] for i, axis in enumerate("xyz")})
    if source:
        payload["source"] = "wave"
    return payload


class FeedbackInput:
    """Keep original source and receive deadlines when sending repeated IPC updates."""
    def __init__(self, max_age_s, future_tolerance_s, clock=time.monotonic, wall_ns=time.time_ns):
        if not math.isfinite(max_age_s) or max_age_s <= 0 or not math.isfinite(future_tolerance_s) or future_tolerance_s < 0:
            raise ValueError("invalid tactile freshness limits")
        self.max_age_s, self.future_tolerance_s = max_age_s, future_tolerance_s
        self.clock, self.wall_ns = clock, wall_ns
        self.stamp = self.seen = self.deadline = 0
        self.forces = {}

    def observe(self, message):
        stamp, now = int(message.timestamp_ns), self.clock()
        age = (self.wall_ns()-stamp)/1e9
        if stamp <= 0 or stamp < self.stamp or not -self.future_tolerance_s <= age < self.max_age_s:
            self.forces = {}
            return
        if stamp == self.stamp:
            # Invalid flags can revoke an existing finger immediately, even in
            # a repeated source frame, while its original deadline is retained.
            for side, forces in self.forces.items():
                valid = getattr(message, side+"_valid")
                if len(valid) != 5:
                    raise ValueError("tactile wrench requires 5 validity flags")
                for index, finger in enumerate(FINGERS):
                    if not valid[TACTILE_FINGERS.index(finger)]:
                        forces[index] = [0.0]*3
            return  # A replay of the same source frame cannot extend its lifetime.
        self.stamp, self.seen = stamp, now
        self.deadline = min(now+self.max_age_s, now+self.max_age_s-age)
        self.forces = {side: fingertip_forces(getattr(message, side), getattr(message, side+"_valid"))
                       for side in HANDS}

    def packet(self, control, enabled):
        now = self.clock()
        allowed = False
        control_deadline = now
        if control is not None:
            seen, state = control
            control_deadline = seen+.5
            allowed = (now < control_deadline and state.sharpa_state == 1
                       and state.active_source in (2, 3) and not state.execution_paused
                       and not state.execution_dry_run)
        deadline = min(self.deadline, control_deadline)
        return dict(monotonic_s=now, expires_s=max(now, deadline), timestamp_ns=self.stamp,
                    sides={side: dict(enabled=bool(allowed and now < deadline and enabled.get(side)),
                        forces=self.forces.get(side, [[0.0]*3 for _ in FINGERS])) for side in HANDS})


class FeedbackWriter:
    """One bounded atomic pipe write; SDK initialization must never block ROS callbacks."""
    def __init__(self, stream):
        self.fd = stream.fileno()
        os.set_blocking(self.fd, False)

    def send(self, packet):
        data = (json.dumps(packet, separators=(",", ":"), allow_nan=False)+"\n").encode()
        if len(data) > 4096:
            raise ValueError("AE01 feedback packet exceeds pipe atomic-write limit")
        try:
            return os.write(self.fd, data) == len(data)
        except (BlockingIOError, BrokenPipeError, OSError):
            return False


class ForceRelay:
    """SDK-side force watchdog, independent of the acquisition/retarget polling loop."""
    def __init__(self, api, max_age_s, future_tolerance_s, clock=time.monotonic, wall_ns=time.time_ns):
        if not math.isfinite(max_age_s) or max_age_s <= 0 or not math.isfinite(future_tolerance_s) or future_tolerance_s < 0:
            raise ValueError("invalid tactile freshness limits")
        self.api, self.max_age_s, self.future_tolerance_s = api, max_age_s, future_tolerance_s
        self.clock, self.wall_ns = clock, wall_ns
        self.lock = threading.RLock()
        self.devices, self.states, self.active, self.errors = {}, {}, {}, {}
        self.pending_clear = set()
        self.latest = None
        self.input_fresh = False
        self.stop = threading.Event()
        self.thread = None

    def receive(self, packet):
        with self.lock:
            self.latest = packet

    def read_stream(self, stream):
        try:
            while not self.stop.is_set():
                line = stream.readline(4097)
                if not line:
                    break
                try:
                    packet = json.loads(line) if len(line) <= 4096 else None
                except (ValueError, TypeError):
                    packet = None
                self.receive(packet)
        finally:
            self.receive(None)

    def update(self, states, devices):
        with self.lock:
            for side, device in devices.items():
                if self.devices.get(side) is not device:
                    self.detach(side)
                    self.devices[side] = device
                    self.pending_clear.add(side)
            self.states = states

    def _task(self, side, payload):
        code, _ = self.devices[side].set_task(json.dumps(payload, allow_nan=False))
        if code != self.api.ErrorCode.SUCCESS:
            raise RuntimeError(f"{side} {payload['key']}: {code}")

    def _clear(self, side):
        self.pending_clear.add(side)
        try:
            self._task(side, dict(key="clear_3d_force", device_side=side.upper()))
            self.pending_clear.discard(side)
            self.active[side] = False
            self.errors.pop(side, None)
        except Exception as exc:
            self.errors[side] = str(exc)
            # Vendor fallback: force zeros even when clear_3d_force is refused.
            try:
                self._task(side, force_payload(side, [[0.0]*3 for _ in FINGERS], source=False))
                self.active[side] = False
            except Exception as zero_exc:
                self.errors[side] += "; " + str(zero_exc)

    def detach(self, side):
        with self.lock:
            if side in self.devices:
                self._clear(side)
            self.devices.pop(side, None)
            self.active.pop(side, None)
            self.pending_clear.discard(side)

    def _fresh(self, packet, now):
        if not isinstance(packet, dict):
            return False
        try:
            age = (self.wall_ns()-int(packet["timestamp_ns"]))/1e9
            emitted, expires = float(packet["monotonic_s"]), float(packet["expires_s"])
            return (all(map(math.isfinite, (emitted, expires))) and 0 <= now-emitted < self.max_age_s
                    and now < expires <= emitted+self.max_age_s+.001
                    and -self.future_tolerance_s <= age < self.max_age_s)
        except (ValueError, TypeError, KeyError, OverflowError):
            return False

    def tick(self):
        with self.lock:
            now, packet = self.clock(), self.latest
            self.input_fresh = self._fresh(packet, now)
            for side in list(self.devices):
                row = self.states.get(side, {})
                ready = (row.get("online") and row.get("ready") and not row.get("error")
                         and all(row.get(key, 0) > 0 and 0 <= now-row[key] < self.max_age_s
                                 for key in ("raw_seen_s", "robot_seen_s")))
                sides = packet.get('sides', {}) if self.input_fresh else {}
                request = sides.get(side, {}) if isinstance(sides, dict) else {}
                request = request if isinstance(request, dict) else {}
                forces = request.get("forces", [])
                valid = (request.get("enabled") is True and ready and isinstance(forces, list) and len(forces) == 5
                         and all(isinstance(v, list) and len(v) == 3
                                 and all(type(x) in (float, int) and math.isfinite(x) for x in v) for v in forces))
                nonzero = valid and any(value != 0 for finger in forces for value in finger)
                if side in self.pending_clear or (not nonzero and self.active.get(side)):
                    self._clear(side)
                if nonzero and side not in self.pending_clear:
                    try:
                        self._task(side, force_payload(side, forces))
                        self.active[side] = True
                        self.errors.pop(side, None)
                    except Exception as exc:
                        error = str(exc)
                        self._clear(side)
                        self.errors[side] = error + ("; " + self.errors[side] if side in self.errors else "")

    def status(self):
        with self.lock:
            return dict(input_fresh=self.input_fresh, rate_hz=FEEDBACK_HZ,
                        **{side+"_active": bool(self.active.get(side)) for side in HANDS},
                        error="; ".join(self.errors.values()))

    def start(self, stream):
        def run():
            while not self.stop.is_set():
                started = self.clock()
                try:
                    self.tick()
                except Exception as exc:
                    # A malformed IPC packet must not kill the watchdog with
                    # the last force still latched in the SDK vibration engine.
                    with self.lock:
                        self.latest = None
                        for side in list(self.devices):
                            self._clear(side)
                        self.errors['watchdog'] = str(exc)
                self.stop.wait(max(0., 1/FEEDBACK_HZ-(self.clock()-started)))
        threading.Thread(target=self.read_stream, args=(stream,), daemon=True, name="ae01-force-input").start()
        self.thread = threading.Thread(target=run, daemon=True, name="ae01-force-watchdog")
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        for side in list(self.devices):
            self.detach(side)
