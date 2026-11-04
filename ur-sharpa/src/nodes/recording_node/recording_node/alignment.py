"""Bounded, recording-only timestamp alignment; never synthesize source frames."""
from collections import Counter, deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SourceFrame:
    timestamp_ns: int
    received_ns: int
    value: object


class TimestampAligner:
    """Match each source at most once to a fixed 30 Hz (or configured) grid.

    Emit as soon as all sources have an unused frame in the target window.
    Missing sources have a hard deadline, including at stop. SharpA modalities
    must share the hand-state timestamp, so commands/HIL flags stay paired with q.
    The caller supplies monotonic-derived wall time and owns synchronization.
    """

    def __init__(self, names, frequency, wait_frames=2, capacity=16):
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("alignment frequency must be positive and finite")
        if not 0 < wait_frames <= 2 or capacity < 4:
            raise ValueError("alignment requires <=2 frames of waiting and >=4 slots")
        self.names = tuple(names)
        self.frequency = float(frequency)
        self.wait_ns = round(wait_frames * 1e9 / frequency)
        self.skew_ns = round(.6e9 / frequency)
        self.camera_skew_ns = round(1e9 / frequency)
        self.capacity = capacity
        self.queues = {name: deque() for name in self.names}
        self.last_seen = dict.fromkeys(self.names, 0)
        self.last_used = dict.fromkeys(self.names, 0)
        self.rejected = Counter()
        self.overflows = Counter()
        self.missing = Counter()
        self.origin_ns = None
        self.index = 0
        self.max_emit_delay_ms = 0.

    def start(self, origin_ns):
        self.origin_ns = int(origin_ns)
        self.index = 0
        self.last_used = dict.fromkeys(self.names, 0)
        self.rejected.clear()
        self.overflows.clear()
        self.missing.clear()
        self.max_emit_delay_ms = 0.
        for name, values in self.queues.items():
            while values and values[0].timestamp_ns < origin_ns - self._skew_ns(name):
                values.popleft()

    def _skew_ns(self, name):
        # Independent camera clocks drift relative to the state clock. A
        # one-period window permits ordered, unique assignment without rate
        # conversion; original image timestamps remain authoritative.
        return self.camera_skew_ns if name in ("ego", "left_wrist", "right_wrist") else self.skew_ns

    def push(self, name, timestamp_ns, received_ns, value):
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= self.last_seen[name]:
            # Repeated cached CameraSets are expected; backwards sources aren't.
            if timestamp_ns < self.last_seen[name]:
                self.rejected[name] += 1
            return False
        self.last_seen[name] = timestamp_ns
        values = self.queues[name]
        if len(values) == self.capacity:
            discarded = values.popleft()
            if (self.origin_ns is not None
                    and discarded.timestamp_ns >= self.next_target_ns - self._skew_ns(name)
                    and discarded.timestamp_ns > self.last_used[name]):
                self.overflows[name] += 1
        values.append(SourceFrame(timestamp_ns, int(received_ns), value))
        return True

    @property
    def next_target_ns(self):
        return self.origin_ns + round(self.index * 1e9 / self.frequency)

    def pop_ready(self, now_ns, *, end_ns=None, force=False):
        if self.origin_ns is None:
            return None
        target = self.next_target_ns
        if target > (now_ns if end_ns is None else min(now_ns, end_ns)):
            return None
        selected = {}
        # hand is selected first so q, successful q_cmd, HIL and tactile stay paired.
        order = sorted(self.names, key=lambda name: name != "hand")
        for name in order:
            values = self.queues[name]
            skew_ns = self._skew_ns(name)
            while values and (values[0].timestamp_ns < target - skew_ns
                              or values[0].timestamp_ns <= self.last_used[name]):
                values.popleft()
            candidates = [v for v in values
                          if abs(v.timestamp_ns - target) <= skew_ns
                          and v.received_ns <= now_ns
                          and (end_ns is None or v.timestamp_ns <= end_ns)]
            if name in ("tau", "wrench", "deform"):
                hand = selected.get("hand")
                candidates = [v for v in candidates
                              if hand is not None and v.timestamp_ns == hand.timestamp_ns]
            # Consume the oldest eligible source, rather than jumping between
            # neighbours when asynchronous clocks jitter across a half-period.
            # Greedy nearest matching can skip a real frame and then need to
            # reuse its successor on the next target.
            chosen = min(candidates, key=lambda v: v.timestamp_ns, default=None)
            selected[name] = chosen
        # A lagging member of the SharpA bundle can still arrive within the deadline.
        if not force and now_ns < target + self.wait_ns and any(v is None for v in selected.values()):
            return None
        for name, chosen in selected.items():
            if chosen is None:
                self.missing[name] += 1
            else:
                self.last_used[name] = chosen.timestamp_ns
        self.index += 1
        self.max_emit_delay_ms = max(self.max_emit_delay_ms, (now_ns - target) / 1e6)
        return target, selected

    def status(self):
        return dict(wait_limit_ms=self.wait_ns / 1e6, max_skew_ms=self.skew_ns / 1e6,
                    camera_max_skew_ms=self.camera_skew_ns / 1e6,
                    emitted_rows=self.index, missing=dict(self.missing),
                    overflows=dict(self.overflows), backwards=dict(self.rejected),
                    max_emit_delay_ms=round(self.max_emit_delay_ms, 3))
