"""The 7-state gate machine, as a pure module.

Three independent machines live here:

    UR      : stopped -> homed -> running        (3 states)
    SharpA  : stopped -> running                 (2 states)
    Record  : idle    -> recording               (2 states)

Nothing in this file imports rclpy, opens a socket, or touches hardware, so the
whole transition table -- including every refusal reason -- is testable offline.
`status_node.py` is a thin shell: it owns the ROS handles and the HTTP server
and delegates every decision to `GateMachine`.

The state values deliberately mirror `ur_msgs/ControlStatus` and
`ur_node/arbitration.py`. `arbitration.py` is the consumer: it reads
`ur_state` + `active_source` off `/control/status` and decides which producer
may drive the arms. So the semantics here must be exactly the ones it assumes:

    stopped  -> it accepts nothing
    homed    -> it accepts ONLY bias
    running  -> it accepts bias + the active source

which is why "start UR" lands in HOMED and not RUNNING: the operator gets one
state in which only the homing trajectory can move the arms.

Every refusal carries a reason string. A silent refusal makes a stalled arm
indistinguishable from a disconnected producer, which is exactly the failure
mode this whole design is trying to avoid -- so `gate_reason` is populated on
every rejected transition and surfaced on the web page.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable


class UrState(IntEnum):
    """Mirrors ControlStatus.UR_* and arbitration.UrState."""

    STOPPED = 0
    HOMED = 1
    RUNNING = 2


class SharpaState(IntEnum):
    """Mirrors ControlStatus.SHARPA_*."""

    STOPPED = 0
    RUNNING = 1


class RecordState(IntEnum):
    """Mirrors ControlStatus.RECORD_*."""

    IDLE = 0
    RECORDING = 1


class Source(IntEnum):
    """Mirrors ArmCommand.SOURCE_* and arbitration.Source."""

    UNKNOWN = 0
    BIAS = 1
    VIVE = 2
    ACTOR = 3


# Launch-parameter mode -> the source that is gated in while RUNNING.
MODE_SOURCES = {
    "teleop": Source.VIVE,
    "deploy": Source.ACTOR,
    # Offline replay emits the same executable-action contract as deploy;
    # keeping it on the actor gate prevents a replay stream from bypassing
    # the normal UR running-state checks.
    "replay": Source.ACTOR,
}

# The three camera views the recorder needs. Names match CameraSet fields.
CAMERA_VIEWS = ("ego", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class Outcome:
    """Result of an attempted transition.

    `bias_request` / `record_command` are the side effects the caller must
    publish. Keeping them in the return value rather than doing the publish
    here is what makes the machine testable without ROS.
    """

    accepted: bool
    reason: str = ""
    bias_request: str = ""
    record_command: str = ""
    hardware_stop: bool | None = None

    def __bool__(self) -> bool:
        return self.accepted


def _ok(**kwargs) -> Outcome:
    return Outcome(True, **kwargs)


def _no(reason: str) -> Outcome:
    return Outcome(False, reason)


@dataclass
class CameraFreshness:
    """Per-view health based on timestamp progress, not pipeline latency.

    Recording is asynchronous and preserves each camera's source timestamp, so
    a fixed capture/encode delay must not block recording. A view is healthy
    when it is valid and its source timestamp continues to advance. Repeated
    timestamps still become stale even if other views keep `/camera/set`
    arriving.
    """

    # view -> (sensor sampling timestamp in integer nanoseconds, valid flag)
    stamps: dict = field(default_factory=dict)
    # view -> wall-clock time at which that view's source stamp last advanced.
    progress_received: dict = field(default_factory=dict)
    # Wall-clock time the last CameraSet arrived, for "camera_node is alive".
    last_arrival: float | None = None

    def update(self, view_stamps_ns: dict, view_valid: dict, arrival: float) -> None:
        self.last_arrival = arrival
        for view in CAMERA_VIEWS:
            ns = view_stamps_ns.get(view)
            if ns is None:
                continue
            stamp = int(ns)
            previous = self.stamps.get(view)
            if previous is None or stamp != previous[0]:
                self.progress_received[view] = arrival
            self.stamps[view] = (stamp, bool(view_valid.get(view, False)))

    def stale_views(self, now: float, max_age_s: float) -> list:
        """Views that are missing, invalid, or have stopped advancing."""
        out = []
        for view in CAMERA_VIEWS:
            entry = self.stamps.get(view)
            if entry is None:
                out.append((view, None, "never received"))
                continue
            stamp, valid = entry
            progressed = self.progress_received.get(view)
            age = None if progressed is None else now - progressed
            if not valid:
                out.append((view, age, "flagged invalid"))
            elif stamp <= 0 or progressed is None:
                out.append((view, age, "missing timestamp"))
            elif age > max_age_s:
                out.append((view, age, f"not advancing {age:.2f}s"))
        return out

    def ages(self, now: float) -> dict:
        out = {}
        for view in CAMERA_VIEWS:
            progressed = self.progress_received.get(view)
            out[view] = None if progressed is None else round(now - progressed, 3)
        return out


# --- ur_safe_jog folded in -------------------------------------------------
# Same hard caps as the CLI tool. These are the numbers the commissioning
# procedure was signed off against; the web page must not widen them.
MAX_JOG_DISTANCE_MM = 5.0
MAX_JOG_SPEED_M_S = 0.02
MAX_JOG_ACCELERATION_M_S2 = 0.1
JOG_AXES = ("x", "y", "z")
JOG_SIDES = ("left", "right")


@dataclass(frozen=True)
class JogRequest:
    """One validated Cartesian nudge. Field names match `ur_safe_jog.JogRequest`."""

    side: str
    axis: str
    distance_mm: float
    speed_m_s: float = 0.01
    acceleration_m_s2: float = 0.05

    @property
    def confirmation_token(self) -> str:
        """Identical format to the CLI token.

        The confirmation CHANNEL moved from a typed CLI token to a button
        click, but the token itself is kept as the dry-run's identity: the
        confirm click has to name the exact preview it is confirming, so a
        stale page cannot confirm a jog the operator never previewed.
        """
        return f"{self.side}:{self.axis}:{self.distance_mm:+g}mm"


def validate_jog(side, axis, distance_mm, speed_m_s, acceleration_m_s2) -> JogRequest:
    """Build a JogRequest or raise ValueError. Same bounds as ur_safe_jog."""
    if side not in JOG_SIDES:
        raise ValueError("side must be left or right")
    if axis not in JOG_AXES:
        raise ValueError("axis must be x, y, or z")
    distance = _finite(distance_mm, "distance_mm")
    if distance == 0.0 or abs(distance) > MAX_JOG_DISTANCE_MM:
        raise ValueError(
            f"distance_mm must be nonzero and within +/-{MAX_JOG_DISTANCE_MM:g}"
        )
    speed = _bounded(speed_m_s, "speed_m_s", MAX_JOG_SPEED_M_S)
    accel = _bounded(acceleration_m_s2, "acceleration_m_s2", MAX_JOG_ACCELERATION_M_S2)
    return JogRequest(side=side, axis=axis, distance_mm=distance,
                      speed_m_s=speed, acceleration_m_s2=accel)


def _finite(value, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be a finite number")
    return out


def _bounded(value, field_name: str, maximum: float) -> float:
    out = _finite(value, field_name)
    if out <= 0.0 or out > maximum:
        raise ValueError(f"{field_name} must be in (0, {maximum:g}]")
    return out


def build_jog_target(current_tcp_pose, request: JogRequest) -> tuple:
    """Current RTDE TCP pose (6,) + request -> target pose (6,).

    Copied from `ur_safe_jog.build_target_pose` so the preview shown on the page
    is the same arithmetic the executor will do.
    """
    if len(current_tcp_pose) != 6:
        raise ValueError("current_tcp_pose must contain exactly 6 values")
    target = [_finite(v, "current_tcp_pose") for v in current_tcp_pose]
    target[JOG_AXES.index(request.axis)] += request.distance_mm / 1000.0
    return tuple(target)


@dataclass
class JogPreview:
    """A completed dry run, awaiting confirmation."""

    request: JogRequest
    current_pose: tuple
    target_pose: tuple
    created_at: float

    def payload(self) -> dict:
        return {
            "side": self.request.side,
            "axis": self.request.axis,
            "distance_mm": self.request.distance_mm,
            "speed_m_s": self.request.speed_m_s,
            "acceleration_m_s2": self.request.acceleration_m_s2,
            "current_tcp_pose": list(self.current_pose),
            "target_tcp_pose": list(self.target_pose),
            "confirmation_token": self.request.confirmation_token,
            "max_distance_mm": MAX_JOG_DISTANCE_MM,
            "max_speed_m_s": MAX_JOG_SPEED_M_S,
            "max_acceleration_m_s2": MAX_JOG_ACCELERATION_M_S2,
            "created_at": self.created_at,
        }


@dataclass
class GateMachine:
    """Pure seven-state controller used by the ROS node and its HTTP page.

    The machine deliberately owns no ROS objects.  ``attempt_*`` methods return
    side effects for the shell to publish and always retain a human-readable
    refusal in ``gate_reason``.  Bias completion is reported by the bias node as
    JSON; callers feed it through :meth:`on_bias_status`.
    """

    mode: str = "teleop"
    camera_max_age_s: float = 0.25
    execution_provider: str = "policy"
    ur_state: UrState = UrState.STOPPED
    sharpa_state: SharpaState = SharpaState.STOPPED
    record_state: RecordState = RecordState.IDLE
    gate_reason: str = ""
    session_id: str = ""
    bias_phase: str = "idle"
    bias_request: str = ""
    machine_ready: bool = False
    ur_hardware_ready: bool = False
    sharpa_hardware_ready: bool = False
    last_bias_status: dict = field(default_factory=dict)
    cameras: CameraFreshness = field(default_factory=CameraFreshness)
    jog_preview: JogPreview | None = None

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in MODE_SOURCES:
            raise ValueError(f"mode must be one of {sorted(MODE_SOURCES)}")
        self.execution_provider = str(self.execution_provider).strip().lower()
        if self.execution_provider not in {"policy", "replay"}:
            raise ValueError("execution_provider must be policy or replay")
        if self.camera_max_age_s <= 0:
            raise ValueError("camera_max_age_s must be positive")

    @property
    def active_source(self) -> Source:
        return MODE_SOURCES[self.mode]

    def _set_reason(self, reason: str) -> Outcome:
        self.gate_reason = reason
        return _no(reason)

    def _clear_reason(self) -> None:
        self.gate_reason = ""

    def update_hardware_readiness(self, ur_ready: bool | None = None,
                                  sharpa_ready: bool | None = None) -> None:
        if ur_ready is not None:
            self.ur_hardware_ready = bool(ur_ready)
        if sharpa_ready is not None:
            self.sharpa_hardware_ready = bool(sharpa_ready)

    def update_cameras(self, view_stamps_ns: dict, view_valid: dict,
                       arrival: float | None = None) -> None:
        self.cameras.update(view_stamps_ns, view_valid,
                            time.time() if arrival is None else arrival)
        if self.record_state == RecordState.IDLE and not self.gate_reason.startswith("record"):
            return
        stale = self.cameras.stale_views(time.time(), self.camera_max_age_s)
        if not stale:
            self._clear_reason()

    def camera_gate_reason(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        stale = self.cameras.stale_views(now, self.camera_max_age_s)
        if not stale:
            return ""
        details = ", ".join(
            f"{view}({why})" for view, _age, why in stale
        )
        return f"record blocked: camera views not fresh: {details}"

    def on_bias_status(self, payload: dict | str) -> None:
        if isinstance(payload, str):
            import json
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {"phase": "invalid", "detail": payload}
        self.last_bias_status = dict(payload or {})
        self.bias_phase = str(self.last_bias_status.get("phase", ""))
        if self.bias_phase in {"failed", "error"}:
            if self.bias_request == "home_both":
                self.machine_ready = False
            self._set_reason(f"bias failed: {self.last_bias_status.get('detail', 'unknown error')}")
            return
        if self.bias_phase == "done":
            request = self.bias_request
            if request in {"home_ur", "home_both"}:
                self.ur_state = UrState.HOMED
            if request in {"home_sharpa", "home_both"}:
                self.sharpa_state = SharpaState.STOPPED
            if request == "home_both":
                self.machine_ready = True
            self.bias_request = ""
            self._clear_reason()

    def attempt_start_ur(self) -> Outcome:
        if self.ur_state != UrState.STOPPED:
            return self._set_reason(f"start UR blocked: UR is already {self.ur_state.name.lower()}")
        self.bias_request = "home_ur"
        self.bias_phase = "requested"
        self._clear_reason()
        return _ok(bias_request="home_ur", hardware_stop=False)

    def attempt_run_ur(self) -> Outcome:
        if not self.machine_ready:
            return self._set_reason("machine start blocked: reset both UR and SharpA first")
        if not self.ur_hardware_ready:
            return self._set_reason("machine start blocked: UR is not in Normal mode")
        if not self.sharpa_hardware_ready:
            return self._set_reason("machine start blocked: SharpA state is unavailable")
        if self.ur_state == UrState.STOPPED:
            return self._set_reason("run UR blocked: UR is stopped; start/home UR first")
        if self.ur_state != UrState.HOMED:
            return self._set_reason("run UR blocked: UR is already running")
        if self.bias_phase not in {"done", "idle", ""}:
            return self._set_reason(f"run UR blocked: bias phase is {self.bias_phase}")
        self.ur_state = UrState.RUNNING
        self.sharpa_state = SharpaState.RUNNING
        self._clear_reason()
        return _ok(hardware_stop=False)

    def attempt_reset_ur(self) -> Outcome:
        if self.ur_state == UrState.RUNNING:
            return self._set_reason("UR reset blocked: pause the machine first")
        self.bias_request = "home_ur"
        self.bias_phase = "requested"
        self.record_state = RecordState.IDLE
        self.session_id = ""
        self.ur_state = UrState.STOPPED
        self._clear_reason()
        return _ok(bias_request="home_ur", hardware_stop=False)

    def attempt_reset_both(self) -> Outcome:
        """Return both arms and both hands to their deployment home pose."""
        if self.ur_state == UrState.RUNNING or self.sharpa_state == SharpaState.RUNNING:
            return self._set_reason("machine reset blocked: pause the machine first")
        if not self.ur_hardware_ready:
            return self._set_reason("machine reset blocked: UR is not in Normal mode")
        if not self.sharpa_hardware_ready:
            return self._set_reason("machine reset blocked: SharpA state is unavailable")
        record_command = "stop" if self.record_state == RecordState.RECORDING else ""
        self.bias_request = "home_both"
        self.bias_phase = "requested"
        self.record_state = RecordState.IDLE
        self.session_id = ""
        self.ur_state = UrState.STOPPED
        self.sharpa_state = SharpaState.STOPPED
        self.machine_ready = False
        self.jog_preview = None
        self._clear_reason()
        return _ok(
            bias_request="home_both",
            record_command=record_command,
            hardware_stop=False,
        )

    def attempt_start_deploy(self) -> Outcome:
        """Open the actor command gate after a complete reset."""
        if self.mode != "deploy":
            return self._set_reason(
                f"start deploy blocked: launch mode is {self.mode}, expected deploy"
            )
        return self.attempt_run_ur()

    def attempt_set_execution_provider(self, provider: str) -> Outcome:
        """Select the only executable-action producer actor may accept.

        The selector can only change while the machine gate is closed. This
        makes a provider change an explicit stopped-state operation instead of
        allowing policy and replay trajectories to race at the actor input.
        """
        requested = str(provider).strip().lower()
        if self.mode != "deploy":
            return self._set_reason(
                f"provider switch blocked: launch mode is {self.mode}, expected deploy"
            )
        if requested not in {"policy", "replay"}:
            return self._set_reason(
                f"provider switch blocked: unsupported provider {requested!r}"
            )
        if self.ur_state == UrState.RUNNING or self.sharpa_state == SharpaState.RUNNING:
            return self._set_reason(
                "provider switch blocked: stop the machine before switching"
            )
        self.execution_provider = requested
        self._clear_reason()
        return _ok()

    def attempt_pause(self) -> Outcome:
        """Freeze both robots while keeping every teleop process connected."""
        if self.ur_state != UrState.RUNNING or self.sharpa_state != SharpaState.RUNNING:
            return self._set_reason("machine pause blocked: machine is not running")
        record_command = "stop" if self.record_state == RecordState.RECORDING else ""
        self.ur_state = UrState.HOMED
        self.sharpa_state = SharpaState.STOPPED
        self.record_state = RecordState.IDLE
        self.session_id = ""
        self.jog_preview = None
        self.gate_reason = "machine paused"
        return _ok(record_command=record_command, hardware_stop=True)

    def attempt_stop_deploy(self) -> Outcome:
        return self.attempt_pause()

    def attempt_start_sharpa(self) -> Outcome:
        if self.sharpa_state == SharpaState.RUNNING:
            return self._set_reason("start SharpA blocked: SharpA is already running")
        self.bias_request = "home_sharpa"
        self.bias_phase = "requested"
        self._clear_reason()
        return _ok(bias_request="home_sharpa")

    def attempt_start_recording(self, session_id: str | None = None,
                                now: float | None = None) -> Outcome:
        if self.record_state == RecordState.RECORDING:
            return self._set_reason("record start blocked: recording is already active")
        # Recording is an asynchronous observation pipeline.  It must remain
        # available while the machines are paused, and Vive is not part of the
        # recorded action contract.  Individual samples retain source
        # timestamps/validity so they can be aligned and audited afterwards.
        camera_reason = self.camera_gate_reason(now)
        if camera_reason:
            return self._set_reason(camera_reason)
        self.record_state = RecordState.RECORDING
        self.session_id = session_id or time.strftime("episode_%Y%m%d_%H%M%S")
        self._clear_reason()
        return _ok(record_command="start")

    def attempt_stop_recording(self) -> Outcome:
        if self.record_state != RecordState.RECORDING:
            return self._set_reason("record stop blocked: recording is idle")
        self.record_state = RecordState.IDLE
        self.session_id = ""
        self._clear_reason()
        return _ok(record_command="stop")

    def attempt_discard_recording(self) -> Outcome:
        if self.record_state != RecordState.RECORDING:
            return self._set_reason("record discard blocked: recording is idle")
        self.record_state = RecordState.IDLE
        self.session_id = ""
        self._clear_reason()
        return _ok(record_command="discard")

    def dry_run_jog(self, side: str, axis: str, distance_mm: float,
                    speed_m_s: float, acceleration_m_s2: float,
                    current_pose, now: float | None = None) -> Outcome:
        try:
            req = validate_jog(side, axis, distance_mm, speed_m_s, acceleration_m_s2)
            target = build_jog_target(current_pose, req)
        except (TypeError, ValueError) as exc:
            return self._set_reason(f"jog dry-run refused: {exc}")
        self.jog_preview = JogPreview(req, tuple(float(v) for v in current_pose),
                                      target, time.time() if now is None else now)
        self._clear_reason()
        return _ok()

    def confirm_jog(self, token: str, executor: Callable[[JogPreview], object] | None = None) -> Outcome:
        preview = self.jog_preview
        if preview is None:
            return self._set_reason("jog confirm refused: perform a dry run first")
        if str(token) != preview.request.confirmation_token:
            return self._set_reason("jog confirm refused: confirmation does not match the dry run")
        try:
            if executor is not None:
                executor(preview)
        except Exception as exc:
            return self._set_reason(f"jog execution failed: {exc}")
        self.jog_preview = None
        self._clear_reason()
        return _ok()

    def status_payload(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        return {
            "ur_state": int(self.ur_state),
            "sharpa_state": int(self.sharpa_state),
            "record_state": int(self.record_state),
            "active_source": int(self.active_source),
            "mode": self.mode,
            "execution_provider": self.execution_provider,
            "machine_ready": self.machine_ready,
            "machine_running": self.ur_state == UrState.RUNNING and self.sharpa_state == SharpaState.RUNNING,
            "ur_hardware_ready": self.ur_hardware_ready,
            "sharpa_hardware_ready": self.sharpa_hardware_ready,
            "gate_reason": self.gate_reason,
            "session_id": self.session_id,
            "bias_phase": self.bias_phase,
            "camera_ages_s": self.cameras.ages(now),
            "camera_gate_reason": self.camera_gate_reason(now),
            "jog_preview": self.jog_preview.payload() if self.jog_preview else None,
        }
