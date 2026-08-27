"""Policy action slice to UR5 joint-plan conversion.

The core is deliberately independent of ROS so malformed actions and identity
handling can be tested without a running graph. The ROS wrapper only translates
messages and publishes one terminal ExecutionDone for every received action.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .trajectory import ExecutionTraceWriter, build_trajectory_payload


ARM_MAX_STEP_RAD = 0.25
HAND_MAX_STEP_RAD = 0.4
HARDWARE_ACK_TIMEOUT_S = 1.5


@dataclass(frozen=True)
class IkPlan:
    left_joint: np.ndarray
    right_joint: np.ndarray
    hand_joint: np.ndarray


@dataclass(frozen=True)
class PlannedAction:
    action: Any
    plan: IkPlan
    provider: str


class ActorIkError(ValueError):
    pass


class SequentialPlanBuffer:
    """Keep one executing plan and, at most, one prefetched successor.

    Admission and execution are deliberately separate: an action may be
    planned and buffered while the current chunk is still moving, but it is
    not promoted to ``current`` until ``finish_current`` is called after the
    current chunk's final hardware acknowledgements.
    """

    def __init__(self) -> None:
        self._current: PlannedAction | None = None
        self._next: PlannedAction | None = None

    @property
    def current(self) -> PlannedAction | None:
        return self._current

    @property
    def next(self) -> PlannedAction | None:
        return self._next

    @property
    def full(self) -> bool:
        return self._current is not None and self._next is not None

    def planning_seeds(
        self, measured_left: Any, measured_right: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._current is None:
            return (
                np.asarray(measured_left, dtype=np.float32),
                np.asarray(measured_right, dtype=np.float32),
            )
        return (
            np.asarray(self._current.plan.left_joint[-1], dtype=np.float32),
            np.asarray(self._current.plan.right_joint[-1], dtype=np.float32),
        )

    def admit(self, item: PlannedAction) -> str:
        if self._current is None:
            self._current = item
            return "current"
        if self._next is not None:
            raise ActorIkError("next actor action is already buffered")
        current_action = self._current.action
        if str(item.provider) != str(self._current.provider):
            raise ActorIkError("buffered actor actions must use the same provider")
        if str(item.action.session_id) != str(current_action.session_id):
            raise ActorIkError("buffered actor actions must use the same session")
        self._next = item
        return "next"

    def finish_current(self) -> tuple[PlannedAction, PlannedAction | None]:
        if self._current is None:
            raise ActorIkError("no current actor action to finish")
        finished = self._current
        self._current = self._next
        self._next = None
        return finished, self._current

    def clear(self) -> list[PlannedAction]:
        items = [item for item in (self._current, self._next) if item is not None]
        self._current = None
        self._next = None
        return items


class ChunkLatencyTracker:
    """Measure the command gap between consecutive executed chunks."""

    SCHEMA = "sharpa.actor_ik.chunk_latency.v1"

    def __init__(self) -> None:
        self.reset_statistics()

    def reset_statistics(
        self, *, session_id: str = "", provider: str = "policy"
    ) -> None:
        self._session_id = str(session_id)
        self._provider = str(provider)
        self._completed: dict[str, Any] | None = None
        self._current_latency_ms: float | None = None
        self._total_latency_ms = 0.0
        self._sample_count = 0
        self._previous_identity: dict[str, Any] | None = None
        self._current_identity: dict[str, Any] | None = None
        self._paused_at_ns: int | None = None

    def clear_pending_gap(self) -> bool:
        had_pending = self._completed is not None
        self._completed = None
        return had_pending

    def pause(self, paused_at_ns: int) -> bool:
        if self._paused_at_ns is not None:
            return False
        self._paused_at_ns = int(paused_at_ns)
        return True

    def resume(self, resumed_at_ns: int) -> bool:
        if self._paused_at_ns is None:
            return False
        resumed_at_ns = int(resumed_at_ns)
        pause_duration_ns = max(0, resumed_at_ns - self._paused_at_ns)
        if self._completed is not None:
            completed_at_ns = self._completed["completed_at_ns"]
            self._completed["completed_at_ns"] = (
                resumed_at_ns
                if completed_at_ns is None
                else int(completed_at_ns) + pause_duration_ns
            )
        self._paused_at_ns = None
        return True

    def ensure_context(self, *, session_id: str, provider: str) -> bool:
        session_id = str(session_id)
        provider = str(provider)
        if self._session_id == session_id and self._provider == provider:
            return False
        self.reset_statistics(session_id=session_id, provider=provider)
        return True

    @staticmethod
    def _identity(
        *, session_id: str, request_id: int, action_id: str, revision: int
    ) -> dict[str, Any]:
        return {
            "session_id": str(session_id),
            "request_id": int(request_id),
            "action_id": str(action_id),
            "revision": int(revision),
        }

    def mark_chunk_complete(
        self,
        *,
        session_id: str,
        provider: str,
        request_id: int,
        action_id: str,
        revision: int,
        completed_at_ns: int,
    ) -> None:
        self.ensure_context(session_id=session_id, provider=provider)
        identity = self._identity(
            session_id=session_id,
            request_id=request_id,
            action_id=action_id,
            revision=revision,
        )
        self._completed = {
            "identity": identity,
            "completed_at_ns": (
                None if self._paused_at_ns is not None else int(completed_at_ns)
            ),
        }

    def mark_next_command_published(
        self,
        *,
        session_id: str,
        provider: str,
        request_id: int,
        action_id: str,
        revision: int,
        published_at_ns: int,
    ) -> dict[str, Any] | None:
        if self.ensure_context(session_id=session_id, provider=provider):
            return None
        if self._completed is None:
            return None
        if self._paused_at_ns is not None:
            return None
        identity = self._identity(
            session_id=session_id,
            request_id=request_id,
            action_id=action_id,
            revision=revision,
        )
        previous_identity = self._completed["identity"]
        if identity == previous_identity:
            return None
        elapsed_ns = int(published_at_ns) - int(self._completed["completed_at_ns"])
        if elapsed_ns < 0:
            self._completed = None
            return None
        latency_ms = elapsed_ns / 1_000_000.0
        self._current_latency_ms = latency_ms
        self._total_latency_ms += latency_ms
        self._sample_count += 1
        self._previous_identity = previous_identity
        self._current_identity = identity
        self._completed = None
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        average_latency_ms = (
            self._total_latency_ms / self._sample_count
            if self._sample_count
            else None
        )
        return {
            "schema": self.SCHEMA,
            "provider": self._provider,
            "session_id": self._session_id,
            "current_latency_ms": self._current_latency_ms,
            "average_latency_ms": average_latency_ms,
            "sample_count": self._sample_count,
            "paused": self._paused_at_ns is not None,
            "waiting_for_next_chunk": self._completed is not None,
            "previous_request_id": (
                self._previous_identity["request_id"]
                if self._previous_identity is not None
                else None
            ),
            "previous_action_id": (
                self._previous_identity["action_id"]
                if self._previous_identity is not None
                else ""
            ),
            "current_request_id": (
                self._current_identity["request_id"]
                if self._current_identity is not None
                else None
            ),
            "current_action_id": (
                self._current_identity["action_id"]
                if self._current_identity is not None
                else ""
            ),
        }


def limit_joint_target(current: Any, target: Any, max_delta: float) -> np.ndarray:
    current_array = np.asarray(current, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if current_array.shape != target_array.shape:
        raise ActorIkError("current and target joints must have the same shape")
    if not np.isfinite(current_array).all() or not np.isfinite(target_array).all():
        raise ActorIkError("current and target joints must be finite")
    if not np.isfinite(max_delta) or max_delta <= 0.0:
        raise ActorIkError("max_delta must be finite and positive")
    delta = np.clip(target_array - current_array, -max_delta, max_delta)
    return current_array + delta


def next_step_due(published_at: float, frequency_hz: float) -> float:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ActorIkError("frequency_hz must be finite and positive")
    return float(published_at) + 1.0 / float(frequency_hz)


def execution_edges(
    execute_length: int, rtc: int, executed_in_slice: int
) -> tuple[bool, bool]:
    """Return ``(execute_fetch, execute_done)`` after one acknowledged step."""

    length = int(execute_length)
    rtc_steps = int(rtc)
    executed = int(executed_in_slice)
    if length <= 0:
        raise ActorIkError("execute_length must be positive")
    if rtc_steps < 0 or (rtc_steps >= length and rtc_steps != 0):
        raise ActorIkError("rtc must be zero or smaller than execute_length")
    if executed < 0 or executed > length:
        raise ActorIkError("executed_in_slice is outside the action")
    execute_done = executed == length
    execute_fetch = (
        execute_done if rtc_steps == 0 else executed == length - rtc_steps
    )
    return execute_fetch, execute_done


def _array(value: Any, length: int, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    expected = length * dimension
    if array.size != expected:
        raise ActorIkError(f"{name} has {array.size} values; expected {expected}")
    if not np.isfinite(array).all():
        raise ActorIkError(f"{name} contains non-finite values")
    return array.reshape(length, dimension)


class ActorIkCore:
    """Convert one ExecutableAction into a complete joint plan."""

    def __init__(self, kinematics: Any):
        self.kinematics = kinematics

    def make_plan(
        self,
        *,
        left_eef: Any,
        right_eef: Any,
        hand_joint: Any,
        execute_length: int,
        left_seed: Any,
        right_seed: Any,
    ) -> IkPlan:
        length = int(execute_length)
        if length <= 0:
            raise ActorIkError("execute_length must be positive")
        left = _array(left_eef, length, 9, "left_eef")
        right = _array(right_eef, length, 9, "right_eef")
        hand = _array(hand_joint, length, 44, "hand_joint")
        left_near = np.asarray(left_seed, dtype=np.float64).reshape(-1)
        right_near = np.asarray(right_seed, dtype=np.float64).reshape(-1)
        if left_near.shape != (6,) or right_near.shape != (6,):
            raise ActorIkError("joint seeds must each have shape (6,)")

        left_plan = np.empty((length, 6), dtype=np.float32)
        right_plan = np.empty((length, 6), dtype=np.float32)
        for index in range(length):
            left_next = np.asarray(
                self.kinematics.arm_ik_eef9(
                    left[index], left_near, "left", ref_joint=left_near
                ),
                dtype=np.float64,
            )
            right_next = np.asarray(
                self.kinematics.arm_ik_eef9(
                    right[index], right_near, "right", ref_joint=right_near
                ),
                dtype=np.float64,
            )
            if left_next.shape != (6,) or right_next.shape != (6,):
                raise ActorIkError("IK must return one finite joint vector per arm")
            if not np.isfinite(left_next).all() or not np.isfinite(right_next).all():
                raise ActorIkError("IK returned non-finite joints")
            left_plan[index] = left_next
            right_plan[index] = right_next
            left_near, right_near = left_next, right_next
        return IkPlan(left_plan, right_plan, hand.astype(np.float32, copy=True))


class ActorIkNode:
    """ROS-free action planner facade used by offline integrations.

    The executable ROS node below owns transport and acknowledgement handling;
    this facade keeps the same planning contract available to callers that do
    not have a running rclpy context.
    """

    def __init__(self, kinematics: Any):
        self.core = ActorIkCore(kinematics)

    def make_plan(self, **kwargs: Any) -> IkPlan:
        return self.core.make_plan(**kwargs)


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import String
    from ur_msgs.msg import (
        ActorTrajectory,
        ArmCommand,
        ControlStatus,
        ExecutionDone,
        ExecutableAction,
        HandCommand,
    )
    from ur_msgs.msg import HandStateFrame, UrStateFrame
    from ur_common.kinematics import UrCommonKinematics
    from ur_common.geometry import CANONICAL_EEF_FRAME
    from ur_common.safety import SafetyConfig

    class RosActorIkNode(Node):
        def __init__(self):
            super().__init__("actor_ik_node")
            self.declare_parameter(
                "execution_log_root",
                str(
                    Path(__file__).resolve().parents[4]
                    / "log"
                    / "deploy"
                ),
            )
            self.declare_parameter("execution_log_retention_days", 7)
            self.declare_parameter("execution_log_max_runs", 100)
            self.declare_parameter(
                "execution_log_max_total_bytes", 5 * 1024 * 1024 * 1024
            )
            self._kin = UrCommonKinematics(config=SafetyConfig.v1_defaults())
            self._core = ActorIkCore(self._kin)
            self._left = np.zeros(6, dtype=np.float32)
            self._right = np.zeros(6, dtype=np.float32)
            self._hand = np.zeros(44, dtype=np.float32)
            self._ur_valid = False
            self._hand_valid = False
            self._arm_pub = self.create_publisher(ArmCommand, "/ur/command", 10)
            self._hand_pub = self.create_publisher(HandCommand, "/sharpa/command", 10)
            self._done_pub = self.create_publisher(ExecutionDone, "/policy/execution_done", 10)
            self._trajectory_pub = self.create_publisher(
                ActorTrajectory, "/actor_ik/trajectory", 10
            )
            self._latency_pub = self.create_publisher(
                String, "/actor_ik/latency_status", 10
            )
            self._latency = ChunkLatencyTracker()
            self._plans = SequentialPlanBuffer()
            self._pending = {}
            self._active = None
            self._active_fetch_sent = False
            self._scheduled = None
            self._execution_provider = "policy"
            self._machine_running = False
            self._execution_paused = False
            self._execution_dry_run = False
            self._pause_started = None
            self._held_completion = None
            # STOP clears Actor-owned statistics immediately, but deliberately
            # leaves the last published Monitor snapshot untouched.  The next
            # live/preview start publishes this empty snapshot.
            self._latency_monitor_clear_pending = True
            self._trace = ExecutionTraceWriter(
                str(self.get_parameter("execution_log_root").value),
                retention_days=int(
                    self.get_parameter("execution_log_retention_days").value
                ),
                max_runs=int(self.get_parameter("execution_log_max_runs").value),
                max_total_bytes=int(
                    self.get_parameter("execution_log_max_total_bytes").value
                ),
            )
            self.create_subscription(
                ExecutableAction,
                "/policy/executable_action",
                lambda message: self._action(message, "policy"),
                10,
            )
            self.create_subscription(
                ExecutableAction,
                "/replay/executable_action",
                lambda message: self._action(message, "replay"),
                10,
            )
            self.create_subscription(
                String,
                "/control/execution_provider",
                self._provider_update,
                10,
            )
            self.create_subscription(
                String,
                "/deploy/execution_event",
                self._external_event,
                100,
            )
            self.create_subscription(
                ControlStatus, "/control/status", self._control_status, 10
            )
            self.create_subscription(UrStateFrame, "/ur/state", self._ur_state, 10)
            self.create_subscription(HandStateFrame, "/sharpa/state", self._hand_state, 10)
            from ur_msgs.msg import HardwareCommandResult
            self.create_subscription(
                HardwareCommandResult,
                "/ur/command_result",
                lambda message: self._ack(message, "ur"),
                10,
            )
            self.create_subscription(
                HardwareCommandResult,
                "/sharpa/command_result",
                lambda message: self._ack(message, "sharpa"),
                10,
            )
            self._watchdog = self.create_timer(0.005, self._process)

        def _provider_update(self, message):
            provider = str(message.data).strip().lower()
            if provider in {"policy", "replay"}:
                if (
                    provider != self._execution_provider
                    and self._plans.current is not None
                ):
                    self._abort_active("execution provider changed")
                if provider != self._execution_provider:
                    previous = self._execution_provider
                    self._execution_provider = provider
                    self._trace.event(
                        "execution_provider_changed",
                        node="actor_ik_node",
                        previous_provider=previous,
                        provider=provider,
                    )

        def _external_event(self, message):
            try:
                payload = json.loads(message.data or "{}")
                if not isinstance(payload, dict) or not payload.get("event"):
                    raise ValueError("execution event must be a JSON object with event")
                self._trace.record(payload)
            except Exception as exc:
                self.get_logger().warning(f"invalid deploy execution event: {exc}")

        def _control_status(self, message):
            live_running = bool(
                message.ur_state == ControlStatus.UR_RUNNING
                and message.sharpa_state == ControlStatus.SHARPA_RUNNING
            )
            dry_run = bool(message.execution_dry_run)
            running = bool(live_running or dry_run)
            paused = bool(message.execution_paused)
            mode_changed = bool(
                self._machine_running
                and running
                and dry_run != self._execution_dry_run
            )
            started = bool(not self._machine_running and running)
            if self._machine_running and (not running or mode_changed):
                final_state = {
                    "provider": self._execution_provider,
                    "mode": "preview" if self._execution_dry_run else "live",
                    "active_action": self._active,
                    "buffered_action": (
                        None
                        if self._plans.next is None
                        else (
                            self._plans.next.action.action_id,
                            int(self._plans.next.action.revision),
                        )
                    ),
                    "scheduled_step": (
                        int(self._scheduled["step"])
                        if self._scheduled is not None
                        else None
                    ),
                    "pending_steps": [
                        {
                            "session_id": str(value["action"].session_id),
                            "request_id": int(value["action"].request_id),
                            "action_id": key[0],
                            "revision": key[1],
                            "step_index": key[2],
                            "received_acks": sorted(value["acks"]),
                            "missing_acks": sorted({"ur", "sharpa"} - value["acks"]),
                            "pending_ms": (
                                time.monotonic() - float(value["started"])
                            )
                            * 1000.0,
                        }
                        for key, value in self._pending.items()
                    ],
                }
                self._execution_paused = False
                self._pause_started = None
                self._abort_active(
                    "execution mode changed"
                    if mode_changed
                    else "deployment stopped",
                    publish_latency=False,
                )
                self._latency.reset_statistics(provider=self._execution_provider)
                self._latency_monitor_clear_pending = True
                path = self._trace.stop(
                    "execution_mode_changed" if mode_changed else "execution_stopped",
                    final_state=final_state,
                )
                if path:
                    self.get_logger().info(f"execution trace finished: {path}")
            if running and (started or mode_changed):
                try:
                    path = self._trace.start(
                        mode="preview" if dry_run else "live",
                        provider=self._execution_provider,
                    )
                    self.get_logger().info(f"execution trace started: {path}")
                except Exception as exc:
                    self.get_logger().error(f"execution trace unavailable: {exc}")
            elif running and paused and not self._execution_paused:
                self._execution_paused = True
                self._pause_started = time.monotonic()
                self._latency.pause(time.monotonic_ns())
                self._trace.event("execution_paused", node="actor_ik_node")
                self._publish_latency()
            elif running and self._execution_paused and not paused:
                now = time.monotonic()
                pause_duration = max(0.0, now - float(self._pause_started or now))
                for pending in self._pending.values():
                    pending["started"] += pause_duration
                if self._scheduled is not None:
                    self._scheduled["due"] = (
                        now
                        if self._scheduled["due"] is None
                        else self._scheduled["due"] + pause_duration
                    )
                self._execution_paused = False
                self._pause_started = None
                self._latency.resume(time.monotonic_ns())
                self._trace.event(
                    "execution_resumed",
                    node="actor_ik_node",
                    pause_duration_ms=pause_duration * 1000.0,
                )
                self._publish_latency()
                self._release_held_completion()
            self._machine_running = running
            self._execution_dry_run = dry_run
            if started and self._latency_monitor_clear_pending:
                self._publish_latency()
                self._latency_monitor_clear_pending = False
            if running and not paused:
                self._process()

        def destroy_node(self):
            path = self._trace.stop(
                "actor_node_destroyed",
                final_state={
                    "provider": self._execution_provider,
                    "mode": "preview" if self._execution_dry_run else "live",
                    "active_action": self._active,
                    "buffered_action": (
                        None
                        if self._plans.next is None
                        else (
                            self._plans.next.action.action_id,
                            int(self._plans.next.action.revision),
                        )
                    ),
                    "pending_step_count": len(self._pending),
                },
            )
            if path:
                self.get_logger().info(f"execution trace finished: {path}")
            return super().destroy_node()

        def _publish_latency(self):
            payload = self._latency.snapshot()
            payload["timestamp_ns"] = self.get_clock().now().nanoseconds
            message = String()
            message.data = json.dumps(payload, separators=(",", ":"))
            self._latency_pub.publish(message)

        def _clear_latency_gap(self, *, publish=True):
            if self._latency.clear_pending_gap():
                if publish:
                    self._publish_latency()

        def _release_held_completion(self):
            held = self._held_completion
            if held is None:
                return
            self._held_completion = None
            self._finish_action(
                held["action"], held["success"], held["steps"], held["error"]
            )

        def _complete_or_hold(self, action, success, steps, error=""):
            if self._execution_paused:
                self._held_completion = {
                    "action": action,
                    "success": bool(success),
                    "steps": int(steps),
                    "error": str(error),
                }
                return
            self._finish_action(action, success, steps, error)

        @staticmethod
        def _same_action(left, right):
            return (
                str(left.session_id) == str(right.session_id)
                and int(left.request_id) == int(right.request_id)
                and str(left.action_id) == str(right.action_id)
                and int(left.revision) == int(right.revision)
            )

        def _finish_action(self, action, success, steps, error=""):
            current = self._plans.current
            if current is None or not self._same_action(current.action, action):
                self.get_logger().error(
                    "actor plan buffer lost current action identity"
                )
                self._active = None
                self._done(action, False, steps, "actor plan buffer identity mismatch")
                return

            if success:
                fetch_at_done = not self._active_fetch_sent
                _, promoted = self._plans.finish_current()
                self._active = None
                self._active_fetch_sent = False
                self._done(
                    action,
                    True,
                    steps,
                    error,
                    execute_fetch=fetch_at_done,
                )
                if promoted is not None:
                    self._activate_current()
                return

            dropped = self._plans.clear()
            self._active = None
            self._active_fetch_sent = False
            self._done(action, False, steps, error)
            for item in dropped[1:]:
                queued = item.action
                self._done(
                    queued,
                    False,
                    int(queued.execute_start),
                    f"previous actor action failed: {error}",
                )

        def _abort_active(self, error, *, publish_latency=True):
            action = None
            executed_steps = 0
            if self._held_completion is not None:
                action = self._held_completion["action"]
                executed_steps = int(self._held_completion["steps"])
            elif self._scheduled is not None:
                action = self._scheduled["action"]
                executed_steps = int(action.execute_start) + int(
                    self._scheduled["step"]
                )
            elif self._pending:
                pending = next(iter(self._pending.values()))
                action = pending["action"]
                executed_steps = int(action.execute_start) + int(pending["step"])
            self._scheduled = None
            self._pending.clear()
            self._held_completion = None
            self._active = None
            self._active_fetch_sent = False
            self._clear_latency_gap(publish=publish_latency)
            buffered = self._plans.clear()
            if action is not None:
                self._done(action, False, executed_steps, error)
            for item in buffered:
                queued = item.action
                if action is not None and self._same_action(queued, action):
                    continue
                self._done(
                    queued,
                    False,
                    int(queued.execute_start),
                    f"buffered action cancelled: {error}",
                )

        def _ur_state(self, message):
            left = np.asarray(message.left_joint, dtype=np.float32)
            right = np.asarray(message.right_joint, dtype=np.float32)
            self._ur_valid = bool(
                message.valid
                and left.shape == (6,)
                and right.shape == (6,)
                and np.isfinite(left).all()
                and np.isfinite(right).all()
            )
            if self._ur_valid:
                self._left = left
                self._right = right

        def _hand_state(self, message):
            hand = np.asarray(message.q, dtype=np.float32)
            self._hand_valid = bool(
                message.left_valid
                and message.right_valid
                and hand.shape == (44,)
                and np.isfinite(hand).all()
            )
            if self._hand_valid:
                self._hand = hand

        def _execution_update(
            self,
            action,
            *,
            execute_fetch,
            execute_done,
            success,
            steps,
            error="",
        ):
            message = ExecutionDone()
            message.session_id = action.session_id
            message.request_id = action.request_id
            message.action_id = action.action_id
            message.revision = action.revision
            message.execute_start = action.execute_start
            message.execute_length = action.execute_length
            message.executed_steps = steps
            message.success = success
            message.execute_fetch = bool(execute_fetch)
            message.execute_done = bool(execute_done)
            message.error = error
            message.timestamp_ns = self.get_clock().now().nanoseconds
            self._done_pub.publish(message)
            self._trace.event(
                "execution_state_updated",
                provider=self._execution_provider,
                session_id=str(action.session_id),
                request_id=int(action.request_id),
                action_id=str(action.action_id),
                revision=int(action.revision),
                executed_steps=int(steps),
                execute_fetch=bool(execute_fetch),
                execute_done=bool(execute_done),
                success=bool(success),
                error=str(error),
            )

        def _done(
            self, action, success, steps, error="", *, execute_fetch=False
        ):
            self._execution_update(
                action,
                execute_fetch=bool(execute_fetch and success),
                execute_done=True,
                success=success,
                steps=steps,
                error=error,
            )

        def _fetch(self, action, executed_steps):
            if self._active_fetch_sent:
                return
            self._active_fetch_sent = True
            self._execution_update(
                action,
                execute_fetch=True,
                execute_done=False,
                success=True,
                steps=executed_steps,
            )

        def _ack(self, message, expected_device):
            if int(message.source) != int(ArmCommand.SOURCE_ACTOR):
                return
            if str(message.device) != expected_device:
                self.get_logger().warning(
                    f"ignoring command result from {message.device!r} on "
                    f"{expected_device} result topic"
                )
                return
            key = (message.action_id, int(message.revision), int(message.step_index))
            pending = self._pending.get(key)
            self._trace.event(
                "hardware_command_result_received",
                node="actor_ik_node",
                action_id=str(message.action_id),
                revision=int(message.revision),
                step_index=int(message.step_index),
                device=str(message.device),
                dry_run=bool(message.dry_run),
                success=bool(message.success),
                failure_code=str(message.failure_code),
                failure_message=str(message.failure_message),
                matched_pending=pending is not None,
            )
            if pending is None:
                return
            if bool(message.dry_run) != bool(pending["dry_run"]):
                self.get_logger().warning(
                    f"ignoring {expected_device} result with mismatched dry_run mode"
                )
                return
            pending["acks"].add(str(message.device))
            if not message.success:
                pending["failed"] = str(message.failure_message or message.failure_code)
            self._trace.event(
                "command_ack",
                device=str(message.device),
                action_id=str(message.action_id),
                revision=int(message.revision),
                step_index=int(message.step_index),
                dry_run=bool(message.dry_run),
                success=bool(message.success),
                failure_code=str(message.failure_code),
                failure_message=str(message.failure_message),
            )
            if pending["acks"] >= {"ur", "sharpa"}:
                self._finish_pending(key, pending)

        def _finish_pending(self, key, pending):
            self._pending.pop(key, None)
            if pending["failed"]:
                self._scheduled = None
                self._clear_latency_gap()
                self._complete_or_hold(
                    pending["action"],
                    False,
                    pending["action"].execute_start + pending["step"],
                    pending["failed"],
                )
                return
            executed_in_slice = int(pending["step"]) + 1
            execute_fetch, execute_done = execution_edges(
                pending["action"].execute_length,
                getattr(pending["action"], "rtc", 0),
                executed_in_slice,
            )
            if execute_done:
                action = pending["action"]
                self._latency.mark_chunk_complete(
                    session_id=action.session_id,
                    provider=self._execution_provider,
                    request_id=action.request_id,
                    action_id=action.action_id,
                    revision=action.revision,
                    completed_at_ns=time.monotonic_ns(),
                )
                self._publish_latency()
                self._complete_or_hold(
                    action, True, action.execute_start + action.execute_length
                )
                return
            if execute_fetch:
                self._fetch(
                    pending["action"],
                    int(pending["action"].execute_start) + executed_in_slice,
                )
            self._scheduled = {
                "action": pending["action"],
                "plan": pending["plan"],
                "step": pending["step"] + 1,
                "due": next_step_due(
                    pending["published_at"], pending["action"].frequency_hz
                ),
            }

        def _process(self):
            if self._execution_paused:
                return
            now = time.monotonic()
            for key, pending in list(self._pending.items()):
                if now - pending["started"] > pending["timeout"]:
                    received_acks = sorted(pending["acks"])
                    missing_acks = sorted({"ur", "sharpa"} - pending["acks"])
                    self._trace.event(
                        "hardware_command_ack_timeout",
                        node="actor_ik_node",
                        session_id=str(pending["action"].session_id),
                        request_id=int(pending["action"].request_id),
                        action_id=str(pending["action"].action_id),
                        revision=int(pending["action"].revision),
                        step_index=int(pending["step"]),
                        received_acks=received_acks,
                        missing_acks=missing_acks,
                        pending_ms=(now - pending["started"]) * 1000.0,
                        success=False,
                    )
                    self._pending.pop(key, None)
                    self._scheduled = None
                    self._clear_latency_gap()
                    self._complete_or_hold(
                        pending["action"],
                        False,
                        pending["action"].execute_start + pending["step"],
                        "hardware command acknowledgement timeout",
                    )
                    return
            scheduled = self._scheduled
            if scheduled is None or self._pending or now < scheduled["due"]:
                return
            self._scheduled = None
            try:
                self._publish_step(
                    scheduled["action"], scheduled["plan"], scheduled["step"]
                )
            except Exception as exc:
                self._clear_latency_gap()
                self.get_logger().error(f"action execution failed: {exc}")
                self._complete_or_hold(
                    scheduled["action"],
                    False,
                    scheduled["action"].execute_start + scheduled["step"],
                    str(exc),
                )

        def _publish_step(self, action, plan, step):
            if not self._ur_valid or not self._hand_valid:
                raise ActorIkError("valid UR and SharpA state are required")
            left_target = limit_joint_target(
                self._left, plan.left_joint[step], ARM_MAX_STEP_RAD
            )
            right_target = limit_joint_target(
                self._right, plan.right_joint[step], ARM_MAX_STEP_RAD
            )
            hand_target = limit_joint_target(
                self._hand, plan.hand_joint[step], HAND_MAX_STEP_RAD
            )
            arm = ArmCommand()
            arm.source = ArmCommand.SOURCE_ACTOR
            arm.action_id = action.action_id
            arm.revision = action.revision
            arm.step_index = step
            arm.period_s = 1.0 / action.frequency_hz
            arm.left_joint = left_target.tolist()
            arm.right_joint = right_target.tolist()
            arm.left_enable = True
            arm.right_enable = True
            arm.dry_run = bool(self._execution_dry_run)
            hand = HandCommand()
            hand.source = HandCommand.SOURCE_ACTOR
            hand.action_id = action.action_id
            hand.revision = action.revision
            hand.step_index = step
            hand.q_cmd = hand_target.tolist()
            hand.left_enable = True
            hand.right_enable = True
            hand.dry_run = bool(self._execution_dry_run)
            key = (action.action_id, int(action.revision), step)
            published_at = time.monotonic()
            self._pending[key] = {
                "action": action,
                "plan": plan,
                "step": step,
                "acks": set(),
                "failed": "",
                "dry_run": bool(self._execution_dry_run),
                "started": published_at,
                "published_at": published_at,
                "timeout": max(
                    HARDWARE_ACK_TIMEOUT_S, 4.0 / float(action.frequency_hz)
                ),
            }
            self._arm_pub.publish(arm)
            self._hand_pub.publish(hand)
            self._trace.event(
                "command_published",
                action_id=str(action.action_id),
                revision=int(action.revision),
                step_index=int(step),
                dry_run=bool(self._execution_dry_run),
                left_arm_joint=left_target.tolist(),
                right_arm_joint=right_target.tolist(),
                hand_joint=hand_target.tolist(),
            )
            if step == 0:
                sample = self._latency.mark_next_command_published(
                    session_id=action.session_id,
                    provider=self._execution_provider,
                    request_id=action.request_id,
                    action_id=action.action_id,
                    revision=action.revision,
                    published_at_ns=time.monotonic_ns(),
                )
                if sample is not None:
                    self._publish_latency()

        def _publish_trajectory(self, action, plan):
            payload = build_trajectory_payload(
                action,
                plan,
                provider=self._execution_provider,
                dry_run=self._execution_dry_run,
                trace_dir=self._trace.path,
            )
            message = ActorTrajectory()
            message.header.stamp = self.get_clock().now().to_msg()
            for field in (
                "provider",
                "dry_run",
                "trace_dir",
                "session_id",
                "request_id",
                "action_id",
                "revision",
                "timestamp_ns",
                "frequency_hz",
                "action_length",
                "execute_start",
                "execute_length",
                "rtc",
                "eef_frame",
                "left_eef",
                "right_eef",
                "left_arm_joint",
                "right_arm_joint",
                "hand_joint",
                "ik_applied",
            ):
                setattr(message, field, payload[field])
            message.diagnostics_json = json.dumps(
                payload["diagnostics"], separators=(",", ":")
            )
            self._trajectory_pub.publish(message)
            self._trace.trajectory(payload)
            self._trace.event(
                "trajectory_published",
                provider=self._execution_provider,
                action_id=str(action.action_id),
                revision=int(action.revision),
                execute_start=int(action.execute_start),
                execute_length=int(action.execute_length),
                rtc=int(getattr(action, "rtc", 0)),
            )

        def _activate_current(self):
            current = self._plans.current
            if current is None:
                return
            action = current.action
            self._active = (action.action_id, int(action.revision))
            self._active_fetch_sent = False
            self._scheduled = {
                "action": action,
                "plan": current.plan,
                "step": 0,
                "due": None if self._execution_paused else time.monotonic(),
            }
            self._trace.event(
                "actor_action_started",
                node="actor_ik_node",
                provider=str(current.provider),
                session_id=str(action.session_id),
                request_id=int(action.request_id),
                action_id=str(action.action_id),
                revision=int(action.revision),
            )
            self._process()

        def _action(self, action, provider="policy"):
            try:
                self._trace.event(
                    "actor_action_received",
                    node="actor_ik_node",
                    provider=str(provider),
                    session_id=str(action.session_id),
                    request_id=int(action.request_id),
                    action_id=str(action.action_id),
                    revision=int(action.revision),
                    execute_start=int(action.execute_start),
                    execute_length=int(action.execute_length),
                    rtc=int(getattr(action, "rtc", 0)),
                )
                if provider != self._execution_provider:
                    self.get_logger().warning(
                        f"ignoring {provider} action while provider is "
                        f"{self._execution_provider}"
                    )
                    self._done(
                        action,
                        False,
                        int(action.execute_start),
                        f"execution provider is {self._execution_provider}",
                    )
                    return
                if not self._machine_running:
                    raise ActorIkError("machine execution gate is not running")
                if not self._ur_valid or not self._hand_valid:
                    raise ActorIkError("valid UR and SharpA state are required")
                if self._plans.full:
                    raise ActorIkError("next actor action is already buffered")
                if self._plans.current is not None:
                    current_action = self._plans.current.action
                    if str(action.session_id) != str(current_action.session_id):
                        raise ActorIkError(
                            "buffered actor actions must use the same session"
                        )
                if self._latency.ensure_context(
                    session_id=action.session_id, provider=provider
                ):
                    self._publish_latency()
                if action.eef_frame != CANONICAL_EEF_FRAME:
                    raise ActorIkError(f"unsupported eef_frame {action.eef_frame!r}")
                if not np.isfinite(action.frequency_hz) or not 5.0 <= action.frequency_hz <= 125.0:
                    raise ActorIkError("frequency_hz must be in [5, 125]")
                rtc = int(getattr(action, "rtc", 0))
                if rtc < 0 or (rtc >= int(action.execute_length) and rtc != 0):
                    raise ActorIkError(
                        "rtc must be zero or smaller than execute_length"
                    )
                left_seed, right_seed = self._plans.planning_seeds(
                    self._left, self._right
                )
                if action.left_eef and action.right_eef:
                    plan = self._core.make_plan(left_eef=action.left_eef, right_eef=action.right_eef, hand_joint=action.hand_joint, execute_length=action.execute_length, left_seed=left_seed, right_seed=right_seed)
                else:
                    left = _array(action.left_arm_joint, action.execute_length, 6, "left_arm_joint")
                    right = _array(action.right_arm_joint, action.execute_length, 6, "right_arm_joint")
                    hand = _array(action.hand_joint, action.execute_length, 44, "hand_joint")
                    plan = IkPlan(left.astype(np.float32), right.astype(np.float32), hand.astype(np.float32))
                self._trace.event(
                    "actor_plan_ready",
                    node="actor_ik_node",
                    provider=str(provider),
                    session_id=str(action.session_id),
                    request_id=int(action.request_id),
                    action_id=str(action.action_id),
                    revision=int(action.revision),
                    execute_length=int(action.execute_length),
                )
                self._publish_trajectory(action, plan)
                disposition = self._plans.admit(
                    PlannedAction(action=action, plan=plan, provider=provider)
                )
                if disposition == "current":
                    self._activate_current()
                else:
                    self._trace.event(
                        "actor_action_buffered",
                        node="actor_ik_node",
                        provider=str(provider),
                        session_id=str(action.session_id),
                        request_id=int(action.request_id),
                        action_id=str(action.action_id),
                        revision=int(action.revision),
                    )
            except Exception as exc:
                self.get_logger().error(f"action IK failed: {exc}")
                self._clear_latency_gap()
                self._done(action, False, action.execute_start, str(exc))

    rclpy.init(args=args)
    node = RosActorIkNode()
    try:
        rclpy.spin(node)
    except (rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
