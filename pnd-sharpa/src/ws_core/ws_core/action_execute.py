#!/usr/bin/env python3
"""Schedule joint-space action plans and send PND action frames."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import threading
import time
import zlib
from typing import Any

import numpy as np
import rclpy
from deploy_common.joints import ADAM_COMMAND_JOINTS_19, SHARPA_JOINT_NAMES
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ws_msgs.msg import ActionPlan, ExecutionDone, PndAction, Status

try:
    from deploy_common import protocol as deploy_protocol
except ImportError:
    deploy_protocol = None

from .common import age_ms, json_bytes, json_dumps, json_or_raw, make_status, now_ns, set_header


FALLBACK_MAGIC = b"PND1"
FALLBACK_VERSION = 1
FALLBACK_FRAME_TYPE_ACTION = 1
FALLBACK_HEADER_STRUCT = struct.Struct("!4sBBHQqII")


@dataclass
class JointPlan:
    plan_id: int
    request_id: int
    request_stamp_ns: int
    source: str
    source_kind: str
    action_hz: float
    elapsed_ns: np.ndarray
    adam_q: np.ndarray
    sharpa_q: np.ndarray
    adam_valid: bool
    sharpa_valid: bool
    selected_abs62: np.ndarray | None
    debug: dict[str, Any]
    received_time: float
    server_execution: dict[str, Any] | None = None
    next_step: int = 0
    started_monotonic_ns: int | None = None
    completed: bool = False

    @property
    def horizon(self) -> int:
        return int(self.elapsed_ns.size)


@dataclass
class Selection:
    plan: JointPlan
    step_index: int


class ActionExecute(Node):
    def __init__(self) -> None:
        super().__init__("action_execute")
        self.declare_parameter("listen_host", "0.0.0.0")
        self.declare_parameter("listen_port", 15010)
        self.declare_parameter("action_plan_topic", "/ws/action_plan")
        self.declare_parameter("execution_done_topic", "/ws/execution_done")
        self.declare_parameter("action_topic", "/ws/action")
        self.declare_parameter("status_topic", "/ws/action_execute/status")
        self.declare_parameter("plan_debug_topic", "/ws/action_execute/plan_debug")
        self.declare_parameter("safety_topic", "/ws/action_execute/safety")
        self.declare_parameter("socket_timeout_s", 0.2)
        self.declare_parameter("action_ttl_ms", 120)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("enable_adam", True)
        self.declare_parameter("enable_sharpa", True)
        self.declare_parameter("actor_send_hz", 15.0)
        self.declare_parameter("execution_mode", "synchronous")
        self.declare_parameter("sharpa_control_mode", "position")

        self.listen_host = str(self.get_parameter("listen_host").value)
        self.listen_port = int(self.get_parameter("listen_port").value)
        self.action_plan_topic = str(self.get_parameter("action_plan_topic").value)
        self.execution_done_topic = str(
            self.get_parameter("execution_done_topic").value
        )
        self.action_topic = str(self.get_parameter("action_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.plan_debug_topic = str(self.get_parameter("plan_debug_topic").value)
        self.safety_topic = str(self.get_parameter("safety_topic").value)
        self.socket_timeout_s = float(self.get_parameter("socket_timeout_s").value)
        self.action_ttl_ms = int(self.get_parameter("action_ttl_ms").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.enable_adam = bool(self.get_parameter("enable_adam").value)
        self.enable_sharpa = bool(self.get_parameter("enable_sharpa").value)
        self.actor_send_hz = float(self.get_parameter("actor_send_hz").value)
        self.execution_mode = str(self.get_parameter("execution_mode").value).lower()
        self.sharpa_control_mode = str(
            self.get_parameter("sharpa_control_mode").value
        ).lower()
        self._validate_parameters()
        self.actor_period_s = 1.0 / self.actor_send_hz

        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.server_sock: socket.socket | None = None
        self.client_sock: socket.socket | None = None
        self.client_addr: tuple[str, int] | None = None
        self.listening = False
        self.connected = False
        self.accept_count = 0
        self.current_plan: JointPlan | None = None
        self.last_accepted_request_id = 0
        self.action_seq = 0
        self.last_action_seq: int | None = None
        self.last_action_time: float | None = None
        self.last_plan_time: float | None = None
        self.last_sent_plan_id: int | None = None
        self.last_sent_step: int | None = None
        self.plans_received = 0
        self.plans_rejected = 0
        self.plans_completed = 0
        self.done_published = 0
        self.action_published = 0
        self.frames_sent = 0
        self.frames_failed = 0
        self.idle_ticks = 0
        self.rows_skipped = 0
        self.scheduler_late_ticks = 0
        self.last_error = ""

        self.create_subscription(ActionPlan, self.action_plan_topic, self._on_plan, 10)
        self.done_pub = self.create_publisher(
            ExecutionDone, self.execution_done_topic, 10
        )
        self.action_pub = self.create_publisher(PndAction, self.action_topic, 10)
        self.status_pub = self.create_publisher(Status, self.status_topic, 10)
        self.plan_debug_pub = self.create_publisher(Status, self.plan_debug_topic, 10)
        self.safety_pub = self.create_publisher(Status, self.safety_topic, 10)
        self.create_timer(0.5, self._publish_status)

        self.server_worker = threading.Thread(target=self._serve, daemon=True)
        self.control_worker = threading.Thread(target=self._control_loop, daemon=True)
        self.server_worker.start()
        self.control_worker.start()
        self.get_logger().info(
            f"action_execute: plan={self.action_plan_topic}, done={self.execution_done_topic}, "
            f"action={self.action_topic}, send_hz={self.actor_send_hz:g}, "
            f"mode={self.execution_mode}"
        )

    def _validate_parameters(self) -> None:
        if not 1 <= self.listen_port <= 65535:
            raise ValueError("listen_port must be in [1, 65535]")
        if self.socket_timeout_s <= 0.0 or self.action_ttl_ms <= 0:
            raise ValueError("socket_timeout_s and action_ttl_ms must be positive")
        if self.actor_send_hz <= 0.0:
            raise ValueError("actor_send_hz must be positive")
        if self.execution_mode != "synchronous":
            raise ValueError("async execution mode is unsupported")
        if self.sharpa_control_mode != "position":
            raise ValueError("only SharpA position mode is supported")

    def _on_plan(self, msg: ActionPlan) -> None:
        try:
            plan = self._parse_plan(msg)
            with self.lock:
                if plan.request_id <= self.last_accepted_request_id:
                    return
                if self.current_plan is not None and not self.current_plan.completed:
                    raise ValueError("synchronous executor already has an active plan")
                self.last_accepted_request_id = plan.request_id
                self.current_plan = plan
                self.plans_received += 1
                self.last_plan_time = time.monotonic()
                self.last_error = ""
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.plans_rejected += 1
                self.last_error = f"action plan rejected: {exc}"
            self.get_logger().error(str(self.last_error))

    def _parse_plan(self, msg: ActionPlan) -> JointPlan:
        parsed = json_or_raw(msg.payload_json)
        payload = parsed.get("json") if parsed.get("valid") else None
        if not isinstance(payload, dict) or payload.get("schema") != "ws.action_plan.v1":
            raise ValueError("payload schema must be ws.action_plan.v1")
        plan_id = int(msg.plan_id)
        request_id = int(msg.request_id)
        request_stamp_ns = int(msg.request_stamp_ns)
        action_hz = float(msg.action_hz)
        if plan_id <= 0 or request_id <= 0 or request_stamp_ns <= 0 or action_hz <= 0.0:
            raise ValueError("plan IDs, request stamp, and action_hz must be positive")
        source_kind = str(payload.get("source_kind") or "policy")
        if source_kind not in {"policy", "replay"}:
            raise ValueError("source_kind must be policy or replay")
        if source_kind == "policy" and not np.isclose(action_hz, self.actor_send_hz):
            raise ValueError(
                f"policy action_hz={action_hz:g}, actor_send_hz={self.actor_send_hz:g}"
            )
        elapsed_ns = np.asarray(payload.get("elapsed_ns"), dtype=np.int64)
        if elapsed_ns.ndim != 1 or elapsed_ns.size == 0 or elapsed_ns[0] != 0:
            raise ValueError("elapsed_ns must be non-empty, one-dimensional, and start at zero")
        if np.any(np.diff(elapsed_ns) <= 0):
            raise ValueError("elapsed_ns must be strictly increasing")
        horizon = int(elapsed_ns.size)
        server_execution = self._server_execution_metadata(
            payload,
            horizon=horizon,
            action_hz=action_hz,
        )
        adam = payload.get("adam")
        sharpa = payload.get("sharpa")
        if not isinstance(adam, dict) or not isinstance(sharpa, dict):
            raise ValueError("plan missing adam/sharpa sections")
        if list(adam.get("joint_names") or []) != list(ADAM_COMMAND_JOINTS_19):
            raise ValueError("Adam joint order mismatch")
        if list(sharpa.get("joint_names") or []) != list(SHARPA_JOINT_NAMES):
            raise ValueError("SharpA joint order mismatch")
        adam_q = self._matrix(adam.get("q"), (horizon, 19), "adam.q")
        sharpa_q = self._matrix(sharpa.get("q"), (horizon, 44), "sharpa.q")
        selected = payload.get("selected_action_abs62")
        selected_abs62 = None
        if selected is not None:
            selected_abs62 = self._matrix(selected, (horizon, 62), "selected_action_abs62")
        debug = payload.get("debug")
        return JointPlan(
            plan_id=plan_id,
            request_id=request_id,
            request_stamp_ns=request_stamp_ns,
            source=msg.source,
            source_kind=source_kind,
            action_hz=action_hz,
            elapsed_ns=elapsed_ns,
            adam_q=adam_q,
            sharpa_q=sharpa_q,
            adam_valid=bool(adam.get("valid", True)),
            sharpa_valid=bool(sharpa.get("valid", True)),
            selected_abs62=selected_abs62,
            debug=debug if isinstance(debug, dict) else {},
            received_time=time.monotonic(),
            server_execution=server_execution,
        )

    @staticmethod
    def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
        return np.ascontiguousarray(array)

    @staticmethod
    def _server_execution_metadata(
        payload: dict[str, Any], *, horizon: int, action_hz: float
    ) -> dict[str, Any] | None:
        raw = payload.get("_ws_sharpa_v4")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("_ws_sharpa_v4 must be an object")
        action_id = raw.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("_ws_sharpa_v4.action_id must be nonempty")
        revision = raw.get("revision")
        execute_start = raw.get("execute_start")
        execute_length = raw.get("execute_length")
        action_length = raw.get("action_length")
        if isinstance(revision, bool) or not isinstance(revision, (int, np.integer)):
            raise ValueError("_ws_sharpa_v4.revision must be a non-negative integer")
        if isinstance(execute_start, bool) or not isinstance(
            execute_start, (int, np.integer)
        ):
            raise ValueError(
                "_ws_sharpa_v4.execute_start must be a non-negative integer"
            )
        if isinstance(execute_length, bool) or not isinstance(
            execute_length, (int, np.integer)
        ):
            raise ValueError("_ws_sharpa_v4.execute_length must be a positive integer")
        if isinstance(action_length, bool) or not isinstance(
            action_length, (int, np.integer)
        ):
            raise ValueError("_ws_sharpa_v4.action_length must be a positive integer")
        revision = int(revision)
        execute_start = int(execute_start)
        execute_length = int(execute_length)
        action_length = int(action_length)
        if revision < 0:
            raise ValueError("_ws_sharpa_v4.revision must be non-negative")
        if execute_start < 0:
            raise ValueError("_ws_sharpa_v4.execute_start must be non-negative")
        if execute_length <= 0 or action_length <= 0:
            raise ValueError("_ws_sharpa_v4 action lengths must be positive")
        frequency_hz = raw.get("frequency_hz")
        if isinstance(frequency_hz, bool) or not isinstance(
            frequency_hz, (int, float, np.integer, np.floating)
        ):
            raise ValueError("_ws_sharpa_v4.frequency_hz must be positive")
        frequency_hz = float(frequency_hz)
        if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("_ws_sharpa_v4.frequency_hz must be positive")
        if raw.get("server_driven_execution") is not True:
            raise ValueError("_ws_sharpa_v4.server_driven_execution must be true")
        if execute_start + execute_length > action_length:
            raise ValueError("_ws_sharpa_v4 execution slice exceeds action_length")
        if horizon != execute_length:
            raise ValueError(
                "action plan horizon must equal _ws_sharpa_v4.execute_length"
            )
        if not np.isclose(action_hz, frequency_hz):
            raise ValueError(
                "action_hz must match _ws_sharpa_v4.frequency_hz"
            )
        return {
            "action_id": action_id,
            "revision": revision,
            "execute_start": execute_start,
            "execute_length": execute_length,
            "action_length": action_length,
            "frequency_hz": frequency_hz,
            "server_driven_execution": True,
        }

    def _control_loop(self) -> None:
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            replay_due = self._replay_due_time()
            if replay_due is not None:
                now = time.monotonic()
                if now < replay_due:
                    self.stop_event.wait(min(0.01, replay_due - now))
                    continue
                self._tick()
                next_tick = time.monotonic() + self.actor_period_s
                continue
            now = time.monotonic()
            if now < next_tick:
                self.stop_event.wait(min(0.01, next_tick - now))
                continue
            self._tick()
            next_tick += self.actor_period_s
            finished = time.monotonic()
            if finished > next_tick:
                with self.lock:
                    self.scheduler_late_ticks += 1
                if finished - next_tick > self.actor_period_s:
                    skipped = int((finished - next_tick) / self.actor_period_s)
                    next_tick += skipped * self.actor_period_s

    def _replay_due_time(self) -> float | None:
        with self.lock:
            plan = self.current_plan
            if plan is None or plan.source_kind != "replay" or plan.completed:
                return None
            if plan.started_monotonic_ns is None:
                return time.monotonic()
            if plan.next_step >= plan.horizon:
                return None
            due_ns = plan.started_monotonic_ns + int(plan.elapsed_ns[plan.next_step])
        return due_ns / 1_000_000_000.0

    def _tick(self) -> None:
        with self.lock:
            if not self.dry_run and not self.connected:
                self.idle_ticks += 1
                return
            selection = self._select_locked()
            if selection is None:
                self.idle_ticks += 1
                return
            action_seq = self.action_seq + 1
        sent = self._publish_and_send(selection, action_seq)
        if not sent:
            self._finish_plan(selection.plan, success=False, error=self.last_error)
            return
        done: ExecutionDone | None = None
        with self.lock:
            self.action_seq = action_seq
            self.last_action_seq = action_seq
            self.last_action_time = time.monotonic()
            self.last_sent_plan_id = selection.plan.plan_id
            self.last_sent_step = selection.step_index
            done = self._commit_sent_locked(selection)
        if done is not None:
            self.done_pub.publish(done)

    def _select_locked(self) -> Selection | None:
        if self.current_plan is None:
            return None
        plan = self.current_plan
        if plan.source_kind == "replay":
            return self._select_replay_locked(plan)
        limit = plan.horizon
        if plan.next_step < limit:
            return Selection(plan, plan.next_step)
        return None

    def _select_replay_locked(self, plan: JointPlan) -> Selection | None:
        if plan.completed:
            return None
        current_ns = time.monotonic_ns()
        if plan.started_monotonic_ns is None:
            step_index = 0
        else:
            elapsed = current_ns - plan.started_monotonic_ns
            step_index = int(np.searchsorted(plan.elapsed_ns, elapsed, side="right") - 1)
            step_index = min(plan.horizon - 1, max(0, step_index))
            if step_index == plan.next_step - 1:
                return None
            if step_index < plan.next_step:
                return None
            if step_index > plan.next_step:
                self.rows_skipped += step_index - plan.next_step
        return Selection(plan, step_index)

    def _commit_sent_locked(self, selection: Selection) -> ExecutionDone | None:
        plan = selection.plan
        if plan.started_monotonic_ns is None:
            plan.started_monotonic_ns = time.monotonic_ns() - int(
                plan.elapsed_ns[selection.step_index]
            )
        plan.next_step = selection.step_index + 1
        limit = plan.horizon
        if plan.next_step >= limit and not plan.completed:
            plan.completed = True
            self.plans_completed += 1
            done = self._execution_done(plan, success=True, error="")
            self.done_published += 1
            if self.current_plan is plan:
                self.current_plan = None
            return done
        return None

    def _execution_done(
        self, plan: JointPlan, *, success: bool, error: str
    ) -> ExecutionDone:
        metadata = plan.server_execution or {}
        execute_start = int(metadata.get("execute_start", 0))
        execute_length = int(metadata.get("execute_length", plan.horizon))
        msg = ExecutionDone()
        set_header(msg, "action_execute", self.get_clock())
        msg.request_id = int(plan.request_id)
        msg.action_id = str(metadata.get("action_id") or f"plan-{plan.plan_id}")
        msg.revision = int(metadata.get("revision", 0))
        msg.execute_start = execute_start
        msg.execute_length = execute_length
        msg.executed_steps = execute_start + min(plan.next_step, execute_length)
        msg.success = bool(success)
        msg.done = True
        msg.error = str(error)
        return msg

    def _finish_plan(self, plan: JointPlan, *, success: bool, error: str) -> None:
        with self.lock:
            if plan.completed:
                return
            plan.completed = True
            done = self._execution_done(plan, success=success, error=error)
            self.done_published += 1
            if self.current_plan is plan:
                self.current_plan = None
        self.done_pub.publish(done)

    def _publish_and_send(self, selection: Selection, action_seq: int) -> bool:
        plan = selection.plan
        adam_q = plan.adam_q[selection.step_index]
        sharpa_q = plan.sharpa_q[selection.step_index]
        stamp_ns = now_ns()
        adam_valid = bool(self.enable_adam and plan.adam_valid and not self.dry_run)
        sharpa_valid = bool(self.enable_sharpa and plan.sharpa_valid and not self.dry_run)
        payload = {
            "schema": "pnd.deploy.action.v1",
            "seq": action_seq,
            "stamp_ns": stamp_ns,
            "ttl_ms": self.action_ttl_ms,
            "mode": "dry_run" if self.dry_run else f"{plan.source_kind}_joint_plan",
            "source": {
                "node": "action_execute",
                "kind": plan.source_kind,
                "name": plan.source,
                "plan_id": plan.plan_id,
                "request_id": plan.request_id,
            },
            "action_schema": "ws.action_plan.v1",
            "selected_action_step": selection.step_index,
            "execution": {
                "actor_send_hz": self.actor_send_hz,
                "plan_action_hz": plan.action_hz,
                "period_s": self.actor_period_s,
                "mode": "replay_timeline" if plan.source_kind == "replay" else self.execution_mode,
                "plan_horizon": plan.horizon,
            },
            "adam": {
                "valid": adam_valid,
                "q": adam_q.astype(float).tolist(),
                "source": "action_plan",
                "joint_names": list(ADAM_COMMAND_JOINTS_19),
            },
            "sharpa": {
                "valid": sharpa_valid,
                "control_mode": self.sharpa_control_mode,
                "q": sharpa_q.astype(float).tolist(),
                "dq": [0.0] * 44,
                "tau": [0.0] * 44,
                "source": "action_plan",
                "joint_names": list(SHARPA_JOINT_NAMES),
            },
            "implementation": {
                "scheduler": "action_execute",
                "kinematics": (
                    "upstream_action_ik"
                    if plan.source_kind == "policy"
                    or bool(plan.debug.get("posttrain_inverse_applied"))
                    else "not_required"
                ),
                "dry_run": self.dry_run,
            },
            "adam_valid": adam_valid,
            "sharpa_valid": sharpa_valid,
        }
        if plan.selected_abs62 is not None:
            payload["selected_action_abs62"] = plan.selected_abs62[
                selection.step_index
            ].astype(float).tolist()
        msg = PndAction()
        set_header(msg, "action_execute", self.get_clock())
        msg.seq = action_seq
        msg.stamp_ns = stamp_ns
        msg.payload_json = json_dumps(payload)
        self.action_pub.publish(msg)
        sent = True if self.dry_run else self._send_action_frame(
            json_bytes(payload), action_seq, stamp_ns
        )
        self.plan_debug_pub.publish(
            make_status(
                self.get_clock(),
                "action_execute",
                sent,
                {
                    "schema": "ws.action_execute.plan_debug.v1",
                    "action_seq": action_seq,
                    "plan_id": plan.plan_id,
                    "source_kind": plan.source_kind,
                    "step": selection.step_index,
                    "provider_debug": plan.debug,
                },
            )
        )
        self.safety_pub.publish(
            make_status(
                self.get_clock(),
                "action_execute",
                sent,
                {
                    "schema": "ws.action_execute.safety.v1",
                    "action_seq": action_seq,
                    "decision": "dry_run_no_send" if self.dry_run else ("allow" if sent else "send_failed"),
                    "safety_rules": ["typed_joint_order", "finite_values", "provider_limits"],
                },
            )
        )
        with self.lock:
            self.action_published += 1
            if sent:
                self.frames_sent += 1
                self.last_error = ""
            else:
                self.frames_failed += 1
        return sent

    def _serve(self) -> None:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(self.socket_timeout_s)
            sock.bind((self.listen_host, self.listen_port))
            sock.listen(1)
            with self.lock:
                self.server_sock = sock
                self.listening = True
            while not self.stop_event.is_set():
                try:
                    client, addr = sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._configure_tcp(client)
                with self.lock:
                    old = self.client_sock
                    self.client_sock = client
                    self.client_addr = addr
                    self.connected = True
                    self.accept_count += 1
                if old is not None:
                    old.close()
        except OSError as exc:
            with self.lock:
                self.last_error = f"tcp server failed: {exc}"
        finally:
            with self.lock:
                self.listening = False
                if self.server_sock is sock:
                    self.server_sock = None
            if sock is not None:
                sock.close()

    def _send_action_frame(self, payload: bytes, seq: int, stamp_ns: int) -> bool:
        with self.lock:
            sock = self.client_sock
        if sock is None:
            with self.lock:
                self.last_error = "no actor_node TCP client connected"
            return False
        try:
            with self.send_lock:
                frame_type = int(
                    getattr(deploy_protocol, "FRAME_TYPE_ACTION", FALLBACK_FRAME_TYPE_ACTION)
                    if deploy_protocol is not None
                    else FALLBACK_FRAME_TYPE_ACTION
                )
                if deploy_protocol is not None and hasattr(deploy_protocol, "send_frame"):
                    deploy_protocol.send_frame(sock, frame_type, payload, seq, stamp_ns)
                elif deploy_protocol is not None and hasattr(deploy_protocol, "pack_frame"):
                    sock.sendall(deploy_protocol.pack_frame(frame_type, payload, seq, stamp_ns))
                else:
                    sock.sendall(self._fallback_pack_frame(frame_type, payload, seq, stamp_ns))
            return True
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                if self.client_sock is sock:
                    self.client_sock = None
                    self.client_addr = None
                    self.connected = False
                self.last_error = f"action frame send failed: {exc}"
            try:
                sock.close()
            except OSError:
                pass
            return False

    def _configure_tcp(self, sock: socket.socket) -> None:
        if deploy_protocol is not None and hasattr(deploy_protocol, "configure_tcp"):
            deploy_protocol.configure_tcp(sock, timeout_s=None)
        else:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    @staticmethod
    def _fallback_pack_frame(frame_type: int, payload: bytes, seq: int, stamp_ns: int) -> bytes:
        checksum = zlib.crc32(payload) & 0xFFFFFFFF
        header = FALLBACK_HEADER_STRUCT.pack(
            FALLBACK_MAGIC,
            FALLBACK_VERSION,
            frame_type & 0xFF,
            0,
            seq & 0xFFFFFFFFFFFFFFFF,
            stamp_ns,
            len(payload),
            checksum,
        )
        return header + payload

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self.lock:
            current = self.current_plan
            payload = {
                "schema": "ws.action_execute.status.v2",
                "endpoint": {
                    "host": self.listen_host,
                    "port": self.listen_port,
                    "listening": self.listening,
                    "connected": self.connected,
                    "client": self.client_addr,
                    "accept_count": self.accept_count,
                },
                "topics": {
                    "action_plan": self.action_plan_topic,
                    "execution_done": self.execution_done_topic,
                    "action": self.action_topic,
                },
                "control": {
                    "dry_run": self.dry_run,
                    "actor_send_hz": self.actor_send_hz,
                    "execution_mode": self.execution_mode,
                },
                "plan": {
                    "current_id": current.plan_id if current else None,
                    "source_kind": current.source_kind if current else None,
                    "next_step": current.next_step if current else None,
                    "horizon": current.horizon if current else None,
                    "last_sent_id": self.last_sent_plan_id,
                    "last_sent_step": self.last_sent_step,
                    "last_plan_age_ms": age_ms(self.last_plan_time, now),
                },
                "counts": {
                    "plans_received": self.plans_received,
                    "plans_rejected": self.plans_rejected,
                    "plans_completed": self.plans_completed,
                    "done_published": self.done_published,
                    "action_published": self.action_published,
                    "frames_sent": self.frames_sent,
                    "frames_failed": self.frames_failed,
                    "idle_ticks": self.idle_ticks,
                    "rows_skipped": self.rows_skipped,
                    "scheduler_late_ticks": self.scheduler_late_ticks,
                },
                "last": {
                    "action_seq": self.last_action_seq,
                    "action_age_ms": age_ms(self.last_action_time, now),
                },
                "last_error": self.last_error,
            }
            ok = self.listening and not self.last_error
        self.status_pub.publish(make_status(self.get_clock(), "action_execute", ok, payload))

    def destroy_node(self) -> bool:
        self.stop_event.set()
        with self.lock:
            sockets = (self.server_sock, self.client_sock)
            self.server_sock = None
            self.client_sock = None
            self.connected = False
        for sock in sockets:
            if sock is None:
                continue
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self.server_worker.join(timeout=1.0)
        self.control_worker.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActionExecute()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
