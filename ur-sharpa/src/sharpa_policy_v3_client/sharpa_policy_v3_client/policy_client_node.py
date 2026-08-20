from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
import json
import math
import threading
import time
from typing import Any

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from sharpa_policy_v3_interfaces.msg import (
    ExecutionFeedback,
    PolicyActionV3,
    PolicyFault,
)
from sharpa_policy_v3_interfaces.srv import BuildObservation, ResetPolicy

from .action import ParsedPolicyActionV3
from .serialization import unpackb
from .session import PolicySessionRuntime
from .transport import (
    PolicyClosedError,
    PolicyConcurrencyError,
    PolicyDependencyError,
    PolicyHttpError,
    PolicyProtocolError,
    PolicyServerError,
    PolicyStateError,
    PolicyTransportError,
    PolicyV3Transport,
    PolicyWebSocketError,
)


FAULT_SCHEMA = "sharpa_policy_fault.v1"
FEEDBACK_SCHEMA = "sharpa_policy_execution_feedback.v1"


class StateObservationError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        self.code = str(code)
        self.retryable = bool(retryable)
        super().__init__(message)


class PolicyClientNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        self._declare_parameters()
        self.server_base_url = self._string_parameter("server_base_url", nonempty=True)
        self.prompt = self._string_parameter("prompt")
        configured_session_id = self._string_parameter("session_id")
        self.auto_reset = self._bool_parameter("auto_reset")
        self.require_execution_feedback = self._bool_parameter(
            "require_execution_feedback"
        )
        self.request_poll_hz = self._positive_float_parameter("request_poll_hz")
        self.connect_timeout_s = self._positive_float_parameter("connect_timeout_s")
        self.request_timeout_s = self._positive_float_parameter("request_timeout_s")
        self.reconnect_initial_s = self._positive_float_parameter("reconnect_initial_s")
        self.reconnect_max_s = self._positive_float_parameter("reconnect_max_s")
        if self.reconnect_max_s < self.reconnect_initial_s:
            raise ValueError("reconnect_max_s must be >= reconnect_initial_s")
        self.max_message_size = self._positive_int_parameter("max_message_size")
        self.transport = PolicyV3Transport(
            self.server_base_url,
            http_timeout_s=self.connect_timeout_s,
            connect_timeout_s=self.connect_timeout_s,
            inference_timeout_s=self.request_timeout_s,
            max_message_size=self.max_message_size,
        )
        self.runtime = PolicySessionRuntime(
            self.transport,
            session_id=configured_session_id or None,
            prompt=self.prompt,
            auto_reset=self.auto_reset,
        )

        self.action_pub = self.create_publisher(
            PolicyActionV3,
            self._string_parameter("action_topic", nonempty=True),
            10,
        )
        self.fault_pub = self.create_publisher(
            PolicyFault,
            self._string_parameter("fault_topic", nonempty=True),
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            self._string_parameter("status_topic", nonempty=True),
            10,
        )
        self.observation_client = self.create_client(
            BuildObservation,
            self._string_parameter("build_observation_service", nonempty=True),
        )
        self.feedback_sub = self.create_subscription(
            ExecutionFeedback,
            self._string_parameter("execution_feedback_topic", nonempty=True),
            self._on_execution_feedback,
            10,
        )
        self.reset_service = self.create_service(
            ResetPolicy,
            "/sharpa/v3/policy/reset",
            self._on_reset,
        )

        self._state_lock = threading.RLock()
        self._phase = "starting"
        self._connected = False
        self._inflight = False
        self._awaiting_feedback = False
        self._pending_action: ParsedPolicyActionV3 | None = None
        self._wire_feedback: dict[str, Any] | None = None
        self._last_error = ""
        self._closing = False
        self._epoch = 0
        self._bootstrap_future: Future[Any] | None = None
        self._recovery_future: Future[Any] | None = None
        self._infer_future: Future[Any] | None = None

        self._event_loop = asyncio.new_event_loop()
        self._event_thread = threading.Thread(
            target=self._run_event_loop,
            name="sharpa-policy-v3-asyncio",
            daemon=True,
        )
        self._event_thread.start()
        self.request_timer = self.create_timer(
            1.0 / self.request_poll_hz,
            self._request_tick,
        )
        self.status_timer = self.create_timer(1.0, self._publish_status)
        bootstrap_epoch = self._epoch
        self._bootstrap_future = asyncio.run_coroutine_threadsafe(
            self.runtime.start(),
            self._event_loop,
        )
        self._bootstrap_future.add_done_callback(
            lambda completed, request_epoch=bootstrap_epoch: self._on_bootstrap_done(
                completed,
                request_epoch,
            )
        )

    def _declare_parameters(self) -> None:
        parameters = {
            "server_base_url": "http://127.0.0.1:5500",
            "session_id": "",
            "prompt": "",
            "auto_reset": True,
            "request_poll_hz": 20.0,
            "require_execution_feedback": True,
            "connect_timeout_s": 5.0,
            "request_timeout_s": 90.0,
            "reconnect_initial_s": 0.5,
            "reconnect_max_s": 8.0,
            "max_message_size": 64 * 1024 * 1024,
            "build_observation_service": "/sharpa/v3/state/build_observation",
            "action_topic": "/sharpa/v3/policy/action",
            "execution_feedback_topic": ("/sharpa/v3/policy/execution_feedback"),
            "fault_topic": "/sharpa/v3/policy/fault",
            "status_topic": "/sharpa/v3/policy/status",
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_forever()

    def _on_bootstrap_done(
        self,
        future: Future[Any],
        request_epoch: int,
    ) -> None:
        with self._state_lock:
            if self._bootstrap_future is not future:
                return
            self._bootstrap_future = None
            if future.cancelled() or self._closing or request_epoch != self._epoch:
                return
        try:
            metadata = future.result()
        except Exception as exc:
            self._handle_async_failure(
                "bootstrap_failed",
                exc,
                expected_epoch=request_epoch,
            )
            return
        with self._state_lock:
            if self._closing or request_epoch != self._epoch:
                return
            self._connected = True
            self._phase = "ready"
            self._last_error = ""
        self.get_logger().info(
            "policy v3 connected: "
            f"family={metadata['policy_family']} "
            f"format={metadata['metadata_format']['format_id']} "
            f"session={self.runtime.session_id}"
        )

    def _request_tick(self) -> None:
        with self._state_lock:
            if (
                self._closing
                or not self._connected
                or self._inflight
                or (self.require_execution_feedback and self._awaiting_feedback)
            ):
                return
            if not self.observation_client.service_is_ready():
                self._phase = "waiting:state_service"
                return
            active_format = self.runtime.active_metadata_format
            if active_format is None:
                self._phase = "waiting:metadata"
                return
            feedback = dict(self._wire_feedback) if self._wire_feedback else None
            epoch = self._epoch
            snapshot = self.runtime.snapshot()
            request = BuildObservation.Request()
            request.metadata_format_json = json.dumps(
                active_format,
                separators=(",", ":"),
                allow_nan=False,
            )
            request.session_id = snapshot.session_id
            request.request_id = snapshot.request_id
            request.timestamp_ns = time.time_ns()
            request.prompt = self.prompt
            request.execution_feedback_json = (
                json.dumps(feedback, separators=(",", ":"), allow_nan=False)
                if feedback is not None
                else ""
            )
            request.max_message_size = self.transport.effective_message_size
            future = self.observation_client.call_async(request)
            self._inflight = True
            self._phase = "building_observation"
            self._infer_future = future
        future.add_done_callback(
            lambda completed, request_epoch=epoch: self._on_observation_done(
                completed,
                request_epoch,
            )
        )

    def _on_observation_done(self, future: Future[Any], request_epoch: int) -> None:
        with self._state_lock:
            if self._infer_future is not future:
                return
            if future.cancelled() or self._closing or request_epoch != self._epoch:
                self._clear_inference_locked(future)
                return
        try:
            response = future.result()
        except Exception as exc:
            with self._state_lock:
                if self._infer_future is not future:
                    return
                self._clear_inference_locked(future)
                if self._closing or request_epoch != self._epoch:
                    return
                self._phase = "waiting:state_service"
                self._last_error = str(exc)
            return
        if not response.success:
            error = StateObservationError(
                response.error_code or "observation_build_failed",
                response.error_message or "state node did not build an observation",
                retryable=bool(response.retryable),
            )
            if error.retryable:
                with self._state_lock:
                    if self._infer_future is not future:
                        return
                    self._clear_inference_locked(future)
                    if self._closing or request_epoch != self._epoch:
                        return
                    self._phase = f"waiting:{error.code}"
                    self._last_error = str(error)
                return
            self._handle_async_failure(
                error.code,
                error,
                expected_epoch=request_epoch,
                source_future=future,
            )
            return
        try:
            observation = unpackb(
                bytes(response.observation_msgpack),
                max_size=self.transport.effective_message_size,
            )
            inference = self.runtime.infer_observation(observation)
            inference_future = asyncio.run_coroutine_threadsafe(
                inference,
                self._event_loop,
            )
        except Exception as exc:
            if "inference" in locals():
                inference.close()
            self._handle_async_failure(
                "invalid_state_observation",
                exc,
                expected_epoch=request_epoch,
                source_future=future,
            )
            return
        with self._state_lock:
            if self._infer_future is not future:
                inference_future.cancel()
                return
            if self._closing or request_epoch != self._epoch:
                self._clear_inference_locked(future)
                inference_future.cancel()
                return
            self._infer_future = inference_future
            self._phase = "inflight"
        inference_future.add_done_callback(
            lambda completed, request_epoch=request_epoch: self._on_inference_done(
                completed,
                request_epoch,
            )
        )

    def _on_inference_done(self, future: Future[Any], request_epoch: int) -> None:
        with self._state_lock:
            if self._infer_future is not future:
                return
            if future.cancelled() or self._closing or request_epoch != self._epoch:
                self._clear_inference_locked(future)
                return
        try:
            action = future.result()
        except Exception as exc:
            self._handle_async_failure(
                "inference_failed",
                exc,
                expected_epoch=request_epoch,
                source_future=future,
            )
            return
        try:
            action_message = self._action_message(action)
        except (OverflowError, TypeError, ValueError) as exc:
            self._handle_async_failure(
                "action_ros_conversion_failed",
                exc,
                expected_epoch=request_epoch,
                source_future=future,
                correlated_action=action,
            )
            return
        with self._state_lock:
            if self._infer_future is not future:
                return
            if self._closing or not self._connected or request_epoch != self._epoch:
                self._clear_inference_locked(future)
                return
            self._pending_action = action
            if self.require_execution_feedback:
                self._awaiting_feedback = True
                self._phase = "awaiting_execution_feedback"
            else:
                self._wire_feedback = {
                    "last_action_id": action.action_id,
                    "executed_steps": 0,
                    "success": False,
                }
                self._awaiting_feedback = False
                self._phase = "ready"
            try:
                self.action_pub.publish(action_message)
            except Exception as exc:
                self._handle_async_failure(
                    "action_ros_publish_failed",
                    exc,
                    expected_epoch=request_epoch,
                    source_future=future,
                )
                return
            self._clear_inference_locked(future)

    def _clear_inference_locked(self, future: Future[Any]) -> None:
        if self._infer_future is future:
            self._infer_future = None
            self._inflight = False

    @staticmethod
    def _exception_retryable(exception: Exception) -> bool:
        if isinstance(exception, StateObservationError):
            return exception.retryable
        if isinstance(exception, PolicyServerError):
            return exception.retryable
        if isinstance(
            exception,
            (
                PolicyProtocolError,
                PolicyClosedError,
                PolicyConcurrencyError,
                PolicyDependencyError,
                PolicyStateError,
            ),
        ):
            return False
        if isinstance(exception, FutureTimeoutError):
            return True
        if isinstance(exception, PolicyHttpError):
            return (
                exception.status == 0
                or exception.status in {408, 409, 425, 429}
                or 500 <= exception.status < 600
            )
        if isinstance(exception, PolicyWebSocketError):
            return True
        if isinstance(exception, PolicyTransportError):
            return False
        return False

    def _handle_async_failure(
        self,
        code: str,
        exception: Exception,
        *,
        expected_epoch: int | None = None,
        source_future: Future[Any] | None = None,
        correlated_action: ParsedPolicyActionV3 | None = None,
    ) -> bool:
        retryable = self._exception_retryable(exception)
        with self._state_lock:
            if source_future is not None and self._infer_future is not source_future:
                return retryable
            if self._closing or (
                expected_epoch is not None and expected_epoch != self._epoch
            ):
                if source_future is not None:
                    self._clear_inference_locked(source_future)
                return retryable
            if source_future is not None:
                self._clear_inference_locked(source_future)
            action = correlated_action or self._pending_action
            request_id = (
                exception.request_id
                if isinstance(exception, PolicyServerError)
                else None
            )
            if action is None and request_id is None and source_future is not None:
                request_id = self.runtime.snapshot().request_id
            self._connected = False
            self._inflight = False
            self._awaiting_feedback = False
            self._pending_action = None
            self._phase = "fault"
            self._last_error = str(exception)
            self._publish_fault(
                code=code,
                message=str(exception),
                retryable=retryable,
                safe_stop=True,
                clear_plan=True,
                correlated_action=action,
                request_id=request_id,
            )
            if retryable:
                self._schedule_recovery(expected_epoch=expected_epoch)
            else:
                self._schedule_terminal_disconnect(expected_epoch=expected_epoch)
        return retryable

    def _schedule_terminal_disconnect(
        self,
        *,
        expected_epoch: int | None = None,
    ) -> None:
        with self._state_lock:
            if self._closing or (
                expected_epoch is not None and expected_epoch != self._epoch
            ):
                return
            if self._recovery_future is not None and not self._recovery_future.done():
                return
            disconnect = self.runtime.disconnect()
            try:
                future = asyncio.run_coroutine_threadsafe(
                    disconnect,
                    self._event_loop,
                )
            except Exception:
                disconnect.close()
                return
            self._recovery_future = future
        future.add_done_callback(self._on_recovery_done)

    def _schedule_recovery(self, *, expected_epoch: int | None = None) -> None:
        with self._state_lock:
            if self._closing or (
                expected_epoch is not None and expected_epoch != self._epoch
            ):
                return
            if self._recovery_future is not None and not self._recovery_future.done():
                return
            recovery_epoch = self._epoch
            recovery = self._recover_loop(recovery_epoch)
            try:
                future = asyncio.run_coroutine_threadsafe(
                    recovery,
                    self._event_loop,
                )
            except Exception as exc:
                recovery.close()
                self._phase = "fault"
                self._last_error = str(exc)
                return
            self._recovery_future = future
        future.add_done_callback(self._on_recovery_done)

    def _on_recovery_done(self, future: Future[Any]) -> None:
        with self._state_lock:
            if self._recovery_future is future:
                self._recovery_future = None

    async def _recover_loop(self, recovery_epoch: int) -> None:
        delay = self.reconnect_initial_s
        while True:
            with self._state_lock:
                if self._closing or recovery_epoch != self._epoch:
                    return
            await asyncio.sleep(delay)
            with self._state_lock:
                if self._closing or recovery_epoch != self._epoch:
                    return
            try:
                await self.runtime.reconnect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retryable = self._exception_retryable(exc)
                with self._state_lock:
                    if self._closing or recovery_epoch != self._epoch:
                        return
                    self._connected = False
                    self._phase = "reconnecting" if retryable else "fault"
                    self._last_error = str(exc)
                    self._publish_fault(
                        code="reconnect_failed",
                        message=str(exc),
                        retryable=retryable,
                        safe_stop=True,
                        clear_plan=True,
                    )
                if not retryable:
                    return
                delay = min(delay * 2.0, self.reconnect_max_s)
                continue
            with self._state_lock:
                if self._closing or recovery_epoch != self._epoch:
                    return
                self._connected = True
                self._phase = "ready"
                self._last_error = ""
                self._wire_feedback = None
                session_id = self.runtime.session_id
            self.get_logger().info(
                f"policy v3 reconnected with new session {session_id}"
            )
            return

    def _on_execution_feedback(self, message: ExecutionFeedback) -> None:
        try:
            with self._state_lock:
                if self._closing:
                    return
                feedback = self._validated_feedback(message)
                self._wire_feedback = feedback
                self._awaiting_feedback = False
                self._pending_action = None
                self._phase = "ready"
        except ValueError as exc:
            self._publish_fault(
                code="invalid_execution_feedback",
                message=str(exc),
                retryable=True,
                safe_stop=True,
                clear_plan=True,
            )
            return

    def _validated_feedback(self, message: ExecutionFeedback) -> dict[str, Any]:
        if message.schema != FEEDBACK_SCHEMA:
            raise ValueError(f"feedback schema must be {FEEDBACK_SCHEMA}")
        if not message.has_action:
            raise ValueError("feedback must identify an action")
        with self._state_lock:
            action = self._pending_action
        if action is None:
            raise ValueError("no policy action is awaiting feedback")
        expected = (
            (message.session_id, action.session_id, "session_id"),
            (int(message.request_id), action.request_id, "request_id"),
            (message.action_id, action.action_id, "action_id"),
            (int(message.revision), action.revision, "revision"),
            (
                int(message.execute_start),
                action.execution.execute_start,
                "execute_start",
            ),
            (
                int(message.execute_length),
                action.execution.execute_length,
                "execute_length",
            ),
        )
        for actual, wanted, field in expected:
            if actual != wanted:
                raise ValueError(f"feedback {field} does not match pending action")
        executed_steps = int(message.executed_steps)
        if executed_steps > action.execution.execute_length:
            raise ValueError("feedback executed_steps exceeds execute_length")
        if message.success and executed_steps != action.execution.execute_length:
            raise ValueError("successful feedback must report the full execute_length")
        return {
            "last_action_id": action.action_id,
            "executed_steps": executed_steps,
            "success": bool(message.success),
        }

    def _on_reset(
        self,
        request: ResetPolicy.Request,
        response: ResetPolicy.Response,
    ) -> ResetPolicy.Response:
        if int(request.request_id) != 0:
            fault = self._fault_message(
                code="invalid_reset_request",
                message="reset request_id must be 0",
                retryable=False,
                safe_stop=True,
                clear_plan=True,
            )
            return self._reset_fault_response(response, request, fault)
        if request.session_id and request.session_id == self.runtime.session_id:
            fault = self._fault_message(
                code="invalid_reset_request",
                message="new session_id must differ from the active session",
                retryable=False,
                safe_stop=True,
                clear_plan=True,
            )
            return self._reset_fault_response(response, request, fault)
        with self._state_lock:
            self._epoch += 1
            reset_epoch = self._epoch
            self._connected = False
            self._phase = "resetting"
            bootstrap_future = self._bootstrap_future
            infer_future = self._infer_future
            recovery_future = self._recovery_future
            self._bootstrap_future = None
            self._infer_future = None
            self._recovery_future = None
            self._inflight = False
            self._awaiting_feedback = False
            self._pending_action = None
            self._wire_feedback = None
        for active_future in (
            bootstrap_future,
            infer_future,
            recovery_future,
        ):
            if active_future is not None:
                active_future.cancel()
        reset_operation = self.runtime.reset(request.session_id or None)
        try:
            future = asyncio.run_coroutine_threadsafe(
                reset_operation,
                self._event_loop,
            )
        except Exception as exc:
            reset_operation.close()
            retryable = self._exception_retryable(exc)
            self._handle_async_failure(
                "reset_failed",
                exc,
                expected_epoch=reset_epoch,
            )
            fault = self._fault_message(
                code="reset_failed",
                message=str(exc),
                retryable=retryable,
                safe_stop=True,
                clear_plan=True,
            )
            return self._reset_fault_response(response, request, fault)
        try:
            result = future.result(timeout=self.request_timeout_s)
        except Exception as exc:
            future.cancel()
            retryable = self._exception_retryable(exc)
            self._handle_async_failure(
                "reset_failed",
                exc,
                expected_epoch=reset_epoch,
            )
            fault = self._fault_message(
                code="reset_failed",
                message=str(exc),
                retryable=retryable,
                safe_stop=True,
                clear_plan=True,
            )
            return self._reset_fault_response(response, request, fault)
        with self._state_lock:
            if self._closing or reset_epoch != self._epoch:
                fault = self._fault_message(
                    code="reset_superseded",
                    message="reset completed after a newer lifecycle transition",
                    retryable=False,
                    safe_stop=True,
                    clear_plan=True,
                )
                return self._reset_fault_response(response, request, fault)
            self._connected = True
            self._awaiting_feedback = False
            self._pending_action = None
            self._wire_feedback = None
            self._phase = "ready"
            self._last_error = ""
        response.schema = result["schema"]
        response.session_id = result["session_id"]
        response.request_id = result["request_id"]
        response.reset = result["reset"]
        response.metadata_format_id = result["metadata_format"]["format_id"]
        response.has_fault = False
        return response

    @staticmethod
    def _reset_fault_response(
        response: ResetPolicy.Response,
        request: ResetPolicy.Request,
        fault: PolicyFault,
    ) -> ResetPolicy.Response:
        response.schema = "sharpa_policy_reset.v1"
        response.session_id = request.session_id
        response.request_id = request.request_id
        response.reset = False
        response.metadata_format_id = ""
        response.has_fault = True
        response.fault = fault
        return response

    def _action_message(self, action: ParsedPolicyActionV3) -> PolicyActionV3:
        self._require_uint(action.request_id, 64, "request_id")
        self._require_uint(action.revision, 64, "revision")
        self._require_uint(action.timestamp_ns, 64, "timestamp_ns")
        self._require_uint(action.execution.action_length, 32, "action_length")
        self._require_uint(action.execution.execute_start, 32, "execute_start")
        self._require_uint(action.execution.execute_length, 32, "execute_length")
        message = PolicyActionV3()
        message.schema = action.schema
        message.session_id = action.session_id
        message.request_id = action.request_id
        message.action_id = action.action_id
        message.revision = action.revision
        message.timestamp_ns = action.timestamp_ns
        message.frequency_hz = action.execution.frequency_hz
        message.action_length = action.execution.action_length
        message.execute_start = action.execution.execute_start
        message.execute_length = action.execution.execute_length
        message.left_wrist_action_type = action.left_wrist_action_type
        message.left_wrist_dimension = action.left_wrist_dimension
        message.left_wrist = action.left_wrist.reshape(-1).tolist()
        message.right_wrist_action_type = action.right_wrist_action_type
        message.right_wrist_dimension = action.right_wrist_dimension
        message.right_wrist = action.right_wrist.reshape(-1).tolist()
        message.hand_joint_dimension = int(action.hand_joint.shape[1])
        message.hand_joint = action.hand_joint.reshape(-1).tolist()
        message.diagnostics_policy_family = action.diagnostics.policy_family
        message.diagnostics_checkpoint_id = action.diagnostics.checkpoint_id
        message.diagnostics_checkpoint_path = action.diagnostics.checkpoint_path
        message.diagnostics_inference_latency_ms = (
            action.diagnostics.inference_latency_ms
        )
        message.has_next_metadata_format = action.next_metadata_format is not None
        if action.next_metadata_format is not None:
            message.next_metadata_format_id = self.runtime.active_metadata_format[
                "format_id"
            ]
        return message

    def _publish_fault(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        safe_stop: bool,
        clear_plan: bool,
        correlated_action: ParsedPolicyActionV3 | None = None,
        request_id: int | None = None,
    ) -> None:
        fault = self._fault_message(
            code=code,
            message=message,
            retryable=retryable,
            safe_stop=safe_stop,
            clear_plan=clear_plan,
            correlated_action=correlated_action,
            request_id=request_id,
        )
        self.fault_pub.publish(fault)
        self.get_logger().error(f"{code}: {message}")

    def _fault_message(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        safe_stop: bool,
        clear_plan: bool,
        correlated_action: ParsedPolicyActionV3 | None = None,
        request_id: int | None = None,
    ) -> PolicyFault:
        fault = PolicyFault()
        fault.schema = FAULT_SCHEMA
        fault.session_id = self.runtime.session_id
        action = correlated_action
        if action is None:
            with self._state_lock:
                action = self._pending_action
        if action is not None:
            fault.has_request_id = True
            fault.request_id = action.request_id
            fault.has_action_id = True
            fault.action_id = action.action_id
            fault.has_revision = True
            fault.revision = action.revision
        elif request_id is not None:
            fault.has_request_id = True
            fault.request_id = request_id
        fault.code = code
        fault.message = message
        fault.retryable = retryable
        fault.safe_stop = safe_stop
        fault.clear_plan = clear_plan
        fault.timestamp_ns = time.time_ns()
        return fault

    def _publish_status(self) -> None:
        snapshot = self.runtime.snapshot()
        with self._state_lock:
            payload = {
                "schema": "sharpa_policy_client_status.v1",
                "phase": self._phase,
                "connected": self._connected,
                "inflight": self._inflight,
                "awaiting_execution_feedback": self._awaiting_feedback,
                "session_id": snapshot.session_id,
                "next_request_id": snapshot.request_id,
                "metadata_format_id": snapshot.format_id,
                "last_action_id": snapshot.last_action_id,
                "last_error": self._last_error,
                "controller_output_enabled": False,
                "state_service_ready": self.observation_client.service_is_ready(),
                "timestamp_ns": time.time_ns(),
            }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.status_pub.publish(message)

    @staticmethod
    def _require_uint(value: int, bits: int, field: str) -> None:
        if value < 0 or value >= 1 << bits:
            raise OverflowError(f"{field} does not fit uint{bits}")

    def _string_parameter(self, name: str, *, nonempty: bool = False) -> str:
        value = self.get_parameter(name).value
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if nonempty and not value:
            raise ValueError(f"{name} must not be empty")
        return value

    def _bool_parameter(self, name: str) -> bool:
        value = self.get_parameter(name).value
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
        return value

    def _positive_float_parameter(self, name: str) -> float:
        value = self.get_parameter(name).value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        output = float(value)
        if not math.isfinite(output) or output <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return output

    def _positive_int_parameter(self, name: str) -> int:
        value = self.get_parameter(name).value
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    def destroy_node(self) -> bool:
        with self._state_lock:
            self._closing = True
            self._epoch += 1
            self._connected = False
            self._phase = "closing"
            bootstrap_future = self._bootstrap_future
            infer_future = self._infer_future
            recovery_future = self._recovery_future
            self._bootstrap_future = None
            self._infer_future = None
            self._recovery_future = None
            self._inflight = False
            self._awaiting_feedback = False
            self._pending_action = None
            self._wire_feedback = None
        self.request_timer.cancel()
        self.status_timer.cancel()
        for active_future in (
            bootstrap_future,
            infer_future,
            recovery_future,
        ):
            if active_future is not None:
                active_future.cancel()
        if self._event_loop.is_running():
            close_future = asyncio.run_coroutine_threadsafe(
                self.runtime.close(),
                self._event_loop,
            )
            try:
                close_future.result(timeout=self.connect_timeout_s)
            except (Exception, KeyboardInterrupt):
                close_future.cancel()
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            try:
                self._event_thread.join(timeout=self.connect_timeout_s)
            except KeyboardInterrupt:
                pass
        if not self._event_loop.is_running():
            try:
                self._event_loop.close()
            except KeyboardInterrupt:
                pass
        return super().destroy_node()


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node = PolicyClientNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
