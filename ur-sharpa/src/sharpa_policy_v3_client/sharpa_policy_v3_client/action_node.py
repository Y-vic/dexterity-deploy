from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from sharpa_policy_v3_interfaces.msg import (
    ArmCommand,
    ExecutionFeedback,
    HandCommand,
    HardwareCommandResult,
    PolicyActionV3,
    PolicyFault,
    UrStateFrame,
)

from .hardware_execution import ActionExecutionError, prepare_executable_action


FEEDBACK_SCHEMA = "sharpa_policy_execution_feedback.v1"
EXECUTION_CONFIRMATION = "ENABLE_DUAL_UR_SHARPA_POLICY_EXECUTION"


class ActionNode(Node):
    def __init__(self) -> None:
        super().__init__("action_node")
        self.declare_parameter("enable_execution", False)
        self.declare_parameter("execution_confirmation", "")
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("action_topic", "/sharpa/v3/policy/action")
        self.declare_parameter(
            "feedback_topic", "/sharpa/v3/policy/execution_feedback"
        )
        self.declare_parameter("fault_topic", "/sharpa/v3/policy/fault")
        self.declare_parameter("ur_state_topic", "/ur_position")
        self.declare_parameter("ur_command_topic", "/sharpa/v3/hardware/ur/command")
        self.declare_parameter(
            "sharpa_command_topic", "/sharpa/v3/hardware/sharpa/command"
        )
        self.declare_parameter(
            "result_topic", "/sharpa/v3/hardware/command_result"
        )
        self.declare_parameter("stop_topic", "/sharpa/v3/hardware/stop")
        self.enabled = bool(self.get_parameter("enable_execution").value)
        confirmation = str(self.get_parameter("execution_confirmation").value)
        if self.enabled and confirmation != EXECUTION_CONFIRMATION:
            raise ValueError(
                "action execution requested without the exact execution confirmation"
            )
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        if self.command_timeout_s <= 0.0:
            raise ValueError("command_timeout_s must be positive")

        self.feedback_pub = self.create_publisher(
            ExecutionFeedback,
            str(self.get_parameter("feedback_topic").value),
            10,
        )
        self.arm_pub = self.create_publisher(
            ArmCommand,
            str(self.get_parameter("ur_command_topic").value),
            10,
        )
        self.hand_pub = self.create_publisher(
            HandCommand,
            str(self.get_parameter("sharpa_command_topic").value),
            10,
        )
        self.stop_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("stop_topic").value),
            10,
        )
        self.action_sub = self.create_subscription(
            PolicyActionV3,
            str(self.get_parameter("action_topic").value),
            self._on_action,
            10,
        )
        self.ur_state_sub = self.create_subscription(
            UrStateFrame,
            str(self.get_parameter("ur_state_topic").value),
            self._on_ur_state,
            10,
        )
        self.result_sub = self.create_subscription(
            HardwareCommandResult,
            str(self.get_parameter("result_topic").value),
            self._on_result,
            20,
        )
        self.fault_sub = self.create_subscription(
            PolicyFault,
            str(self.get_parameter("fault_topic").value),
            self._on_fault,
            10,
        )
        self._condition = threading.Condition()
        self._latest_ur: UrStateFrame | None = None
        self._results: dict[tuple[str, int, int], dict[str, HardwareCommandResult]] = {}
        self._active_thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.get_logger().info(f"action coordinator started; execution={self.enabled}")

    def _on_ur_state(self, message: UrStateFrame) -> None:
        with self._condition:
            self._latest_ur = message

    def _on_result(self, message: HardwareCommandResult) -> None:
        if message.device not in {"ur", "sharpa"}:
            return
        key = (message.action_id, int(message.revision), int(message.step_index))
        with self._condition:
            self._results.setdefault(key, {})[message.device] = message
            self._condition.notify_all()

    def _on_fault(self, message: PolicyFault) -> None:
        if message.safe_stop:
            self._cancel.set()
            self._publish_stop()

    def _on_action(self, message: PolicyActionV3) -> None:
        with self._condition:
            busy = self._active_thread is not None and self._active_thread.is_alive()
            ur_state = self._latest_ur
        if busy:
            self._publish_feedback(message, 0, False, "action_busy", "another action is active")
            return
        if not self.enabled:
            self._publish_feedback(
                message,
                0,
                False,
                "execution_disabled",
                "hardware execution is disabled",
            )
            return
        try:
            executable = self._prepare(message, ur_state)
        except Exception as exc:
            self._publish_feedback(message, 0, False, "invalid_action", str(exc))
            return
        self._cancel.clear()
        thread = threading.Thread(
            target=self._execute,
            args=(message, executable),
            name=f"action-{message.action_id}",
            daemon=True,
        )
        with self._condition:
            self._active_thread = thread
        thread.start()

    @staticmethod
    def _prepare(message: PolicyActionV3, ur_state: UrStateFrame | None) -> Any:
        if ur_state is None or not ur_state.valid or not ur_state.normal_mode:
            raise ActionExecutionError("fresh normal-mode UR state is required")
        if ur_state.eef_dimension != 9:
            raise ActionExecutionError("UR state EEF dimension must be 9")
        if len(ur_state.left_eef) != 9 or len(ur_state.right_eef) != 9:
            raise ActionExecutionError("UR state must contain both EEF poses")
        action_length = int(message.action_length)
        left = np.asarray(message.left_wrist, dtype=np.float32)
        right = np.asarray(message.right_wrist, dtype=np.float32)
        hands = np.asarray(message.hand_joint, dtype=np.float32)
        if left.size != action_length * int(message.left_wrist_dimension):
            raise ActionExecutionError("left wrist action shape is inconsistent")
        if right.size != action_length * int(message.right_wrist_dimension):
            raise ActionExecutionError("right wrist action shape is inconsistent")
        if hands.size != action_length * int(message.hand_joint_dimension):
            raise ActionExecutionError("hand action shape is inconsistent")
        left = left.reshape(action_length, int(message.left_wrist_dimension))
        right = right.reshape(action_length, int(message.right_wrist_dimension))
        hands = hands.reshape(action_length, int(message.hand_joint_dimension))
        return prepare_executable_action(
            action_id=message.action_id,
            frequency_hz=float(message.frequency_hz),
            action_length=action_length,
            execute_start=int(message.execute_start),
            execute_length=int(message.execute_length),
            left_wrist_action_type=message.left_wrist_action_type,
            right_wrist_action_type=message.right_wrist_action_type,
            left_wrist=left,
            right_wrist=right,
            hand_joint=hands,
            current_left_wire=np.asarray(ur_state.left_eef, dtype=np.float32),
            current_right_wire=np.asarray(ur_state.right_eef, dtype=np.float32),
        )

    def _execute(self, source: PolicyActionV3, action: Any) -> None:
        executed = 0
        failure_code = ""
        failure_message = ""
        period_s = 1.0 / action.frequency_hz
        next_deadline = time.monotonic()
        try:
            for offset in range(action.execute_length):
                if self._cancel.is_set():
                    raise ActionExecutionError("action execution cancelled")
                step_index = action.execute_start + offset
                key = (source.action_id, int(source.revision), step_index)
                with self._condition:
                    self._results.pop(key, None)
                arm = ArmCommand()
                arm.action_id = source.action_id
                arm.revision = source.revision
                arm.step_index = step_index
                arm.period_s = period_s
                arm.eef_dimension = 9
                arm.left_eef = action.left_wrist[offset].tolist()
                arm.right_eef = action.right_wrist[offset].tolist()
                hand = HandCommand()
                hand.action_id = source.action_id
                hand.revision = source.revision
                hand.step_index = step_index
                hand.joint_dimension = 22
                hand.left_joint = action.hand_joint[offset, :22].tolist()
                hand.right_joint = action.hand_joint[offset, 22:].tolist()
                self.arm_pub.publish(arm)
                self.hand_pub.publish(hand)
                results = self._wait_results(key)
                for device in ("ur", "sharpa"):
                    result = results[device]
                    if not result.success:
                        raise ActionExecutionError(
                            f"{device}: {result.failure_code}: {result.failure_message}"
                        )
                executed += 1
                next_deadline += period_s
                remaining = next_deadline - time.monotonic()
                if remaining > 0.0 and self._cancel.wait(remaining):
                    raise ActionExecutionError("action execution cancelled")
        except Exception as exc:
            failure_code = "execution_cancelled" if self._cancel.is_set() else "execution_failed"
            failure_message = str(exc)
            self._publish_stop()
        success = executed == action.execute_length and not failure_code
        self._publish_feedback(
            source,
            executed,
            success,
            failure_code,
            failure_message,
        )
        with self._condition:
            self._active_thread = None

    def _wait_results(
        self,
        key: tuple[str, int, int],
    ) -> dict[str, HardwareCommandResult]:
        deadline = time.monotonic() + self.command_timeout_s
        with self._condition:
            while True:
                results = self._results.get(key, {})
                if set(results) == {"ur", "sharpa"}:
                    return self._results.pop(key)
                if self._cancel.is_set():
                    raise ActionExecutionError("action execution cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = sorted({"ur", "sharpa"}.difference(results))
                    raise ActionExecutionError(
                        f"hardware command timeout; missing {','.join(missing)} ACK"
                    )
                self._condition.wait(remaining)

    def _publish_stop(self) -> None:
        message = Bool()
        message.data = True
        self.stop_pub.publish(message)

    def _publish_feedback(
        self,
        action: PolicyActionV3,
        executed_steps: int,
        success: bool,
        failure_code: str,
        failure_message: str,
    ) -> None:
        feedback = ExecutionFeedback()
        feedback.schema = FEEDBACK_SCHEMA
        feedback.session_id = action.session_id
        feedback.has_action = True
        feedback.request_id = action.request_id
        feedback.action_id = action.action_id
        feedback.revision = action.revision
        feedback.execute_start = action.execute_start
        feedback.execute_length = action.execute_length
        feedback.executed_steps = executed_steps
        feedback.success = success
        feedback.failure_code = failure_code
        feedback.failure_message = failure_message
        feedback.timestamp_ns = time.time_ns()
        self.feedback_pub.publish(feedback)

    def destroy_node(self) -> bool:
        self._cancel.set()
        self._publish_stop()
        thread = self._active_thread
        if thread is not None:
            thread.join(timeout=2.0)
        return super().destroy_node()


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node: ActionNode | None = None
    try:
        node = ActionNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
