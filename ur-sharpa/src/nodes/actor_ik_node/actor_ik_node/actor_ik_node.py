"""Policy action slice to UR5 joint-plan conversion.

The core is deliberately independent of ROS so malformed actions and identity
handling can be tested without a running graph. The ROS wrapper only translates
messages and publishes one terminal ExecutionDone for every received action.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


ARM_MAX_STEP_RAD = 0.25
HAND_MAX_STEP_RAD = 0.4


@dataclass(frozen=True)
class IkPlan:
    left_joint: np.ndarray
    right_joint: np.ndarray
    hand_joint: np.ndarray


class ActorIkError(ValueError):
    pass


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
    from ur_msgs.msg import ArmCommand, ControlStatus, ExecutionDone, ExecutableAction, HandCommand
    from ur_msgs.msg import HandStateFrame, UrStateFrame
    from ur_common.kinematics import UrCommonKinematics
    from ur_common.geometry import CANONICAL_EEF_FRAME
    from ur_common.safety import SafetyConfig

    class RosActorIkNode(Node):
        def __init__(self):
            super().__init__("actor_ik_node")
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
            self._pending = {}
            self._active = None
            self._scheduled = None
            self._execution_provider = "policy"
            self._machine_running = False
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
                if provider != self._execution_provider and self._active is not None:
                    self._abort_active("execution provider changed")
                self._execution_provider = provider

        def _control_status(self, message):
            running = bool(
                message.ur_state == ControlStatus.UR_RUNNING
                and message.sharpa_state == ControlStatus.SHARPA_RUNNING
            )
            if self._machine_running and not running:
                self._abort_active("machine execution gate closed")
            self._machine_running = running

        def _abort_active(self, error):
            action = None
            step = 0
            if self._scheduled is not None:
                action = self._scheduled["action"]
                step = int(self._scheduled["step"])
            elif self._pending:
                pending = next(iter(self._pending.values()))
                action = pending["action"]
                step = int(pending["step"])
            self._scheduled = None
            self._pending.clear()
            self._active = None
            if action is not None:
                self._done(action, False, int(action.execute_start) + step, error)

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

        def _done(self, action, success, steps, error=""):
            message = ExecutionDone()
            message.session_id = action.session_id
            message.request_id = action.request_id
            message.action_id = action.action_id
            message.revision = action.revision
            message.execute_start = action.execute_start
            message.execute_length = action.execute_length
            message.executed_steps = steps
            message.success = success
            message.error = error
            message.timestamp_ns = self.get_clock().now().nanoseconds
            self._done_pub.publish(message)

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
            if pending is None:
                return
            pending["acks"].add(str(message.device))
            if not message.success:
                pending["failed"] = str(message.failure_message or message.failure_code)
            if pending["acks"] >= {"ur", "sharpa"}:
                self._finish_pending(key, pending)

        def _finish_pending(self, key, pending):
            self._pending.pop(key, None)
            if pending["failed"]:
                self._scheduled = None
                self._active = None
                self._done(pending["action"], False, pending["action"].execute_start + pending["step"], pending["failed"])
                return
            if pending["step"] + 1 == pending["action"].execute_length:
                action = pending["action"]
                self._active = None
                self._done(action, True, action.execute_start + action.execute_length)
                return
            self._scheduled = {
                "action": pending["action"],
                "plan": pending["plan"],
                "step": pending["step"] + 1,
                "due": next_step_due(
                    pending["published_at"], pending["action"].frequency_hz
                ),
            }

        def _process(self):
            now = time.monotonic()
            for key, pending in list(self._pending.items()):
                if now - pending["started"] > pending["timeout"]:
                    self._pending.pop(key, None)
                    self._scheduled = None
                    self._active = None
                    self._done(pending["action"], False, pending["action"].execute_start + pending["step"], "hardware command acknowledgement timeout")
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
                self._active = None
                self.get_logger().error(f"action execution failed: {exc}")
                self._done(
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
            hand = HandCommand()
            hand.source = HandCommand.SOURCE_ACTOR
            hand.action_id = action.action_id
            hand.revision = action.revision
            hand.step_index = step
            hand.q_cmd = hand_target.tolist()
            hand.left_enable = True
            hand.right_enable = True
            key = (action.action_id, int(action.revision), step)
            published_at = time.monotonic()
            self._pending[key] = {"action": action, "plan": plan, "step": step, "acks": set(), "failed": "", "started": published_at, "published_at": published_at, "timeout": max(0.5, 4.0 / float(action.frequency_hz))}
            self._arm_pub.publish(arm)
            self._hand_pub.publish(hand)

        def _action(self, action, provider="policy"):
            try:
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
                if self._active is not None:
                    raise ActorIkError("another actor action is still executing")
                if not self._ur_valid or not self._hand_valid:
                    raise ActorIkError("valid UR and SharpA state are required")
                if action.eef_frame != CANONICAL_EEF_FRAME:
                    raise ActorIkError(f"unsupported eef_frame {action.eef_frame!r}")
                if not np.isfinite(action.frequency_hz) or not 5.0 <= action.frequency_hz <= 125.0:
                    raise ActorIkError("frequency_hz must be in [5, 125]")
                if action.left_eef and action.right_eef:
                    plan = self._core.make_plan(left_eef=action.left_eef, right_eef=action.right_eef, hand_joint=action.hand_joint, execute_length=action.execute_length, left_seed=self._left, right_seed=self._right)
                else:
                    left = _array(action.left_arm_joint, action.execute_length, 6, "left_arm_joint")
                    right = _array(action.right_arm_joint, action.execute_length, 6, "right_arm_joint")
                    hand = _array(action.hand_joint, action.execute_length, 44, "hand_joint")
                    plan = IkPlan(left.astype(np.float32), right.astype(np.float32), hand.astype(np.float32))
                self._active = (action.action_id, int(action.revision))
                self._scheduled = {
                    "action": action,
                    "plan": plan,
                    "step": 0,
                    "due": time.monotonic(),
                }
                self._process()
            except Exception as exc:
                self.get_logger().error(f"action IK failed: {exc}")
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
