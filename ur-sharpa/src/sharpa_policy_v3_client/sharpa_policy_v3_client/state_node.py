from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from sharpa_policy_v3_interfaces.msg import (
    CameraFrame as CameraFrameMsg,
    DeformationFrame as DeformationFrameMsg,
    HandStateFrame,
    PolicyFault,
    RobotStateFrame,
    TauFrame as TauFrameMsg,
    UrStateFrame,
    WrenchFrame as WrenchFrameMsg,
)
from sharpa_policy_v3_interfaces.srv import BuildObservation

from .buffers import (
    BufferCapacityError,
    CameraFrame,
    DeformationFrame,
    FrameValidationError,
    ObservationBuffers,
    StateFrame,
    TauFrame,
    WrenchFrame,
)
from .observation import (
    ObservationBuilder,
    ObservationCapacityError,
    ObservationNotReady,
    ObservationValidationError,
    validate_format_capacity,
)
from .serialization import MessageTooLargeError, SerializationError, packb


FAULT_SCHEMA = "sharpa_policy_fault.v1"


class StateNode(Node):
    def __init__(self) -> None:
        super().__init__("state_node")
        self._declare_parameters()
        self.aggregate_hardware_state = self._bool_parameter(
            "aggregate_hardware_state"
        )
        self.state_merge_hz = self._positive_float_parameter("state_merge_hz")
        self.max_state_skew_ns = int(
            self._positive_float_parameter("max_state_skew_ms") * 1_000_000
        )
        self.max_state_age_ns = int(
            self._positive_float_parameter("max_state_age_ms") * 1_000_000
        )
        self.max_message_size = self._positive_int_parameter("max_message_size")
        max_buffer_bytes = self._positive_int_parameter("max_buffer_bytes")
        self.buffers = ObservationBuffers(
            camera_frame_capacity=self._positive_int_parameter(
                "ego_buffer_frames"
            ),
            camera_byte_capacity=max_buffer_bytes,
            state_frame_capacity=self._positive_int_parameter(
                "state_buffer_frames"
            ),
            state_byte_capacity=max_buffer_bytes,
            tau_frame_capacity=self._positive_int_parameter("tau_buffer_frames"),
            tau_byte_capacity=max_buffer_bytes,
            wrench_frame_capacity=self._positive_int_parameter(
                "wrench_buffer_frames"
            ),
            wrench_byte_capacity=max_buffer_bytes,
            deformation_frame_capacity=self._positive_int_parameter(
                "deformation_buffer_frames"
            ),
            deformation_byte_capacity=max_buffer_bytes,
        )
        self.observation_builder = ObservationBuilder(self.buffers)
        self.fault_pub = self.create_publisher(
            PolicyFault,
            self._string_parameter("fault_topic", nonempty=True),
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            self._string_parameter("state_status_topic", nonempty=True),
            10,
        )
        self._source_lock = threading.Lock()
        self._latest_ur_state: UrStateFrame | None = None
        self._latest_hand_state: HandStateFrame | None = None
        self._last_merged_state_key: tuple[int, int] | None = None
        self._has_merged_state = False
        self._create_source_subscriptions()
        self.build_observation_service = self.create_service(
            BuildObservation,
            self._string_parameter("build_observation_service", nonempty=True),
            self._on_build_observation,
        )
        self.status_timer = self.create_timer(1.0, self._publish_status)

    def _declare_parameters(self) -> None:
        parameters = {
            "aggregate_hardware_state": False,
            "state_merge_hz": 30.0,
            "max_state_skew_ms": 50.0,
            "max_state_age_ms": 250.0,
            "max_message_size": 64 * 1024 * 1024,
            "max_buffer_bytes": 64 * 1024 * 1024,
            "ego_buffer_frames": 3,
            "state_buffer_frames": 19,
            "tau_buffer_frames": 19,
            "wrench_buffer_frames": 19,
            "deformation_buffer_frames": 3,
            "ego_camera_topic": "/sharpa/v3/source/ego_cam",
            "left_wrist_camera_topic": "/sharpa/v3/source/left_wrist_cam",
            "right_wrist_camera_topic": "/sharpa/v3/source/right_wrist_cam",
            "state_topic": "/sharpa/v3/source/state",
            "ur_state_topic": "/ur_position",
            "hand_state_topic": "/sharpa/hand_state",
            "tau_topic": "/sharpa/v3/source/tau",
            "wrench_topic": "/sharpa/v3/source/wrench",
            "deformation_topic": "/sharpa/v3/source/deformation",
            "build_observation_service": "/sharpa/v3/state/build_observation",
            "fault_topic": "/sharpa/v3/policy/fault",
            "state_status_topic": "/sharpa/v3/state/status",
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)

    def _create_source_subscriptions(self) -> None:
        self.camera_subscriptions = []
        camera_topics = {
            "ego_cam": self._string_parameter("ego_camera_topic", nonempty=True),
            "left_wrist_cam": self._string_parameter(
                "left_wrist_camera_topic",
                nonempty=True,
            ),
            "right_wrist_cam": self._string_parameter(
                "right_wrist_camera_topic",
                nonempty=True,
            ),
        }
        for camera_name, topic in camera_topics.items():
            self.camera_subscriptions.append(
                self.create_subscription(
                    CameraFrameMsg,
                    topic,
                    self._camera_callback(camera_name),
                    10,
                )
            )
        state_topic = self._string_parameter("state_topic", nonempty=True)
        if self.aggregate_hardware_state:
            self.state_pub = self.create_publisher(RobotStateFrame, state_topic, 10)
            self.ur_state_sub = self.create_subscription(
                UrStateFrame,
                self._string_parameter("ur_state_topic", nonempty=True),
                self._on_ur_hardware_state,
                10,
            )
            self.hand_state_sub = self.create_subscription(
                HandStateFrame,
                self._string_parameter("hand_state_topic", nonempty=True),
                self._on_hand_hardware_state,
                10,
            )
            self.state_merge_timer = self.create_timer(
                1.0 / self.state_merge_hz,
                self._merge_hardware_state,
            )
        else:
            self.state_sub = self.create_subscription(
                RobotStateFrame,
                state_topic,
                self._on_state,
                10,
            )
        self.tau_sub = self.create_subscription(
            TauFrameMsg,
            self._string_parameter("tau_topic", nonempty=True),
            self._on_tau,
            10,
        )
        self.wrench_sub = self.create_subscription(
            WrenchFrameMsg,
            self._string_parameter("wrench_topic", nonempty=True),
            self._on_wrench,
            10,
        )
        self.deformation_sub = self.create_subscription(
            DeformationFrameMsg,
            self._string_parameter("deformation_topic", nonempty=True),
            self._on_deformation,
            10,
        )

    def _camera_callback(self, camera_name: str) -> Any:
        def callback(message: CameraFrameMsg) -> None:
            self._on_camera(camera_name, message)

        return callback

    def _on_camera(self, camera_name: str, message: CameraFrameMsg) -> None:
        try:
            self.buffers.push_camera(
                camera_name,
                CameraFrame(
                    timestamp_ns=int(message.timestamp_ns),
                    encoding=message.encoding,
                    data=bytes(message.data),
                    valid=message.valid,
                ),
            )
        except (
            FrameValidationError,
            BufferCapacityError,
            TypeError,
            ValueError,
        ) as exc:
            self._invalid_source(camera_name, exc)

    def _on_ur_hardware_state(self, message: UrStateFrame) -> None:
        with self._source_lock:
            self._latest_ur_state = message

    def _on_hand_hardware_state(self, message: HandStateFrame) -> None:
        with self._source_lock:
            self._latest_hand_state = message

    def _merge_hardware_state(self) -> None:
        with self._source_lock:
            ur_state = self._latest_ur_state
            hand_state = self._latest_hand_state
        if ur_state is None or hand_state is None:
            return
        key = (int(ur_state.timestamp_ns), int(hand_state.timestamp_ns))
        if key == self._last_merged_state_key:
            return
        now_ns = time.time_ns()
        if (
            now_ns - key[0] > self.max_state_age_ns
            or now_ns - key[1] > self.max_state_age_ns
            or abs(key[0] - key[1]) > self.max_state_skew_ns
        ):
            return
        if (
            not ur_state.valid
            or not hand_state.left_valid
            or not hand_state.right_valid
            or ur_state.joint_dimension != 6
            or ur_state.eef_dimension != 9
            or hand_state.joint_dimension != 22
            or ur_state.eef_frame != "robot_base"
            or len(ur_state.left_joint) != 6
            or len(ur_state.right_joint) != 6
            or len(ur_state.left_eef) != 9
            or len(ur_state.right_eef) != 9
            or len(hand_state.left_joint) != 22
            or len(hand_state.right_joint) != 22
        ):
            return
        message = RobotStateFrame()
        message.timestamp_ns = max(key)
        message.has_left_wrist_joint = True
        message.left_wrist_joint_dimension = 6
        message.left_wrist_joint = list(ur_state.left_joint)
        message.has_left_wrist_eef = True
        message.left_wrist_eef_dimension = 9
        message.left_wrist_eef = list(ur_state.left_eef)
        message.left_wrist_eef_frame = "robot_base"
        message.has_right_wrist_joint = True
        message.right_wrist_joint_dimension = 6
        message.right_wrist_joint = list(ur_state.right_joint)
        message.has_right_wrist_eef = True
        message.right_wrist_eef_dimension = 9
        message.right_wrist_eef = list(ur_state.right_eef)
        message.right_wrist_eef_frame = "robot_base"
        message.hand_joint_dimension = 22
        message.has_left_hand_joint = True
        message.left_hand_joint = list(hand_state.left_joint)
        message.has_right_hand_joint = True
        message.right_hand_joint = list(hand_state.right_joint)
        message.valid = True
        if not self._on_state(message):
            return
        self._last_merged_state_key = key
        self._has_merged_state = True
        self.state_pub.publish(message)

    def _on_state(self, message: RobotStateFrame) -> bool:
        try:
            if not message.has_left_wrist_eef and message.left_wrist_eef_frame:
                raise ValueError(
                    "left_wrist_eef_frame must be empty when has-field is false"
                )
            if not message.has_right_wrist_eef and message.right_wrist_eef_frame:
                raise ValueError(
                    "right_wrist_eef_frame must be empty when has-field is false"
                )
            expected_hand_dimension = (
                22
                if message.has_left_hand_joint or message.has_right_hand_joint
                else 0
            )
            self._require_dimensions(
                "hand_joint",
                (message.hand_joint_dimension, expected_hand_dimension),
            )
            left_joint = self._optional_float_vector(
                message.has_left_wrist_joint,
                message.left_wrist_joint_dimension,
                message.left_wrist_joint,
                "left_wrist_joint",
            )
            right_joint = self._optional_float_vector(
                message.has_right_wrist_joint,
                message.right_wrist_joint_dimension,
                message.right_wrist_joint,
                "right_wrist_joint",
            )
            left_eef = self._optional_float_vector(
                message.has_left_wrist_eef,
                message.left_wrist_eef_dimension,
                message.left_wrist_eef,
                "left_wrist_eef",
                expected_dimension=9,
            )
            right_eef = self._optional_float_vector(
                message.has_right_wrist_eef,
                message.right_wrist_eef_dimension,
                message.right_wrist_eef,
                "right_wrist_eef",
                expected_dimension=9,
            )
            left_hand = self._optional_float_vector(
                message.has_left_hand_joint,
                message.hand_joint_dimension,
                message.left_hand_joint,
                "left_hand_joint",
                expected_dimension=22,
                shared_dimension=True,
            )
            right_hand = self._optional_float_vector(
                message.has_right_hand_joint,
                message.hand_joint_dimension,
                message.right_hand_joint,
                "right_hand_joint",
                expected_dimension=22,
                shared_dimension=True,
            )
            self.buffers.push_state(
                StateFrame(
                    timestamp_ns=int(message.timestamp_ns),
                    left_joint=left_joint,
                    left_eef=left_eef,
                    left_eef_frame=(
                        message.left_wrist_eef_frame
                        if message.has_left_wrist_eef
                        else None
                    ),
                    right_joint=right_joint,
                    right_eef=right_eef,
                    right_eef_frame=(
                        message.right_wrist_eef_frame
                        if message.has_right_wrist_eef
                        else None
                    ),
                    left_hand_joint=left_hand,
                    right_hand_joint=right_hand,
                    valid=message.valid,
                )
            )
            return True
        except (
            FrameValidationError,
            BufferCapacityError,
            TypeError,
            ValueError,
        ) as exc:
            self._invalid_source("state", exc)
            return False

    def _on_tau(self, message: TauFrameMsg) -> None:
        try:
            self._require_dimensions(
                "tau",
                (message.joint_dimension, 22),
                (len(message.left), 22),
                (len(message.right), 22),
                (len(message.left_valid), 22),
                (len(message.right_valid), 22),
            )
            self.buffers.push_tau(
                TauFrame(
                    timestamp_ns=int(message.timestamp_ns),
                    left=np.asarray(message.left, dtype=np.float32),
                    right=np.asarray(message.right, dtype=np.float32),
                    left_valid=np.asarray(message.left_valid, dtype=np.bool_),
                    right_valid=np.asarray(message.right_valid, dtype=np.bool_),
                )
            )
        except (
            FrameValidationError,
            BufferCapacityError,
            TypeError,
            ValueError,
        ) as exc:
            self._invalid_source("tau", exc)

    def _on_wrench(self, message: WrenchFrameMsg) -> None:
        try:
            element_count = 5 * 6
            self._require_dimensions(
                "wrench",
                (message.fingertip_count, 5),
                (message.wrench_dimension, 6),
                (len(message.left), element_count),
                (len(message.right), element_count),
                (len(message.left_valid), 5),
                (len(message.right_valid), 5),
            )
            self.buffers.push_wrench(
                WrenchFrame(
                    timestamp_ns=int(message.timestamp_ns),
                    left=np.asarray(message.left, dtype=np.float32).reshape(5, 6),
                    right=np.asarray(message.right, dtype=np.float32).reshape(5, 6),
                    left_valid=np.asarray(message.left_valid, dtype=np.bool_),
                    right_valid=np.asarray(message.right_valid, dtype=np.bool_),
                )
            )
        except (
            FrameValidationError,
            BufferCapacityError,
            TypeError,
            ValueError,
        ) as exc:
            self._invalid_source("wrench", exc)

    def _on_deformation(self, message: DeformationFrameMsg) -> None:
        try:
            element_count = 5 * 240 * 240
            self._require_dimensions(
                "deformation",
                (message.fingertip_count, 5),
                (message.height, 240),
                (message.width, 240),
                (len(message.left), element_count),
                (len(message.right), element_count),
                (len(message.left_valid), 5),
                (len(message.right_valid), 5),
            )
            self.buffers.push_deformation(
                DeformationFrame(
                    timestamp_ns=int(message.timestamp_ns),
                    left=np.asarray(message.left, dtype=np.uint8).reshape(5, 240, 240),
                    right=np.asarray(message.right, dtype=np.uint8).reshape(
                        5, 240, 240
                    ),
                    left_valid=np.asarray(message.left_valid, dtype=np.bool_),
                    right_valid=np.asarray(message.right_valid, dtype=np.bool_),
                )
            )
        except (
            FrameValidationError,
            BufferCapacityError,
            TypeError,
            ValueError,
        ) as exc:
            self._invalid_source("deformation", exc)

    def _on_build_observation(
        self,
        request: BuildObservation.Request,
        response: BuildObservation.Response,
    ) -> BuildObservation.Response:
        if self.aggregate_hardware_state and not self._has_merged_state:
            return self._observation_error(
                response,
                retryable=True,
                code="state_not_ready",
                message="no synchronized UR and SharpA state has been merged",
            )
        try:
            metadata_format = json.loads(request.metadata_format_json)
            execution_feedback = (
                json.loads(request.execution_feedback_json)
                if request.execution_feedback_json
                else None
            )
            requested_limit = int(request.max_message_size)
            if requested_limit <= 0:
                raise ValueError("max_message_size must be positive")
            message_limit = min(requested_limit, self.max_message_size)
            active_format = validate_format_capacity(
                self.buffers,
                metadata_format,
                max_message_size=message_limit,
            )
            observation = self.observation_builder.build(
                active_format,
                session_id=request.session_id,
                request_id=int(request.request_id),
                timestamp_ns=int(request.timestamp_ns),
                prompt=request.prompt,
                execution_feedback=execution_feedback,
                max_message_size=message_limit,
            )
            encoded = packb(observation, max_size=message_limit)
        except ObservationNotReady as exc:
            return self._observation_error(
                response,
                retryable=True,
                code="observation_not_ready",
                message=str(exc),
            )
        except (ObservationCapacityError, MessageTooLargeError) as exc:
            return self._observation_error(
                response,
                retryable=False,
                code="metadata_capacity_exceeded",
                message=str(exc),
            )
        except (
            json.JSONDecodeError,
            ObservationValidationError,
            SerializationError,
            TypeError,
            ValueError,
        ) as exc:
            return self._observation_error(
                response,
                retryable=False,
                code="invalid_observation_request",
                message=str(exc),
            )
        except Exception as exc:
            self.get_logger().exception(f"observation build failed: {exc}")
            return self._observation_error(
                response,
                retryable=True,
                code="observation_build_failed",
                message=str(exc),
            )
        response.success = True
        response.retryable = False
        response.error_code = ""
        response.error_message = ""
        response.observation_msgpack = encoded
        return response

    @staticmethod
    def _observation_error(
        response: BuildObservation.Response,
        *,
        retryable: bool,
        code: str,
        message: str,
    ) -> BuildObservation.Response:
        response.success = False
        response.retryable = retryable
        response.error_code = code
        response.error_message = message
        response.observation_msgpack = b""
        return response

    def _invalid_source(self, source: str, exception: Exception) -> None:
        self.buffers.clear_source(source)
        fault = PolicyFault()
        fault.schema = FAULT_SCHEMA
        fault.code = "invalid_source_frame"
        fault.message = f"{source}: {exception}"
        fault.retryable = True
        fault.safe_stop = True
        fault.clear_plan = True
        fault.timestamp_ns = time.time_ns()
        self.fault_pub.publish(fault)
        self.get_logger().error(f"invalid_source_frame: {fault.message}")

    def _publish_status(self) -> None:
        payload = {
            "schema": "sharpa_state_status.v1",
            "aggregate_hardware_state": self.aggregate_hardware_state,
            "has_merged_state": self._has_merged_state,
            "buffers": {
                "ego_cam": len(self.buffers.camera("ego_cam")),
                "left_wrist_cam": len(self.buffers.camera("left_wrist_cam")),
                "right_wrist_cam": len(self.buffers.camera("right_wrist_cam")),
                "state": len(self.buffers.state),
                "tau": len(self.buffers.tau),
                "wrench": len(self.buffers.wrench),
                "deformation": len(self.buffers.deformation),
            },
            "timestamp_ns": time.time_ns(),
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.status_pub.publish(message)

    @staticmethod
    def _optional_float_vector(
        present: bool,
        dimension: int,
        values: Any,
        field: str,
        *,
        expected_dimension: int | None = None,
        shared_dimension: bool = False,
    ) -> np.ndarray | None:
        if not present:
            if len(values) != 0:
                raise ValueError(f"{field} must be empty when has-field is false")
            if not shared_dimension and int(dimension) != 0:
                raise ValueError(
                    f"{field} dimension must be 0 when has-field is false"
                )
            return None
        actual_dimension = int(dimension)
        if actual_dimension <= 0:
            raise ValueError(f"{field} dimension must be positive")
        if expected_dimension is not None and actual_dimension != expected_dimension:
            raise ValueError(f"{field} dimension must be {expected_dimension}")
        if len(values) != actual_dimension:
            raise ValueError(f"{field} length does not match its dimension")
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _require_dimensions(field: str, *pairs: tuple[int, int]) -> None:
        for actual, expected in pairs:
            if int(actual) != expected:
                raise ValueError(
                    f"{field} dimension mismatch: expected {expected}, got {actual}"
                )

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


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node = StateNode()
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
