#!/usr/bin/env python3
"""Teleop status state machine driven by sensor_msgs/Joy."""

from __future__ import annotations

import os
import json
import struct
import time
from dataclasses import dataclass
from typing import Iterable

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import String

from status_node.common import (
    STATE_DAMPING,
    STATE_D_SHARPA,
    STATE_T_ADAM,
    STATE_T_ADAM_SHARPA,
    STATE_T_INIT,
    STATE_T_INIT_SHARPA,
    age_ms,
    as_bool,
    transient_local_qos,
)


EVENT_LL = "LL"
EVENT_RR = "RR"
EVENT_UU = "UU"
EVENT_DD = "DD"
EVENT_LB = "LB"
EVENT_LT = "LT"
EVENT_LT_B = "LT+B"
EVENT_RT = "RT"
T_RECORD = "t_record"

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)

EVDEV_EVENT_FORMAT = "llHHi"
EVDEV_EVENT_SIZE = struct.calcsize(EVDEV_EVENT_FORMAT)
EV_KEY = 0x01
EV_ABS = 0x03
ABS_X = 0
ABS_Y = 1
ABS_Z = 2
ABS_RZ = 5
ABS_GAS = 9
ABS_BRAKE = 10
ABS_HAT0X = 16
ABS_HAT0Y = 17
EVDEV_AXIS_MAP = {
    ABS_X: 0,
    ABS_Y: 1,
    ABS_Z: 2,
    ABS_RZ: 3,
    ABS_GAS: 4,
    ABS_BRAKE: 5,
    ABS_HAT0X: 6,
    ABS_HAT0Y: 7,
}
EVDEV_BUTTON_MAP = {
    304: 0,  # BTN_SOUTH, A
    305: 1,  # BTN_EAST, B
    308: 2,  # BTN_WEST, X
    307: 3,  # BTN_NORTH, Y
    310: 6,  # BTN_TL, LB
    311: 7,  # BTN_TR, RB
    312: 8,  # BTN_TL2, LT digital edge
    313: 9,  # BTN_TR2, RT digital edge
    314: 10,  # BTN_SELECT
    315: 11,  # BTN_START
    316: 12,  # BTN_MODE
    317: 13,  # BTN_THUMBL
    318: 14,  # BTN_THUMBR
}

TRANSITIONS = {
    STATE_DAMPING: {
        EVENT_RR: STATE_T_INIT,
        EVENT_UU: STATE_D_SHARPA,
    },
    STATE_D_SHARPA: {
        EVENT_DD: STATE_DAMPING,
        EVENT_LL: STATE_DAMPING,
        EVENT_LT_B: STATE_DAMPING,
    },
    STATE_T_INIT: {
        EVENT_LT: STATE_T_ADAM,
        EVENT_UU: STATE_T_INIT_SHARPA,
        EVENT_LB: STATE_T_INIT_SHARPA,
        EVENT_LL: STATE_DAMPING,
        EVENT_LT_B: STATE_DAMPING,
    },
    STATE_T_INIT_SHARPA: {
        EVENT_LT: STATE_T_ADAM_SHARPA,
        EVENT_DD: STATE_T_INIT,
        EVENT_LB: STATE_T_INIT,
        EVENT_LL: STATE_DAMPING,
        EVENT_LT_B: STATE_DAMPING,
    },
    STATE_T_ADAM: {
        EVENT_LT: STATE_T_INIT,
        EVENT_UU: STATE_T_ADAM_SHARPA,
        EVENT_LB: STATE_T_ADAM_SHARPA,
        EVENT_LL: STATE_DAMPING,
        EVENT_LT_B: STATE_DAMPING,
    },
    STATE_T_ADAM_SHARPA: {
        EVENT_LT: STATE_T_INIT_SHARPA,
        EVENT_DD: STATE_T_ADAM,
        EVENT_LB: STATE_T_ADAM,
        EVENT_LL: STATE_DAMPING,
        EVENT_LT_B: STATE_DAMPING,
    },
}


@dataclass(frozen=True)
class InputSnapshot:
    """Logical controller input state after applying parameterized mapping."""

    left: bool
    right: bool
    up: bool
    down: bool
    lb: bool
    lt: bool
    rt: bool
    b: bool

    @property
    def lt_b(self) -> bool:
        return self.lt and self.b


class StatusNode(Node):
    """Publish teleop control state from an Xbox-compatible Joy stream."""

    def __init__(self) -> None:
        super().__init__("status")

        self.declare_parameter("status_topic", "/control_status")
        self.declare_parameter("status_json_topic", "/teleop/status_json")
        self.declare_parameter("publish_status_json", True)
        self.declare_parameter("joy_topics", ["/joy", "/xbox/joy"])
        self.declare_parameter("dpad_x_axis", 6)
        self.declare_parameter("dpad_y_axis", 7)
        self.declare_parameter("dpad_axis_threshold", 0.5)
        self.declare_parameter("dpad_x_right_sign", 1)
        self.declare_parameter("dpad_y_up_sign", -1)
        self.declare_parameter("lb_button", 4)
        self.declare_parameter("b_button", 1)
        self.declare_parameter("lt_axis", 2)
        self.declare_parameter("lt_button", -1)
        self.declare_parameter("lt_axis_threshold", -0.5)
        self.declare_parameter("lt_pressed_when", "below")
        self.declare_parameter("rt_axis", 4)
        self.declare_parameter("rt_button", 9)
        self.declare_parameter("rt_axis_threshold", 0.5)
        self.declare_parameter("rt_pressed_when", "above")
        self.declare_parameter("joy_priority_timeout", 1.0)
        self.declare_parameter("status_period", 0.5)
        self.declare_parameter("robot_state_topic", "/adam_physical_joint_states")
        self.declare_parameter("robot_state_check_period", 0.2)
        self._declare_device_parameters()

        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_json_topic = str(self.get_parameter("status_json_topic").value)
        self.publish_status_json = as_bool(
            self.get_parameter("publish_status_json").value
        )
        self.joy_topics = self._joy_topics_from_parameters()
        self.dpad_x_axis = int(self.get_parameter("dpad_x_axis").value)
        self.dpad_y_axis = int(self.get_parameter("dpad_y_axis").value)
        self.dpad_axis_threshold = abs(
            float(self.get_parameter("dpad_axis_threshold").value)
        )
        self.dpad_x_right_sign = self._sign(
            int(self.get_parameter("dpad_x_right_sign").value), default=1
        )
        self.dpad_y_up_sign = self._sign(
            int(self.get_parameter("dpad_y_up_sign").value), default=-1
        )
        self.lb_button = int(self.get_parameter("lb_button").value)
        self.b_button = int(self.get_parameter("b_button").value)
        self.lt_axis = int(self.get_parameter("lt_axis").value)
        self.lt_button = int(self.get_parameter("lt_button").value)
        self.lt_axis_threshold = float(
            self.get_parameter("lt_axis_threshold").value
        )
        self.lt_pressed_when = self._validated_pressed_when(
            self.get_parameter("lt_pressed_when").value,
            "lt_pressed_when",
        )
        self.rt_axis = int(self.get_parameter("rt_axis").value)
        self.rt_button = int(self.get_parameter("rt_button").value)
        self.rt_axis_threshold = float(
            self.get_parameter("rt_axis_threshold").value
        )
        self.rt_pressed_when = self._validated_pressed_when(
            self.get_parameter("rt_pressed_when").value,
            "rt_pressed_when",
        )
        self.joy_priority_timeout = float(
            self.get_parameter("joy_priority_timeout").value
        )
        self.status_period = float(self.get_parameter("status_period").value)
        self.robot_state_topic = str(self.get_parameter("robot_state_topic").value)
        self.robot_state_check_period = float(
            self.get_parameter("robot_state_check_period").value
        )
        self.requested_device = str(self.get_parameter("device").value).strip()
        self.evdev_device = str(self.get_parameter("evdev_device").value).strip()
        self.device_backend_request = str(
            self.get_parameter("device_backend").value
        ).strip().lower()
        self.read_device = as_bool(self.get_parameter("read_device").value)
        self.publish_joy = as_bool(self.get_parameter("publish_joy").value)
        self.joy_output_topic = str(self.get_parameter("joy_output_topic").value)
        self.axis_count = max(0, int(self.get_parameter("axis_count").value))
        self.button_count = max(0, int(self.get_parameter("button_count").value))
        self.poll_period = float(self.get_parameter("poll_period").value)
        self.device_backend, self.device = self._resolve_device()

        if self.status_period <= 0.0:
            raise ValueError("status_period must be positive")
        if self.poll_period <= 0.0:
            raise ValueError("poll_period must be positive")
        if self.joy_priority_timeout < 0.0:
            raise ValueError("joy_priority_timeout must be non-negative")
        if self.robot_state_check_period <= 0.0:
            raise ValueError("robot_state_check_period must be positive")

        self.robot_states_present = self._robot_state_topic_present()
        self.state = STATE_DAMPING
        self.auto_t_init_pending = False
        self.previous_inputs = InputSnapshot(
            left=False,
            right=False,
            up=False,
            down=False,
            lb=False,
            lt=False,
            rt=False,
            b=False,
        )
        self.last_event = (
            "startup_robot_states_present"
            if self.robot_states_present
            else "startup_robot_states_absent"
        )
        self.last_transition = "startup"
        self.t_record = False
        self.t_record_event = "startup"
        self.robot_state_count = 0
        self.last_robot_state_time: float | None = None
        self.last_joy_topic = ""
        self.active_joy_topic = ""
        self.last_joy_time: float | None = None
        self.active_joy_time: float | None = None
        self.joy_count = 0
        self.window_joy_count = 0
        self.window_time = time.monotonic()
        self.joy_hz = 0.0
        self.topic_inputs: dict[str, InputSnapshot] = {}
        self.last_axes: list[float] = []
        self.last_buttons: list[int] = []
        self.device_source = f"device:{self.device}" if self.device else "device"
        self.device_fd: int | None = None
        self.device_axes = [0.0] * self.axis_count
        self.device_buttons = [0] * self.button_count
        self.device_event_count = 0
        self.last_device_error = ""
        self.last_device_open_attempt = 0.0

        self.status_pub = self.create_publisher(
            String, self.status_topic, transient_local_qos()
        )
        self.status_json_pub = (
            self.create_publisher(String, self.status_json_topic, transient_local_qos())
            if self.publish_status_json
            else None
        )
        self.joy_pub = (
            self.create_publisher(Joy, self.joy_output_topic, 10)
            if self.publish_joy and self.joy_output_topic
            else None
        )
        joy_sub_topics = list(self.joy_topics)
        if self.read_device and self.joy_output_topic in joy_sub_topics:
            joy_sub_topics.remove(self.joy_output_topic)
        self.joy_subs = [
            self.create_subscription(
                Joy, topic, lambda msg, topic=topic: self._handle_joy(msg, topic), 10
            )
            for topic in joy_sub_topics
        ]
        self.robot_state_sub = self.create_subscription(
            JointState,
            self.robot_state_topic,
            self._on_robot_state,
            10,
        )
        if self.read_device:
            self._open_device()
            self.create_timer(self.poll_period, self._poll_device)

        self.create_timer(self.robot_state_check_period, self._check_robot_state_topic)
        self.create_timer(self.status_period, self._publish_status)
        self._publish_status()
        self.get_logger().info(
            "teleop status state machine started: "
            f"state={self.state}, joy_topics={joy_sub_topics}, "
            f"device={self.device if self.read_device else '<disabled>'}, "
            f"requested_device={self.requested_device}, "
            f"device_backend={self.device_backend}, "
            f"lt_axis={self.lt_axis}, lt_button={self.lt_button}, "
            f"lt_threshold={self.lt_axis_threshold}, "
            f"lt_pressed_when={self.lt_pressed_when}, "
            f"rt_axis={self.rt_axis}, rt_button={self.rt_button}, "
            f"rt_threshold={self.rt_axis_threshold}, "
            f"rt_pressed_when={self.rt_pressed_when}, "
            f"robot_state_topic={self.robot_state_topic}, "
            f"robot_states_present={self.robot_states_present}"
        )

    def _declare_device_parameters(self) -> None:
        self.declare_parameter("device", "/dev/input/js0")
        self.declare_parameter("device_backend", "auto")
        self.declare_parameter("evdev_device", "")
        self.declare_parameter("read_device", True)
        self.declare_parameter("publish_joy", False)
        self.declare_parameter("joy_output_topic", "/joy")
        self.declare_parameter("axis_count", 8)
        self.declare_parameter("button_count", 16)
        self.declare_parameter("poll_period", 0.005)

    def _resolve_device(self) -> tuple[str, str]:
        requested = self.requested_device
        backend = self.device_backend_request
        if backend not in {"auto", "joydev", "js", "evdev", "event"}:
            self.get_logger().warning(
                f"device_backend={backend!r} is invalid; using auto"
            )
            backend = "auto"

        if backend in {"joydev", "js"}:
            return "joydev", requested

        if backend in {"evdev", "event"}:
            return "evdev", self._resolve_evdev_path(requested)

        if self.evdev_device:
            return "evdev", self.evdev_device
        if "/event" in requested:
            return "evdev", requested
        if "/js" in requested:
            event_device = self._event_device_for_joydev(requested)
            if event_device:
                return "evdev", event_device
            return "joydev", requested
        return "joydev", requested

    def _resolve_evdev_path(self, requested: str) -> str:
        if self.evdev_device:
            return self.evdev_device
        if "/event" in requested:
            return requested
        if "/js" in requested:
            event_device = self._event_device_for_joydev(requested)
            if event_device:
                return event_device
        return requested

    def _event_device_for_joydev(self, joydev_path: str) -> str:
        joydev_name = os.path.basename(joydev_path)
        if not joydev_name:
            return ""
        try:
            with open("/proc/bus/input/devices", "r", encoding="utf-8") as handle:
                blocks = handle.read().split("\n\n")
        except OSError as exc:
            self.get_logger().warning(
                f"cannot inspect /proc/bus/input/devices for {joydev_name}: {exc}"
            )
            return ""

        for block in blocks:
            handlers = ""
            for line in block.splitlines():
                if line.startswith("H: Handlers="):
                    handlers = line.split("=", 1)[1]
                    break
            if joydev_name not in handlers.split():
                continue
            for handler in handlers.split():
                if handler.startswith("event"):
                    return f"/dev/input/{handler}"
        return ""

    def _joy_topics_from_parameters(self) -> list[str]:
        topics: list[str] = []
        configured = self.get_parameter("joy_topics").value
        if isinstance(configured, str):
            candidates: Iterable[str] = configured.split(",")
        else:
            candidates = configured or []

        for topic in candidates:
            topic_name = str(topic).strip()
            if topic_name and topic_name not in topics:
                topics.append(topic_name)

        return topics or ["/joy"]

    def _robot_state_topic_present(self) -> bool:
        if not self.robot_state_topic:
            return False
        return bool(self.get_publishers_info_by_topic(self.robot_state_topic))

    def _check_robot_state_topic(self) -> None:
        if not self.robot_states_present and self._robot_state_topic_present():
            self._mark_robot_states_present("robot_states_topic_present")

    def _on_robot_state(self, _msg: JointState) -> None:
        self.robot_state_count += 1
        self.last_robot_state_time = time.monotonic()
        self._mark_robot_states_present("robot_states_message")

    def _mark_robot_states_present(self, event: str) -> None:
        was_present = self.robot_states_present
        self.robot_states_present = True
        if not was_present:
            self.get_logger().info(f"robot states detected on {self.robot_state_topic}")

    def _handle_joy(self, msg: Joy, topic: str) -> None:
        now = time.monotonic()
        self.joy_count += 1
        self.window_joy_count += 1
        self.last_axes = [round(float(value), 4) for value in msg.axes]
        self.last_buttons = [int(value) for value in msg.buttons]

        current = self._snapshot(msg)
        had_topic = topic in self.topic_inputs
        previous = self.topic_inputs.get(topic, self._neutral_inputs())
        self.topic_inputs[topic] = current
        if not self._accept_joy_topic(topic, now):
            return

        self.last_joy_topic = topic
        self.last_joy_time = now
        if had_topic:
            self.previous_inputs = previous
            if current.rt and not previous.rt:
                self._toggle_t_record_from_rt()
            event = self._select_event(self._events_from_edges(previous, current))
            if event is not None:
                self._apply_event(event)
        self.previous_inputs = current
        self._publish_status()

    def _snapshot(self, msg: Joy) -> InputSnapshot:
        x_axis = self._axis(msg, self.dpad_x_axis)
        y_axis = self._axis(msg, self.dpad_y_axis)
        signed_x = x_axis * self.dpad_x_right_sign
        signed_y = y_axis * self.dpad_y_up_sign
        left = signed_x < -self.dpad_axis_threshold
        right = signed_x > self.dpad_axis_threshold
        up = signed_y > self.dpad_axis_threshold
        down = signed_y < -self.dpad_axis_threshold

        return InputSnapshot(
            left=left,
            right=right,
            up=up,
            down=down,
            lb=self._button(msg, self.lb_button),
            lt=self._lt_pressed(msg),
            rt=self._rt_pressed(msg),
            b=self._button(msg, self.b_button),
        )

    def _accept_joy_topic(self, topic: str, now: float) -> bool:
        if not self.active_joy_topic:
            self.active_joy_topic = topic
            self.active_joy_time = now
            return True

        if topic == self.active_joy_topic:
            self.active_joy_time = now
            return True

        topic_priority = self._topic_priority(topic)
        active_priority = self._topic_priority(self.active_joy_topic)
        if topic_priority < active_priority:
            self.get_logger().info(
                f"switching Joy source to higher-priority topic {topic}"
            )
            self.active_joy_topic = topic
            self.active_joy_time = now
            return True

        active_age = (
            float("inf")
            if self.active_joy_time is None
            else now - self.active_joy_time
        )
        if active_age > self.joy_priority_timeout:
            self.get_logger().warn(
                "switching Joy source after priority timeout: "
                f"{self.active_joy_topic} -> {topic}"
            )
            self.active_joy_topic = topic
            self.active_joy_time = now
            return True

        return False

    def _topic_priority(self, topic: str) -> int:
        if topic == self.device_source:
            return -1
        try:
            return self.joy_topics.index(topic)
        except ValueError:
            return len(self.joy_topics)

    def _events_from_edges(
        self, previous: InputSnapshot, current: InputSnapshot
    ) -> list[str]:
        events: list[str] = []

        if current.lt_b and not previous.lt_b:
            events.append(EVENT_LT_B)
        if current.left and not previous.left:
            events.append(EVENT_LL)
        if current.right and not previous.right:
            events.append(EVENT_RR)
        if current.up and not previous.up:
            events.append(EVENT_UU)
        if current.down and not previous.down:
            events.append(EVENT_DD)
        if current.lb and not previous.lb:
            events.append(EVENT_LB)
        if current.lt and not previous.lt and not current.b:
            events.append(EVENT_LT)

        return events

    def _select_event(self, events: list[str]) -> str | None:
        if not events:
            return None
        transitions = TRANSITIONS.get(self.state, {})
        for event in events:
            if event in transitions:
                return event
        return events[0]

    def _apply_event(self, event: str) -> None:
        old_state = self.state
        self.last_event = event
        next_state = TRANSITIONS.get(self.state, {}).get(event, self.state)
        if next_state == self.state:
            self.last_transition = f"{self.state}+{event}->keep"
            self.get_logger().info(
                f"teleop input {event} ignored in state {self.state}"
            )
            return

        self.state = next_state
        self.auto_t_init_pending = False
        self.last_transition = f"{old_state}+{event}->{self.state}"
        if self.t_record and not self._is_t_state(self.state):
            self.t_record = False
            self.t_record_event = f"stopped_on_state_exit:{self.last_transition}"
        self.get_logger().info(f"teleop state -> {self.state} ({self.last_transition})")

    def _toggle_t_record_from_rt(self) -> None:
        self.last_event = EVENT_RT
        if self.t_record:
            self.t_record = False
            self.t_record_event = f"stopped_by_{EVENT_RT}"
            self.last_transition = f"{self.state}+{EVENT_RT}->{T_RECORD}_off"
            self.get_logger().info("t_record -> inactive (RT)")
            return

        if not self._is_t_state(self.state):
            self.t_record_event = f"ignored_{EVENT_RT}_outside_t_state:{self.state}"
            self.last_transition = f"{self.state}+{EVENT_RT}->{T_RECORD}_keep_off"
            self.get_logger().info(
                f"t_record input RT ignored outside t_ state ({self.state})"
            )
            return

        self.t_record = True
        self.t_record_event = f"started_by_{EVENT_RT}"
        self.last_transition = f"{self.state}+{EVENT_RT}->{T_RECORD}"
        self.get_logger().info("t_record -> active (RT)")

    def _publish_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self.window_time
        if elapsed > 0.0:
            self.joy_hz = self.window_joy_count / elapsed
        self.window_joy_count = 0
        self.window_time = now

        status_msg = String()
        status_msg.data = self.state
        self.status_pub.publish(status_msg)

        if self.status_json_pub is None:
            return

        json_msg = String()
        json_msg.data = json.dumps(
            {
                "node": "status",
                "state": self.state,
                "mode": self.state,
                "event": self.last_event,
                "transition": self.last_transition,
                "t_record": self.t_record,
                "t_record_event": self.t_record_event,
                "record_state": T_RECORD if self.t_record else "",
                "record_trigger": EVENT_RT,
                "record_requires_t_state": True,
                "robot_states": {
                    "topic": self.robot_state_topic,
                    "present": self.robot_states_present,
                    "count": self.robot_state_count,
                    "last_age_ms": age_ms(self.last_robot_state_time, now),
                    "auto_t_init_pending": self.auto_t_init_pending,
                },
                "status_topic": self.status_topic,
                "joy_topics": self.joy_topics,
                "active_joy_topic": self.active_joy_topic,
                "last_joy_topic": self.last_joy_topic,
                "joy_count": self.joy_count,
                "joy_hz": round(self.joy_hz, 2),
                "last_joy_age_ms": age_ms(self.last_joy_time, now),
                "raw": {
                    "axes": self.last_axes,
                    "buttons": self.last_buttons,
                },
                "device": {
                    "enabled": self.read_device,
                    "path": self.device,
                    "requested_path": self.requested_device,
                    "backend": self.device_backend,
                    "open": self.device_fd is not None,
                    "event_count": self.device_event_count,
                    "last_error": self.last_device_error,
                    "publish_joy": self.joy_pub is not None,
                    "joy_output_topic": self.joy_output_topic,
                },
                "inputs": {
                    "LL": self.previous_inputs.left,
                    "RR": self.previous_inputs.right,
                    "UU": self.previous_inputs.up,
                    "DD": self.previous_inputs.down,
                    "LB": self.previous_inputs.lb,
                    "LT": self.previous_inputs.lt,
                    "RT": self.previous_inputs.rt,
                    "B": self.previous_inputs.b,
                    "LT+B": self.previous_inputs.lt_b,
                },
                "mapping": {
                    "dpad_x_axis": self.dpad_x_axis,
                    "dpad_y_axis": self.dpad_y_axis,
                    "dpad_axis_threshold": self.dpad_axis_threshold,
                    "dpad_x_right_sign": self.dpad_x_right_sign,
                    "dpad_y_up_sign": self.dpad_y_up_sign,
                    "lb_button": self.lb_button,
                    "b_button": self.b_button,
                    "lt_axis": self.lt_axis,
                    "lt_button": self.lt_button,
                    "lt_axis_threshold": self.lt_axis_threshold,
                    "lt_pressed_when": self.lt_pressed_when,
                    "rt_axis": self.rt_axis,
                    "rt_button": self.rt_button,
                    "rt_axis_threshold": self.rt_axis_threshold,
                    "rt_pressed_when": self.rt_pressed_when,
                },
            },
            separators=(",", ":"),
            ensure_ascii=True,
        )
        self.status_json_pub.publish(json_msg)

    def _lt_pressed(self, msg: Joy) -> bool:
        return self._trigger_pressed(
            msg,
            axis=self.lt_axis,
            button=self.lt_button,
            threshold=self.lt_axis_threshold,
            pressed_when=self.lt_pressed_when,
        )

    def _rt_pressed(self, msg: Joy) -> bool:
        return self._trigger_pressed(
            msg,
            axis=self.rt_axis,
            button=self.rt_button,
            threshold=self.rt_axis_threshold,
            pressed_when=self.rt_pressed_when,
        )

    def _trigger_pressed(
        self,
        msg: Joy,
        *,
        axis: int,
        button: int,
        threshold: float,
        pressed_when: str,
    ) -> bool:
        if self._button(msg, button):
            return True
        value = self._axis(msg, axis)
        if pressed_when == "above":
            return value > threshold
        if pressed_when == "abs":
            return abs(value) > abs(threshold)
        return value < threshold

    @staticmethod
    def _axis(msg: Joy, index: int) -> float:
        if index < 0 or index >= len(msg.axes):
            return 0.0
        return float(msg.axes[index])

    @staticmethod
    def _button(msg: Joy, index: int) -> bool:
        if index < 0 or index >= len(msg.buttons):
            return False
        return bool(msg.buttons[index])

    @staticmethod
    def _neutral_inputs() -> InputSnapshot:
        return InputSnapshot(
            left=False,
            right=False,
            up=False,
            down=False,
            lb=False,
            lt=False,
            rt=False,
            b=False,
        )

    @staticmethod
    def _is_t_state(state: str) -> bool:
        return state.startswith("t_")

    @staticmethod
    def _sign(value: int, default: int) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return default

    def _validated_pressed_when(self, value: object, parameter_name: str) -> str:
        normalized = str(value).strip().lower()
        if normalized in {"above", "below", "abs"}:
            return normalized
        self.get_logger().warning(
            f"{parameter_name}={value!r} is invalid; using 'below'"
        )
        return "below"

    def _open_device(self) -> None:
        if not self.device:
            self.last_device_error = "device path is empty"
            return
        try:
            self.device_fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            self.device_fd = None
            self.last_device_error = str(exc)
            self.get_logger().warning(
                f"cannot open Xbox input device {self.device}: {exc}"
            )
            return

        self.last_device_error = ""
        self.topic_inputs.pop(self.device_source, None)
        self.get_logger().info(
            f"reading Xbox input from {self.device} ({self.device_backend})"
        )
        self._publish_device_joy()

    def _close_device(self, reason: str) -> None:
        if self.device_fd is not None:
            try:
                os.close(self.device_fd)
            except OSError:
                pass
        self.device_fd = None
        self.last_device_error = reason
        self.topic_inputs.pop(self.device_source, None)

    def _poll_device(self) -> None:
        if self.device_fd is None:
            now = time.monotonic()
            if now - self.last_device_open_attempt > 1.0:
                self.last_device_open_attempt = now
                self._open_device()
            return

        changed = False
        while True:
            try:
                event_size = (
                    EVDEV_EVENT_SIZE
                    if self.device_backend == "evdev"
                    else JS_EVENT_SIZE
                )
                data = os.read(self.device_fd, event_size * 32)
            except BlockingIOError:
                break
            except OSError as exc:
                self._close_device(str(exc))
                self.get_logger().warning(
                    f"lost Xbox input device {self.device}: {exc}"
                )
                break

            if not data:
                self._close_device("device returned EOF")
                break

            usable = len(data) - (len(data) % event_size)
            for offset in range(0, usable, event_size):
                if self.device_backend == "evdev":
                    changed |= self._handle_evdev_event(data[offset : offset + event_size])
                else:
                    changed |= self._handle_joydev_event(data[offset : offset + event_size])

        if changed:
            self._publish_device_joy()

    def _handle_joydev_event(self, data: bytes) -> bool:
        _stamp, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, data)
        base_type = event_type & ~JS_EVENT_INIT
        changed = False
        if base_type == JS_EVENT_AXIS and number < len(self.device_axes):
            self.device_axes[number] = self._normalize_joydev_axis(value)
            changed = True
        elif base_type == JS_EVENT_BUTTON and number < len(self.device_buttons):
            self.device_buttons[number] = 1 if value else 0
            changed = True
        self.device_event_count += 1
        return changed

    def _handle_evdev_event(self, data: bytes) -> bool:
        _seconds, _microseconds, event_type, code, value = struct.unpack(
            EVDEV_EVENT_FORMAT, data
        )
        changed = False
        if event_type == EV_KEY:
            button_index = EVDEV_BUTTON_MAP.get(code)
            if button_index is not None and button_index < len(self.device_buttons):
                self.device_buttons[button_index] = 1 if value else 0
                changed = True
        elif event_type == EV_ABS:
            axis_index = EVDEV_AXIS_MAP.get(code)
            if axis_index is not None and axis_index < len(self.device_axes):
                self.device_axes[axis_index] = self._normalize_evdev_axis(code, value)
                changed = True
        if event_type in {EV_KEY, EV_ABS}:
            self.device_event_count += 1
        return changed

    def _publish_device_joy(self) -> None:
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "xbox"
        msg.axes = list(self.device_axes)
        msg.buttons = list(self.device_buttons)
        self._handle_joy(msg, self.device_source)
        if self.joy_pub is not None:
            self.joy_pub.publish(msg)

    @staticmethod
    def _normalize_joydev_axis(value: int) -> float:
        scale = 32767.0 if value >= 0 else 32768.0
        normalized = float(value) / scale
        return max(-1.0, min(1.0, normalized))

    @staticmethod
    def _normalize_evdev_axis(code: int, value: int) -> float:
        if code in {ABS_GAS, ABS_BRAKE}:
            normalized = float(value) / 1023.0
        elif code in {ABS_HAT0X, ABS_HAT0Y}:
            normalized = float(value)
        else:
            normalized = (float(value) - 32767.5) / 32767.5
        return max(-1.0, min(1.0, normalized))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StatusNode()
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
