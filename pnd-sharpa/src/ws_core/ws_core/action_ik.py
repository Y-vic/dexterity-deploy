#!/usr/bin/env python3
"""Convert policy wrist/hand predictions into complete PND joint-space plans."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import rclpy
from deploy_common.joints import ADAM_COMMAND_JOINTS_19, SHARPA_JOINT_NAMES
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ws_msgs.msg import ActionPlan, PolicyPred, Status

from .common import age_ms, json_dumps, json_or_raw, make_status, set_header
from .kinematics import (
    ACTION_DIM,
    KinematicsError,
    PndKinematics,
    action62_relative_to_absolute,
    physical62_to_posttrain_absolute,
    posttrain_absolute_to_physical62,
)


class ActionIk(Node):
    def __init__(self) -> None:
        super().__init__("action_ik")
        self.declare_parameter("pred_topic", "/ws/pred")
        self.declare_parameter("action_plan_topic", "/ws/action_plan")
        self.declare_parameter("status_topic", "/ws/action_ik/status")
        self.declare_parameter("model_xml", "")
        self.declare_parameter("include_waist", False)
        self.declare_parameter("include_neck", False)
        self.declare_parameter("enable_adam", True)
        self.declare_parameter("enable_sharpa", True)
        self.declare_parameter("ik_max_nfev", 80)
        self.declare_parameter("ik_pos_weight", 45.0)
        self.declare_parameter("ik_rot_weight", 3.5)
        self.declare_parameter("ik_reg_weight", 0.08)
        self.declare_parameter("ik_smooth_weight", 0.04)
        self.declare_parameter("ik_diff_step", 1e-4)

        self.pred_topic = str(self.get_parameter("pred_topic").value)
        self.action_plan_topic = str(self.get_parameter("action_plan_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.model_xml = str(self.get_parameter("model_xml").value)
        self.include_waist = bool(self.get_parameter("include_waist").value)
        self.include_neck = bool(self.get_parameter("include_neck").value)
        self.enable_adam = bool(self.get_parameter("enable_adam").value)
        self.enable_sharpa = bool(self.get_parameter("enable_sharpa").value)
        self.ik_max_nfev = int(self.get_parameter("ik_max_nfev").value)
        self.ik_pos_weight = float(self.get_parameter("ik_pos_weight").value)
        self.ik_rot_weight = float(self.get_parameter("ik_rot_weight").value)
        self.ik_reg_weight = float(self.get_parameter("ik_reg_weight").value)
        self.ik_smooth_weight = float(self.get_parameter("ik_smooth_weight").value)
        self.ik_diff_step = float(self.get_parameter("ik_diff_step").value)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.work_event = threading.Event()
        self.pending_pred: PolicyPred | None = None
        self.pending_request_id: int | None = None
        self.processing_request_id: int | None = None
        self.pred_received = 0
        self.pred_superseded = 0
        self.plans_published = 0
        self.plan_failures = 0
        self.last_pred_seq: int | None = None
        self.last_plan_id: int | None = None
        self.last_plan_time: float | None = None
        self.last_duration_s: float | None = None
        self.last_error = ""
        self.cached_request_id: int | None = None
        self.cached_plan: ActionPlan | None = None

        self.kinematics = PndKinematics(
            self.model_xml,
            include_waist=self.include_waist,
            include_neck=self.include_neck,
            ik_max_nfev=self.ik_max_nfev,
            ik_pos_weight=self.ik_pos_weight,
            ik_rot_weight=self.ik_rot_weight,
            ik_reg_weight=self.ik_reg_weight,
            ik_smooth_weight=self.ik_smooth_weight,
            ik_diff_step=self.ik_diff_step,
        )
        self.create_subscription(PolicyPred, self.pred_topic, self._on_pred, 10)
        self.plan_pub = self.create_publisher(ActionPlan, self.action_plan_topic, 10)
        self.status_pub = self.create_publisher(Status, self.status_topic, 10)
        self.create_timer(0.5, self._publish_status)
        self.worker = threading.Thread(target=self._work_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"action_ik: pred={self.pred_topic}, plan={self.action_plan_topic}, "
            f"model={self.kinematics.model_xml}"
        )

    def _on_pred(self, msg: PolicyPred) -> None:
        request_id = self._pred_request_id(msg)
        with self.lock:
            cached = (
                self.cached_plan
                if request_id is not None and request_id == self.cached_request_id
                else None
            )
            duplicate_inflight = request_id is not None and request_id in {
                self.pending_request_id,
                self.processing_request_id,
            }
        if cached is not None:
            self.plan_pub.publish(cached)
            return
        if duplicate_inflight:
            return
        with self.lock:
            if self.pending_pred is not None:
                self.pred_superseded += 1
            self.pending_pred = msg
            self.pending_request_id = request_id
            self.pred_received += 1
            self.last_pred_seq = int(msg.seq)
        self.work_event.set()

    def _work_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.work_event.wait(0.1):
                continue
            self.work_event.clear()
            with self.lock:
                pred = self.pending_pred
                self.pending_pred = None
                self.processing_request_id = self.pending_request_id
                self.pending_request_id = None
            if pred is None:
                continue
            started = time.monotonic()
            try:
                plan = self._convert(pred)
                if self.stop_event.is_set() or not rclpy.ok(context=self.context):
                    return
                self.plan_pub.publish(plan)
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.plan_failures += 1
                    self.last_error = str(exc)
                    self.processing_request_id = None
                self.get_logger().error(f"action_ik rejected prediction {pred.seq}: {exc}")
                continue
            with self.lock:
                self.plans_published += 1
                self.last_plan_id = int(plan.plan_id)
                self.last_plan_time = time.monotonic()
                self.last_duration_s = self.last_plan_time - started
                self.last_error = ""
                self.cached_request_id = int(plan.request_id)
                self.cached_plan = plan
                self.processing_request_id = None

    @staticmethod
    def _pred_request_id(msg: PolicyPred) -> int | None:
        parsed = json_or_raw(msg.payload_json)
        payload = parsed.get("json") if parsed.get("valid") else None
        metadata = payload.get("_ws_policy_client") if isinstance(payload, dict) else None
        try:
            request_id = int(metadata.get("request_id"))
        except (AttributeError, TypeError, ValueError):
            return None
        return request_id if request_id > 0 else None

    def _convert(self, pred: PolicyPred) -> ActionPlan:
        parsed = json_or_raw(pred.payload_json)
        payload = parsed.get("json") if parsed.get("valid") else None
        if not isinstance(payload, dict):
            raise ValueError("policy prediction is not valid JSON")
        action, action_key = self._action_matrix(payload)
        server_execution = self._server_execution_metadata(payload, action.shape[0])
        action_hz = (
            float(server_execution["frequency_hz"])
            if server_execution is not None
            else self._positive_float(payload.get("action_hz"), "action_hz")
        )
        metadata = payload.get("_ws_policy_client")
        if not isinstance(metadata, dict):
            raise ValueError("prediction missing _ws_policy_client metadata")
        request_id = self._positive_int(metadata.get("request_id"), "request_id")
        request_stamp_ns = self._positive_int(
            metadata.get("request_stamp_ns"), "request_stamp_ns"
        )
        robot_state = self._anchor_state(metadata.get("robot_state_anchor"))
        action_frame = self._action_frame(payload)
        converted = self.kinematics.convert_state(robot_state)
        input_action_frame = action_frame
        if action_frame == "posttrain_relative_eef_yzx_raw2hand":
            posttrain_anchor = physical62_to_posttrain_absolute(
                converted.hand_pose_62d[None, :]
            )[0]
            posttrain_absolute = action62_relative_to_absolute(action, posttrain_anchor)
            action = posttrain_absolute_to_physical62(posttrain_absolute)
            action_frame = "absolute_current_hip"

        source_kind = str(payload.get("source_kind") or "policy")
        if source_kind not in {"policy", "replay"}:
            raise ValueError("source_kind must be policy or replay")
        end_behavior = str(
            payload.get("end_behavior")
            or ("stop" if source_kind == "replay" else "request_next")
        )
        if end_behavior not in {"stop", "request_next"}:
            raise ValueError("end_behavior must be stop or request_next")

        adam_rows: list[list[float]] = []
        sharpa_rows: list[list[float]] = []
        action_abs_rows: list[list[float]] = []
        ik_reports: list[dict[str, Any]] = []
        previous_qpos: np.ndarray | None = None
        for step_index in range(action.shape[0]):
            targets = self.kinematics.plan_action(
                action_rel62=action,
                anchor_state_62d=converted.hand_pose_62d,
                robot_state=robot_state,
                action_step_index=step_index,
                enable_adam=self.enable_adam,
                enable_sharpa=self.enable_sharpa,
                action_frame=action_frame,
                qpos_previous=previous_qpos,
            )
            if targets.ik_qpos is not None:
                previous_qpos = targets.ik_qpos.copy()
            adam_rows.append(targets.adam_q19)
            sharpa_rows.append(targets.sharpa_q44)
            action_abs_rows.append(targets.selected_abs62.astype(float).tolist())
            ik_reports.append(targets.report.get("ik", {}))

        horizon = int(action.shape[0])
        if "elapsed_ns" in payload:
            elapsed_ns = np.asarray(payload["elapsed_ns"], dtype=np.int64)
            if elapsed_ns.shape != (horizon,) or elapsed_ns[0] != 0:
                raise ValueError(f"elapsed_ns must have shape ({horizon},) and start at zero")
            if np.any(np.diff(elapsed_ns) <= 0):
                raise ValueError("elapsed_ns must be strictly increasing")
        else:
            elapsed_ns = np.rint(
                np.arange(horizon, dtype=np.float64) * 1_000_000_000.0 / action_hz
            ).astype(np.int64)
        plan_payload = {
            "schema": "ws.action_plan.v1",
            "source_kind": source_kind,
            "source": pred.provider,
            "plan_id": int(pred.seq),
            "request_id": request_id,
            "request_stamp_ns": request_stamp_ns,
            "action_hz": action_hz,
            "horizon": horizon,
            "elapsed_ns": elapsed_ns.tolist(),
            "adam": {
                "joint_names": list(ADAM_COMMAND_JOINTS_19),
                "q": adam_rows,
                "valid": self.enable_adam,
            },
            "sharpa": {
                "joint_names": list(SHARPA_JOINT_NAMES),
                "q": sharpa_rows,
                "valid": self.enable_sharpa,
                "control_mode": "position",
            },
            "selected_action_abs62": action_abs_rows,
            "end_behavior": end_behavior,
            "debug": {
                "policy_pred_seq": int(pred.seq),
                "policy_schema": payload.get("schema"),
                "action_key": action_key,
                "input_frame": input_action_frame,
                "ik_input_frame": action_frame,
                "posttrain_inverse_applied": (
                    input_action_frame == "posttrain_relative_eef_yzx_raw2hand"
                ),
                "fk": converted.report,
                "ik": ik_reports,
                "policy_timing": metadata,
            },
        }
        if server_execution is not None:
            plan_payload["_ws_sharpa_v4"] = server_execution
        msg = ActionPlan()
        set_header(msg, "action_ik", self.get_clock())
        msg.plan_id = int(pred.seq)
        msg.request_id = request_id
        msg.request_stamp_ns = request_stamp_ns
        msg.action_hz = action_hz
        msg.source = pred.provider
        msg.payload_json = json_dumps(plan_payload)
        return msg

    @staticmethod
    def _action_matrix(payload: dict[str, Any]) -> tuple[np.ndarray, str]:
        for key in (
            "action_hand_pose_62d",
            "action_hand_pose_62d_relative_eef",
            "action",
            "actions",
            "trajectory",
            "sharpa62_action",
        ):
            if key not in payload:
                continue
            action = np.asarray(payload[key], dtype=np.float32)
            if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] != ACTION_DIM:
                raise ValueError(f"{key} has shape {action.shape}, expected (T, 62)")
            if not np.all(np.isfinite(action)):
                raise ValueError(f"{key} contains NaN or Inf")
            return action, key
        raise ValueError("policy prediction has no 62D action matrix")

    @staticmethod
    def _server_execution_metadata(
        payload: dict[str, Any], action_horizon: int
    ) -> dict[str, Any] | None:
        raw = payload.get("_ws_sharpa_v4")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("_ws_sharpa_v4 must be an object")
        action_id = raw.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("_ws_sharpa_v4.action_id must be nonempty")
        revision = ActionIk._nonnegative_int(
            raw.get("revision"), "_ws_sharpa_v4.revision"
        )
        execute_start = ActionIk._nonnegative_int(
            raw.get("execute_start"), "_ws_sharpa_v4.execute_start"
        )
        execute_length = ActionIk._positive_int(
            raw.get("execute_length"), "_ws_sharpa_v4.execute_length"
        )
        action_length = ActionIk._positive_int(
            raw.get("action_length"), "_ws_sharpa_v4.action_length"
        )
        frequency_hz = ActionIk._positive_float(
            raw.get("frequency_hz"), "_ws_sharpa_v4.frequency_hz"
        )
        if raw.get("server_driven_execution") is not True:
            raise ValueError("_ws_sharpa_v4.server_driven_execution must be true")
        if execute_start + execute_length > action_length:
            raise ValueError("_ws_sharpa_v4 execution slice exceeds action_length")
        if action_horizon != execute_length:
            raise ValueError(
                "action_hand_pose_62d horizon must equal "
                "_ws_sharpa_v4.execute_length"
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

    @staticmethod
    def _action_frame(payload: dict[str, Any]) -> str:
        eef_def = payload.get("eef_def")
        if eef_def is not None:
            if eef_def != "absolute":
                raise ValueError("public policy action eef_def must be absolute")
            return "absolute_current_hip"
        frame = str(
            payload.get("wrist_frame")
            or payload.get("action_wrist_frame")
            or payload.get("action_frame")
            or "relative_eef"
        )
        if frame in {"absolute_current_hip", "absolute_hip", "current_robot_hip"}:
            return "absolute_current_hip"
        if frame == "posttrain_relative_eef_yzx_raw2hand":
            return frame
        return "relative_eef"

    @staticmethod
    def _anchor_state(wrapper: Any) -> dict[str, Any]:
        if not isinstance(wrapper, dict) or wrapper.get("valid") is False:
            raise KinematicsError("inference observation has no valid robot-state anchor")
        payload = wrapper.get("payload")
        if not isinstance(payload, dict):
            raise KinematicsError("robot-state anchor has no payload")
        state = payload.get("json") if payload.get("valid") else None
        if not isinstance(state, dict):
            raise KinematicsError("robot-state anchor payload is not valid JSON")
        return state

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be positive")
        return result

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be non-negative")
        return result

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self.lock:
            payload = {
                "schema": "ws.action_ik.status.v1",
                "topics": {"pred": self.pred_topic, "action_plan": self.action_plan_topic},
                "model_xml": str(self.kinematics.model_xml),
                "counts": {
                    "pred_received": self.pred_received,
                    "pred_superseded": self.pred_superseded,
                    "plans_published": self.plans_published,
                    "plan_failures": self.plan_failures,
                },
                "last": {
                    "pred_seq": self.last_pred_seq,
                    "plan_id": self.last_plan_id,
                    "plan_age_ms": age_ms(self.last_plan_time, now),
                    "duration_s": self.last_duration_s,
                },
                "last_error": self.last_error,
            }
            ok = not self.last_error
        self.status_pub.publish(make_status(self.get_clock(), "action_ik", ok, payload))

    def destroy_node(self) -> bool:
        self.stop_event.set()
        self.work_event.set()
        self.worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActionIk()
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
