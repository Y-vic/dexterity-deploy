#!/usr/bin/env python3
"""Provide a recorded sample as request-driven action segments."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from deploy_common.joints import ADAM_COMMAND_JOINTS_19, SHARPA_JOINT_NAMES
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ws_msgs.msg import ActionPlan, InferenceRequest, PolicyPred, Status

from .common import age_ms, json_dumps, make_status, set_header
from .kinematics import (
    PndKinematics,
    action62_absolute_to_relative,
    physical62_to_posttrain_absolute,
)


@dataclass(frozen=True)
class ReplaySample:
    path: Path
    name: str
    elapsed_ns: np.ndarray
    adam_names: list[str]
    adam_q31: np.ndarray
    adam_q19: np.ndarray
    sharpa_names: list[str]
    sharpa_q44: np.ndarray
    recorded_rate_hz: float

    @property
    def row_count(self) -> int:
        return int(self.elapsed_ns.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.elapsed_ns[-1]) / 1_000_000_000.0

    @classmethod
    def load(cls, sample_dir: str) -> ReplaySample:
        path = Path(sample_dir).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"recording sample directory not found: {path}")
        if not (path / "COMPLETE").is_file():
            raise ValueError(f"recording sample is not finalized (missing COMPLETE): {path}")
        schema_path = path / "schema.json"
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"recording schema not found: {schema_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid recording schema JSON: {exc}") from exc
        if not isinstance(schema, dict) or schema.get("format") != "pnd_local_monitor_columnar":
            raise ValueError("unsupported recording schema format")

        timeline_schema = cls._section(schema, "timeline")
        adam_schema = cls._section(cls._section(schema, "adam"), "physical_31")
        sharpa_schema = cls._section(cls._section(schema, "sharpa"), "joints_44")
        groups = cls._section(adam_schema, "groups")
        adam_indices = [int(value) for value in groups.get("upper_body_19_indices", [])]
        adam_command_names = [
            str(value) for value in groups.get("upper_body_19_names", [])
        ]
        adam_names = [str(value) for value in adam_schema.get("joint_names", [])]
        sharpa_names = [str(value) for value in sharpa_schema.get("joint_names", [])]
        if adam_command_names != list(ADAM_COMMAND_JOINTS_19):
            raise ValueError("recording Adam upper-body joint order mismatch")
        if len(adam_names) != 31 or len(set(adam_names)) != 31:
            raise ValueError("recording Adam physical joint names must contain 31 entries")
        if len(adam_indices) != 19 or len(set(adam_indices)) != 19:
            raise ValueError("recording Adam upper-body indices must be 19 unique entries")
        if min(adam_indices, default=-1) < 0 or max(adam_indices, default=31) >= 31:
            raise ValueError("recording Adam upper-body indices are outside physical_31")
        if sharpa_names != list(SHARPA_JOINT_NAMES):
            raise ValueError("recording SharpA joint order mismatch")

        timeline_file = path / str(timeline_schema.get("clock_file", "timeline/clock.npz"))
        adam_file = path / str(adam_schema.get("file", "adam/physical_31.npz"))
        sharpa_file = path / str(sharpa_schema.get("file", "sharpa/joints_44.npz"))
        try:
            with np.load(timeline_file, allow_pickle=False) as timeline:
                elapsed_ns = np.asarray(timeline["elapsed_ns"], dtype=np.int64)
                rate_hz = float(np.asarray(timeline["sample_rate_hz"]).item())
            with np.load(adam_file, allow_pickle=False) as adam:
                adam_q31 = np.asarray(adam["q"], dtype=np.float64)
                adam_valid = np.asarray(adam["valid"], dtype=np.bool_)
            with np.load(sharpa_file, allow_pickle=False) as sharpa:
                sharpa_q44 = np.asarray(sharpa["q"], dtype=np.float64)
                sharpa_valid = np.asarray(sharpa["valid"], dtype=np.bool_)
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            raise ValueError(f"failed to load recording arrays: {exc}") from exc
        cls._validate_arrays(
            elapsed_ns, adam_q31, adam_valid, sharpa_q44, sharpa_valid, rate_hz
        )
        return cls(
            path=path,
            name=str(schema.get("sample") or path.name),
            elapsed_ns=np.ascontiguousarray(elapsed_ns - elapsed_ns[0]),
            adam_names=adam_names,
            adam_q31=np.ascontiguousarray(adam_q31),
            adam_q19=np.ascontiguousarray(adam_q31[:, adam_indices]),
            sharpa_names=sharpa_names,
            sharpa_q44=np.ascontiguousarray(sharpa_q44),
            recorded_rate_hz=rate_hz,
        )

    @staticmethod
    def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
        section = payload.get(key)
        if not isinstance(section, dict):
            raise ValueError(f"recording schema is missing object section: {key}")
        return section

    @staticmethod
    def _validate_arrays(
        elapsed_ns: np.ndarray,
        adam_q31: np.ndarray,
        adam_valid: np.ndarray,
        sharpa_q44: np.ndarray,
        sharpa_valid: np.ndarray,
        rate_hz: float,
    ) -> None:
        if elapsed_ns.ndim != 1 or elapsed_ns.size == 0:
            raise ValueError(f"timeline elapsed_ns must have shape (N,), got {elapsed_ns.shape}")
        rows = int(elapsed_ns.size)
        if adam_q31.shape != (rows, 31):
            raise ValueError(f"Adam q must have shape ({rows}, 31), got {adam_q31.shape}")
        if sharpa_q44.shape != (rows, 44):
            raise ValueError(f"SharpA q must have shape ({rows}, 44), got {sharpa_q44.shape}")
        if adam_valid.shape != (rows,) or sharpa_valid.shape != (rows,):
            raise ValueError("Adam and SharpA valid arrays must have shape (N,)")
        if not np.all(adam_valid) or not np.all(sharpa_valid):
            raise ValueError("recording contains invalid Adam or SharpA rows")
        if np.any(np.diff(elapsed_ns) <= 0):
            raise ValueError("timeline elapsed_ns must be strictly increasing")
        if not np.all(np.isfinite(adam_q31)) or not np.all(np.isfinite(sharpa_q44)):
            raise ValueError("recording joint targets contain NaN or Inf")
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError(f"invalid recording sample rate: {rate_hz}")

    def robot_state(self, row: int) -> dict[str, Any]:
        return {
            "schema": "pnd.deploy.obs.v1",
            "adam": {
                "name": list(self.adam_names),
                "q": self.adam_q31[row].astype(float).tolist(),
            },
            "sharpa": {
                "name": list(self.sharpa_names),
                "q": self.sharpa_q44[row].astype(float).tolist(),
            },
        }


class Replay(Node):
    def __init__(self) -> None:
        super().__init__("replay")
        self.declare_parameter("sample_dir", "")
        self.declare_parameter("action_horizon", 40)
        self.declare_parameter("actor_send_hz", 30.0)
        self.declare_parameter("playback_rate", 1.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("fk", False)
        self.declare_parameter("model_xml", "")
        self.declare_parameter("inference_request_topic", "/ws/inference/request")
        self.declare_parameter("action_plan_topic", "/ws/action_plan")
        self.declare_parameter("pred_topic", "/ws/pred")
        self.declare_parameter("status_topic", "/ws/replay/status")

        self.sample_dir = str(self.get_parameter("sample_dir").value).strip()
        self.action_horizon = int(self.get_parameter("action_horizon").value)
        self.actor_send_hz = float(self.get_parameter("actor_send_hz").value)
        self.playback_rate = float(self.get_parameter("playback_rate").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.fk = bool(self.get_parameter("fk").value)
        self.model_xml = str(self.get_parameter("model_xml").value)
        self.inference_request_topic = str(
            self.get_parameter("inference_request_topic").value
        )
        self.action_plan_topic = str(self.get_parameter("action_plan_topic").value)
        self.pred_topic = str(self.get_parameter("pred_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self._validate_parameters()
        self.sample = ReplaySample.load(self.sample_dir)
        self.effective_rate_hz = self.sample.recorded_rate_hz * self.playback_rate
        if not math.isclose(
            self.actor_send_hz,
            self.effective_rate_hz,
            rel_tol=0.02,
            abs_tol=0.1,
        ):
            raise ValueError(
                f"actor_send_hz={self.actor_send_hz:g} must match recording playback "
                f"rate {self.effective_rate_hz:g} Hz; replay does not resample"
            )

        self.posttrain_absolute62: np.ndarray | None = None
        if self.fk:
            self.posttrain_absolute62 = self._compute_posttrain_actions()

        self.lock = threading.Lock()
        self.next_row = 0
        self.phase = "sample"
        self.cycle = 0
        self.last_request_id = 0
        self.cached_output: ActionPlan | PolicyPred | None = None
        self.state = "waiting_for_request"
        self.requests_received = 0
        self.outputs_published = 0
        self.retry_publishes = 0
        self.last_segment_start: int | None = None
        self.last_segment_end: int | None = None
        self.last_segment_kind: str | None = None
        self.last_output_time: float | None = None
        self.last_error = ""

        self.create_subscription(
            InferenceRequest,
            self.inference_request_topic,
            self._on_request,
            10,
        )
        self.plan_pub = self.create_publisher(ActionPlan, self.action_plan_topic, 10)
        self.pred_pub = self.create_publisher(PolicyPred, self.pred_topic, 10)
        self.status_pub = self.create_publisher(Status, self.status_topic, 10)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(
            f"replay: sample={self.sample.path}, rows={self.sample.row_count}, "
            f"action_horizon={self.action_horizon}, actor_send_hz={self.actor_send_hz:g}, "
            f"loop={self.loop}, fk={self.fk}, output={'pred' if self.fk else 'plan'}"
        )

    def _validate_parameters(self) -> None:
        if not self.sample_dir:
            raise ValueError("sample_dir is required")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        for name, value in (
            ("actor_send_hz", self.actor_send_hz),
            ("playback_rate", self.playback_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _compute_posttrain_actions(self) -> np.ndarray:
        kinematics = PndKinematics(self.model_xml)
        physical_rows: list[np.ndarray] = []
        for row in range(self.sample.row_count):
            converted = kinematics.convert_state(self.sample.robot_state(row))
            physical_rows.append(converted.hand_pose_62d)
        return physical62_to_posttrain_absolute(
            np.asarray(physical_rows, dtype=np.float32)
        )

    def _on_request(self, msg: InferenceRequest) -> None:
        request_id = int(msg.request_id)
        output: ActionPlan | PolicyPred | None = None
        with self.lock:
            if request_id < self.last_request_id:
                return
            if request_id == self.last_request_id:
                if self.cached_output is None:
                    return
                output = self.cached_output
                self.retry_publishes += 1
            else:
                self.last_request_id = request_id
                self.requests_received += 1
                segment = self._next_segment_locked()
                if segment is None:
                    self.state = "complete"
                    return
                start, end, segment_kind, is_final = segment
                output = self._make_output(msg, start, end, segment_kind, is_final)
                self.cached_output = output
                self.outputs_published += 1
                self.last_segment_start = start
                self.last_segment_end = end
                self.last_segment_kind = segment_kind
                self.last_output_time = time.monotonic()
                self.state = "complete" if is_final else "segment_published"
                self.last_error = ""
        if isinstance(output, PolicyPred):
            self.pred_pub.publish(output)
        elif isinstance(output, ActionPlan):
            self.plan_pub.publish(output)

    def _next_segment_locked(self) -> tuple[int, int, str, bool] | None:
        if self.phase == "complete":
            return None
        if self.phase == "gap":
            self.phase = "sample"
            self.next_row = 0
            self.cycle += 1
            return 0, self.action_horizon, "loop_gap", False

        start = self.next_row
        end = min(start + self.action_horizon, self.sample.row_count)
        self.next_row = end
        if end < self.sample.row_count:
            return start, end, "sample", False
        if self.loop:
            self.phase = "gap"
            return start, end, "sample", False
        self.phase = "complete"
        return start, end, "sample", True

    def _make_output(
        self,
        request: InferenceRequest,
        start: int,
        end: int,
        segment_kind: str,
        is_final: bool,
    ) -> ActionPlan | PolicyPred:
        if segment_kind == "loop_gap":
            rows = np.zeros(self.action_horizon, dtype=np.int64)
            elapsed_ns = self._uniform_elapsed_ns(self.action_horizon)
        else:
            rows = np.arange(start, end, dtype=np.int64)
            elapsed_ns = self.sample.elapsed_ns[start:end] - self.sample.elapsed_ns[start]
            elapsed_ns = np.rint(
                elapsed_ns.astype(np.float64) / self.playback_rate
            ).astype(np.int64)
        if elapsed_ns.size > 1 and np.any(np.diff(elapsed_ns) <= 0):
            raise ValueError("playback_rate collapses recording timeline rows")
        end_behavior = "stop" if is_final else "request_next"
        if self.fk:
            return self._make_fk_prediction(
                request,
                rows,
                elapsed_ns,
                start,
                end,
                segment_kind,
                end_behavior,
            )
        return self._make_joint_plan(
            request,
            rows,
            elapsed_ns,
            start,
            end,
            segment_kind,
            end_behavior,
        )

    def _make_joint_plan(
        self,
        request: InferenceRequest,
        rows: np.ndarray,
        elapsed_ns: np.ndarray,
        start: int,
        end: int,
        segment_kind: str,
        end_behavior: str,
    ) -> ActionPlan:
        payload = self._base_payload(request, start, end, segment_kind, end_behavior)
        payload.update(
            {
                "schema": "ws.action_plan.v1",
                "source_kind": "replay",
                "source": "recording_replay",
                "action_hz": self.effective_rate_hz,
                "horizon": int(rows.size),
                "elapsed_ns": elapsed_ns.tolist(),
                "adam": {
                    "joint_names": list(ADAM_COMMAND_JOINTS_19),
                    "q": self.sample.adam_q19[rows].astype(float).tolist(),
                    "valid": True,
                },
                "sharpa": {
                    "joint_names": list(SHARPA_JOINT_NAMES),
                    "q": self.sample.sharpa_q44[rows].astype(float).tolist(),
                    "valid": True,
                    "control_mode": "position",
                },
            }
        )
        msg = ActionPlan()
        set_header(msg, "replay", self.get_clock())
        msg.plan_id = int(request.request_id)
        msg.request_id = int(request.request_id)
        msg.request_stamp_ns = int(request.request_stamp_ns)
        msg.action_hz = self.effective_rate_hz
        msg.source = "recording_replay"
        msg.payload_json = json_dumps(payload)
        return msg

    def _make_fk_prediction(
        self,
        request: InferenceRequest,
        rows: np.ndarray,
        elapsed_ns: np.ndarray,
        start: int,
        end: int,
        segment_kind: str,
        end_behavior: str,
    ) -> PolicyPred:
        if self.posttrain_absolute62 is None:
            raise RuntimeError("posttrain FK actions are not initialized")
        anchor_row = int(rows[0])
        anchor = self.posttrain_absolute62[anchor_row]
        action = action62_absolute_to_relative(
            self.posttrain_absolute62[rows],
            anchor,
        )
        payload = self._base_payload(request, start, end, segment_kind, end_behavior)
        payload.update(
            {
                "schema": "ws.replay.posttrain_fk.v1",
                "source_kind": "replay",
                "action_hand_pose_62d_relative_eef": action.astype(float).tolist(),
                "action_hz": self.effective_rate_hz,
                "elapsed_ns": elapsed_ns.tolist(),
                "wrist_frame": "posttrain_relative_eef_yzx_raw2hand",
                "_ws_policy_client": {
                    "request_id": int(request.request_id),
                    "request_stamp_ns": int(request.request_stamp_ns),
                    "trigger_action_seq": int(request.trigger_action_seq),
                    "robot_state_anchor": {
                        "valid": True,
                        "payload": {
                            "valid": True,
                            "json": self.sample.robot_state(anchor_row),
                        },
                    },
                },
            }
        )
        msg = PolicyPred()
        set_header(msg, "replay", self.get_clock())
        msg.seq = int(request.request_id)
        msg.provider = "replay_posttrain_fk"
        msg.payload_json = json_dumps(payload)
        return msg

    def _base_payload(
        self,
        request: InferenceRequest,
        start: int,
        end: int,
        segment_kind: str,
        end_behavior: str,
    ) -> dict[str, Any]:
        return {
            "plan_id": int(request.request_id),
            "request_id": int(request.request_id),
            "request_stamp_ns": int(request.request_stamp_ns),
            "end_behavior": end_behavior,
            "debug": {
                "sample": self.sample.name,
                "sample_dir": str(self.sample.path),
                "recorded_rate_hz": self.sample.recorded_rate_hz,
                "playback_rate": self.playback_rate,
                "segment_kind": segment_kind,
                "segment_start_row": start,
                "segment_end_row_exclusive": end,
                "sample_rows": self.sample.row_count,
                "cycle": self.cycle,
                "loop": self.loop,
                "fk": self.fk,
            },
        }

    def _uniform_elapsed_ns(self, rows: int) -> np.ndarray:
        return np.rint(
            np.arange(rows, dtype=np.float64) * 1_000_000_000.0
            / self.effective_rate_hz
        ).astype(np.int64)

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self.lock:
            payload = {
                "schema": "ws.replay.status.v1",
                "topics": {
                    "inference_request": self.inference_request_topic,
                    "output": self.pred_topic if self.fk else self.action_plan_topic,
                },
                "replay": {
                    "state": self.state,
                    "sample_dir": str(self.sample.path),
                    "sample": self.sample.name,
                    "row_count": self.sample.row_count,
                    "next_row": self.next_row,
                    "phase": self.phase,
                    "cycle": self.cycle,
                    "loop": self.loop,
                    "fk": self.fk,
                    "recorded_rate_hz": self.sample.recorded_rate_hz,
                    "actor_send_hz": self.actor_send_hz,
                    "action_horizon": self.action_horizon,
                    "duration_s": self.sample.duration_s,
                    "playback_rate": self.playback_rate,
                },
                "segment": {
                    "kind": self.last_segment_kind,
                    "start_row": self.last_segment_start,
                    "end_row_exclusive": self.last_segment_end,
                },
                "counts": {
                    "requests_received": self.requests_received,
                    "outputs_published": self.outputs_published,
                    "retry_publishes": self.retry_publishes,
                },
                "last": {
                    "request_id": self.last_request_id,
                    "output_age_ms": age_ms(self.last_output_time, now),
                },
                "last_error": self.last_error,
            }
        self.status_pub.publish(make_status(self.get_clock(), "replay", True, payload))


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Replay | None = None
    try:
        node = Replay()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
