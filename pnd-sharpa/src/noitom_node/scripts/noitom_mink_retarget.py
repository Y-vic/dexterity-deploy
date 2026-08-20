#!/usr/bin/env python3
"""Run upper-body Adam Mink without finger/leg tasks or a busy loop."""

from __future__ import annotations

import math
import time

import rclpy
from adam_mink.adam_mink_pro import AdamMinkProNode


OUTPUT_PERIOD_S = 0.01
TIMING_REPORT_PERIOD_S = 2.0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


class OptimizedAdamMinkProNode(AdamMinkProNode):
    """Adam Pro Mink node optimized for upper-body retargeting."""

    def __init__(self) -> None:
        self._timing_samples: list[tuple[float, float]] = []
        self._last_timing_report = time.monotonic()
        self._removed_finger_tasks = 0
        self._removed_leg_tasks = 0
        super().__init__()
        self.get_logger().info(
            "Upper-body Mink optimization active: removed "
            f"{self._removed_finger_tasks} finger tasks and "
            f"{self._removed_leg_tasks} leg tasks; "
            f"{len(self.tasks)} tasks remain; output capped at "
            f"{1.0 / OUTPUT_PERIOD_S:.0f} Hz"
        )

    @staticmethod
    def _is_finger_bone(bone_name: str) -> bool:
        return (
            bone_name.startswith("LeftHand") and bone_name != "LeftHand"
        ) or (
            bone_name.startswith("RightHand") and bone_name != "RightHand"
        )

    @staticmethod
    def _is_leg_bone(bone_name: str) -> bool:
        return bone_name in {
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
        }

    @classmethod
    def _is_omitted_bone(cls, bone_name: str) -> bool:
        return cls._is_finger_bone(bone_name) or cls._is_leg_bone(bone_name)

    def get_bone_frames(self) -> list[str]:
        return [
            bone
            for bone in super().get_bone_frames()
            if not self._is_omitted_bone(bone)
        ]

    def _create_ik_tasks(self):
        original_configs = self.adam_mink_cfg.ik_cfg
        self._removed_finger_tasks = sum(
            self._is_finger_bone(cfg.bone_name) for cfg in original_configs
        )
        self._removed_leg_tasks = sum(
            self._is_leg_bone(cfg.bone_name) for cfg in original_configs
        )
        retained_configs = [
            cfg
            for cfg in original_configs
            if not self._is_omitted_bone(cfg.bone_name)
        ]
        self.adam_mink_cfg.ik_cfg = retained_configs
        retained_bones = {cfg.bone_name for cfg in retained_configs}
        self.bone_name_to_cfg = {
            name: cfg
            for name, cfg in self.bone_name_to_cfg.items()
            if name in retained_bones
        }
        self._rot_offset_quats = {
            name: value
            for name, value in self._rot_offset_quats.items()
            if name in retained_bones
        }
        self._pos_offsets = {
            name: value
            for name, value in self._pos_offsets.items()
            if name in retained_bones
        }
        self.mocap_data = {
            name: value
            for name, value in self.mocap_data.items()
            if name in retained_bones
        }
        return super()._create_ik_tasks()

    def _report_timing(self, now: float) -> None:
        if now - self._last_timing_report < TIMING_REPORT_PERIOD_S:
            return
        samples = self._timing_samples
        self._timing_samples = []
        self._last_timing_report = now
        if not samples:
            return

        loop_times = [sample[0] for sample in samples]
        solve_times = [sample[1] for sample in samples]
        loop_p50 = _percentile(loop_times, 0.50)
        loop_p95 = _percentile(loop_times, 0.95)
        loop_p99 = _percentile(loop_times, 0.99)
        solve_p95 = _percentile(solve_times, 0.95)
        overruns = sum(value > OUTPUT_PERIOD_S for value in loop_times)
        message = (
            f"Mink timing over {len(samples)} cycles: "
            f"loop p50={loop_p50 * 1e3:.2f}ms "
            f"p95={loop_p95 * 1e3:.2f}ms "
            f"p99={loop_p99 * 1e3:.2f}ms; "
            f"solve p95={solve_p95 * 1e3:.2f}ms; "
            f"overruns={overruns}"
        )
        if loop_p95 > OUTPUT_PERIOD_S:
            self.get_logger().warning(message)
        else:
            self.get_logger().debug(message)

    def ik_thread_loop(self) -> None:
        self.get_logger().info("Rate-limited IK thread loop started")
        while rclpy.ok():
            if not self.calibrated:
                time.sleep(OUTPUT_PERIOD_S)
                continue

            loop_start = time.perf_counter()
            with self._data_lock:
                mocap_data_copy = self.mocap_data.copy()
            if self.adam_mink_cfg.human_scale_table:
                self.scale_mocap_data(mocap_data_copy)
            self.offset_mocap_data(mocap_data_copy)
            self.mocap_data_adjusted = mocap_data_copy
            self._update_ik_targets()
            solve_start = time.perf_counter()
            self._solve_ik()
            solve_end = time.perf_counter()
            self._publish_joint_states()
            loop_end = time.perf_counter()

            self._timing_samples.append(
                (loop_end - loop_start, solve_end - solve_start)
            )
            self._report_timing(time.monotonic())

            remaining = OUTPUT_PERIOD_S - (loop_end - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)

        self.get_logger().info("Rate-limited IK thread loop ended")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OptimizedAdamMinkProNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
