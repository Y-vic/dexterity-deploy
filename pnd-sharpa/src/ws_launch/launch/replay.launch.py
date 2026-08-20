"""Launch the workstation graph with recording replay as its action source."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _bool_arg(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _float_arg(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _int_arg(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("adam_sharpa_description"))
    default_model_xml = str(
        description_share / "mujoco" / "adam_pro_sharpa_kinematics.xml"
    )
    workstation_launch = str(
        Path(get_package_share_directory("ws_launch"))
        / "launch"
        / "workstation.launch.py"
    )
    common_arguments = {
        "bind_host": LaunchConfiguration("bind_host"),
        "state_port": LaunchConfiguration("state_port"),
        "tactile_port": LaunchConfiguration("tactile_port"),
        "action_port": LaunchConfiguration("action_port"),
        "action_ttl_ms": LaunchConfiguration("action_ttl_ms"),
        "vision_rtp_port": LaunchConfiguration("vision_rtp_port"),
        "dashboard_port": LaunchConfiguration("dashboard_port"),
        "action_horizon": LaunchConfiguration("action_horizon"),
        "actor_send_hz": LaunchConfiguration("actor_send_hz"),
        "obs_rate_hz": LaunchConfiguration("obs_rate_hz"),
        "dry_run": LaunchConfiguration("dry_run"),
        "enable_adam": LaunchConfiguration("enable_adam"),
        "enable_sharpa": LaunchConfiguration("enable_sharpa"),
        "vision_decode": LaunchConfiguration("vision_decode"),
        "vision_decode_backend": LaunchConfiguration("vision_decode_backend"),
        "vision_gst_launch": LaunchConfiguration("vision_gst_launch"),
        "start_policy_pipeline": "false",
        "execution_mode": "synchronous",
        "execute_steps": "1",
        "hold_while_waiting": "false",
        "model_xml": LaunchConfiguration("model_xml"),
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument("sample_dir"),
            DeclareLaunchArgument("bind_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("state_port", default_value="15020"),
            DeclareLaunchArgument("tactile_port", default_value="15021"),
            DeclareLaunchArgument("action_port", default_value="15010"),
            DeclareLaunchArgument("action_ttl_ms", default_value="120"),
            DeclareLaunchArgument("vision_rtp_port", default_value="5601"),
            DeclareLaunchArgument("dashboard_port", default_value="8088"),
            DeclareLaunchArgument("playback_rate", default_value="1.0"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument("fk", default_value="false"),
            DeclareLaunchArgument("action_horizon", default_value="40"),
            DeclareLaunchArgument("actor_send_hz", default_value="30.0"),
            DeclareLaunchArgument("obs_rate_hz", default_value="30.0"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("enable_adam", default_value="true"),
            DeclareLaunchArgument("enable_sharpa", default_value="true"),
            DeclareLaunchArgument("vision_decode", default_value="true"),
            DeclareLaunchArgument("vision_decode_backend", default_value="auto"),
            DeclareLaunchArgument("vision_gst_launch", default_value="gst-launch-1.0"),
            DeclareLaunchArgument("model_xml", default_value=default_model_xml),
            DeclareLaunchArgument("include_waist", default_value="false"),
            DeclareLaunchArgument("include_neck", default_value="false"),
            DeclareLaunchArgument("ik_max_nfev", default_value="80"),
            DeclareLaunchArgument("ik_pos_weight", default_value="45.0"),
            DeclareLaunchArgument("ik_rot_weight", default_value="3.5"),
            DeclareLaunchArgument("ik_reg_weight", default_value="0.08"),
            DeclareLaunchArgument("ik_smooth_weight", default_value="0.04"),
            DeclareLaunchArgument("ik_diff_step", default_value="0.0001"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(workstation_launch),
                launch_arguments=common_arguments.items(),
            ),
            Node(
                package="ws_core",
                executable="replay",
                name="replay",
                output="screen",
                parameters=[
                    {
                        "sample_dir": LaunchConfiguration("sample_dir"),
                        "action_horizon": ParameterValue(
                            LaunchConfiguration("action_horizon"), value_type=int
                        ),
                        "actor_send_hz": _float_arg("actor_send_hz"),
                        "playback_rate": _float_arg("playback_rate"),
                        "loop": _bool_arg("loop"),
                        "fk": _bool_arg("fk"),
                        "model_xml": LaunchConfiguration("model_xml"),
                        "inference_request_topic": "/ws/inference/request",
                        "action_plan_topic": "/ws/action_plan",
                        "pred_topic": "/ws/pred",
                        "status_topic": "/ws/replay/status",
                    }
                ],
            ),
            Node(
                package="ws_core",
                executable="action_ik",
                name="action_ik",
                output="screen",
                condition=IfCondition(LaunchConfiguration("fk")),
                parameters=[
                    {
                        "pred_topic": "/ws/pred",
                        "action_plan_topic": "/ws/action_plan",
                        "status_topic": "/ws/action_ik/status",
                        "model_xml": LaunchConfiguration("model_xml"),
                        "include_waist": _bool_arg("include_waist"),
                        "include_neck": _bool_arg("include_neck"),
                        "enable_adam": _bool_arg("enable_adam"),
                        "enable_sharpa": _bool_arg("enable_sharpa"),
                        "ik_max_nfev": _int_arg("ik_max_nfev"),
                        "ik_pos_weight": _float_arg("ik_pos_weight"),
                        "ik_rot_weight": _float_arg("ik_rot_weight"),
                        "ik_reg_weight": _float_arg("ik_reg_weight"),
                        "ik_smooth_weight": _float_arg("ik_smooth_weight"),
                        "ik_diff_step": _float_arg("ik_diff_step"),
                    }
                ],
            ),
        ]
    )
