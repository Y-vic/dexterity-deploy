"""Launch the minimal Quest-to-Adam integration stack in safe dry-run mode."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    teleoperation_launch = PathJoinSubstitution(
        [FindPackageShare("teleoperation"), "launch", "teleoperation.launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "adam_dry_run",
                default_value="true",
                description=(
                    "Keep true for Quest/WebVR validation; set false only for "
                    "an attended hardware test."
                ),
            ),
            DeclareLaunchArgument("start_zed", default_value="true"),
            DeclareLaunchArgument("start_foxglove", default_value="true"),
            DeclareLaunchArgument("quest_fix_neck_waist", default_value="false"),
            DeclareLaunchArgument("quest_enable_neck", default_value="true"),
            DeclareLaunchArgument(
                "quest_retarget_method",
                default_value="nonlinear_ik",
                description=(
                    "Quest arm IK method: local_qp, shoulder_prior, "
                    "nonlinear_ik, or elbow_pole."
                ),
            ),
            DeclareLaunchArgument(
                "quest_nonlinear_filter_enabled", default_value="true"
            ),
            DeclareLaunchArgument("zed_video_layout", default_value="mono"),
            DeclareLaunchArgument("zed_video_bitrate", default_value="4000000"),
            DeclareLaunchArgument("zed_quest_stream_port", default_value="5602"),
            DeclareLaunchArgument(
                "quest_access_token",
                default_value=EnvironmentVariable(
                    "PND_QUEST_ACCESS_TOKEN",
                    default_value="",
                ),
            ),
            DeclareLaunchArgument(
                "quest_public_web_url",
                default_value="https://10.10.20.127/webvr/",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(teleoperation_launch),
                launch_arguments={
                    "mode": "teleop",
                    "teleop_source": "quest",
                    "start_bias": "true",
                    "bias_startup_sequence_mode": "bias_init_then_bias",
                    "start_status": "true",
                    "start_adam": "true",
                    "start_manus": "false",
                    "start_sharpa": "false",
                    "start_zed": LaunchConfiguration("start_zed"),
                    "quest_access_token": LaunchConfiguration("quest_access_token"),
                    "quest_public_web_url": LaunchConfiguration("quest_public_web_url"),
                    "start_monitor": "true",
                    "start_actor_node": "false",
                    "start_obs_node": "false",
                    "start_foxglove": LaunchConfiguration("start_foxglove"),
                    "adam_dry_run": LaunchConfiguration("adam_dry_run"),
                    "zed_inference_stream_enabled": "false",
                    "zed_monitor_stream_enabled": LaunchConfiguration("start_zed"),
                    "zed_monitor_stream_host": "10.10.20.127",
                    "zed_monitor_stream_port": "5600",
                    "zed_quest_stream_enabled": LaunchConfiguration("start_zed"),
                    "zed_quest_stream_bind_host": "0.0.0.0",
                    "zed_quest_stream_port": LaunchConfiguration(
                        "zed_quest_stream_port"
                    ),
                    "zed_video_layout": LaunchConfiguration("zed_video_layout"),
                    "zed_video_bitrate": LaunchConfiguration("zed_video_bitrate"),
                    "zed_browser_ui_enabled": "false",
                    "quest_video_enabled": "false",
                    "quest_turn_enabled": "false",
                    "quest_fix_neck_waist": LaunchConfiguration("quest_fix_neck_waist"),
                    "quest_enable_neck": LaunchConfiguration("quest_enable_neck"),
                    "quest_retarget_method": LaunchConfiguration(
                        "quest_retarget_method"
                    ),
                    "quest_nonlinear_filter_enabled": LaunchConfiguration(
                        "quest_nonlinear_filter_enabled"
                    ),
                }.items(),
            ),
        ]
    )
