"""Launch the local recording monitor."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    recording_root = LaunchConfiguration("recording_root")
    require_recording_root_mount = LaunchConfiguration("require_recording_root_mount")
    sample_rate_hz = LaunchConfiguration("sample_rate_hz")
    record_zed_video = LaunchConfiguration("record_zed_video")
    zed_video_rtp_port = LaunchConfiguration("zed_video_rtp_port")
    zed_video_ffmpeg = LaunchConfiguration("zed_video_ffmpeg")
    zed_video_stop_timeout_s = LaunchConfiguration("zed_video_stop_timeout_s")
    require_tactile_fresh_on_start = LaunchConfiguration(
        "require_tactile_fresh_on_start"
    )
    block_recording_on_tactile_error = LaunchConfiguration(
        "block_recording_on_tactile_error"
    )
    tactile_start_max_age_ms = LaunchConfiguration("tactile_start_max_age_ms")
    tactile_error_log_period_s = LaunchConfiguration("tactile_error_log_period_s")

    return LaunchDescription(
        [
            DeclareLaunchArgument("recording_root", default_value="/mnt/t9/recordings"),
            DeclareLaunchArgument("require_recording_root_mount", default_value="true"),
            DeclareLaunchArgument("sample_rate_hz", default_value="30.0"),
            DeclareLaunchArgument("record_zed_video", default_value="true"),
            DeclareLaunchArgument("zed_video_rtp_port", default_value="5600"),
            DeclareLaunchArgument("zed_video_ffmpeg", default_value="ffmpeg"),
            DeclareLaunchArgument("zed_video_stop_timeout_s", default_value="5.0"),
            DeclareLaunchArgument("require_tactile_fresh_on_start", default_value="true"),
            DeclareLaunchArgument("block_recording_on_tactile_error", default_value="false"),
            DeclareLaunchArgument("tactile_start_max_age_ms", default_value="500.0"),
            DeclareLaunchArgument("tactile_error_log_period_s", default_value="1.0"),
            Node(
                package="monitor_node",
                executable="monitor",
                name="monitor",
                output="screen",
                parameters=[
                    {
                        "recording_root": recording_root,
                        "require_recording_root_mount": ParameterValue(
                            require_recording_root_mount,
                            value_type=bool,
                        ),
                        "sample_rate_hz": ParameterValue(sample_rate_hz, value_type=float),
                        "status_json_topic": "/teleop/status_json",
                        "quest_webvr_status_topic": "/quest/webvr_status",
                        "quest_retarget_status_topic": "/quest/retarget_status",
                        "adam_topic": "/adam_physical_joint_states",
                        "sharpa_joint_topic": "/sharpa_physical_joint_states",
                        "tactile_deform_topic": "/sharpa_physical_tactile/deform_images",
                        "tactile_force6d_topic": "/sharpa_physical_tactile/force6d",
                        "tactile_contact_topic": "/sharpa_physical_tactile/contact_points",
                        "zed_status_topic": "/zed/status",
                        "record_zed_video": ParameterValue(
                            record_zed_video,
                            value_type=bool,
                        ),
                        "zed_video_rtp_port": ParameterValue(
                            zed_video_rtp_port,
                            value_type=int,
                        ),
                        "zed_video_ffmpeg": zed_video_ffmpeg,
                        "zed_video_stop_timeout_s": ParameterValue(
                            zed_video_stop_timeout_s,
                            value_type=float,
                        ),
                        "require_tactile_fresh_on_start": ParameterValue(
                            require_tactile_fresh_on_start,
                            value_type=bool,
                        ),
                        "block_recording_on_tactile_error": ParameterValue(
                            block_recording_on_tactile_error,
                            value_type=bool,
                        ),
                        "tactile_start_max_age_ms": ParameterValue(
                            tactile_start_max_age_ms,
                            value_type=float,
                        ),
                        "tactile_error_log_period_s": ParameterValue(
                            tactile_error_log_period_s,
                            value_type=float,
                        ),
                    }
                ],
            ),
        ]
    )
