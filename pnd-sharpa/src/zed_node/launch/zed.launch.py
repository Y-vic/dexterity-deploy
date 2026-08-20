"""Launch the ZED RGB source node."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            DeclareLaunchArgument("jetson_host", default_value="10.10.20.126"),
            DeclareLaunchArgument("jetson_user", default_value="pnd-humanoid"),
            DeclareLaunchArgument("webrtc_port", default_value="8443"),
            DeclareLaunchArgument("start_remote_pipeline", default_value="true"),
            DeclareLaunchArgument("video_bitrate", default_value="8000000"),
            DeclareLaunchArgument("video_fps", default_value="30"),
            DeclareLaunchArgument("video_width", default_value="1280"),
            DeclareLaunchArgument("video_height", default_value="720"),
            DeclareLaunchArgument("video_layout", default_value="mono"),
            DeclareLaunchArgument("monitor_stream_enabled", default_value="true"),
            DeclareLaunchArgument("monitor_stream_host", default_value="10.10.20.127"),
            DeclareLaunchArgument("monitor_stream_port", default_value="5600"),
            DeclareLaunchArgument("inference_stream_enabled", default_value="true"),
            DeclareLaunchArgument(
                "inference_stream_host", default_value="10.10.20.110"
            ),
            DeclareLaunchArgument("inference_stream_port", default_value="5601"),
            DeclareLaunchArgument("quest_stream_enabled", default_value="false"),
            DeclareLaunchArgument("quest_stream_bind_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("quest_stream_port", default_value="5602"),
            DeclareLaunchArgument("watchdog_enabled", default_value="true"),
            DeclareLaunchArgument("watchdog_failure_threshold", default_value="5"),
            DeclareLaunchArgument("watchdog_restart_cooldown_s", default_value="20.0"),
            DeclareLaunchArgument("browser_ui_enabled", default_value="true"),
            DeclareLaunchArgument("browser_ui_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("browser_ui_port", default_value="12100"),
            DeclareLaunchArgument("browser_ui_web_root", default_value=""),
            Node(
                package="zed_node",
                executable="zed",
                name="zed",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "jetson_host": LaunchConfiguration("jetson_host"),
                        "jetson_user": LaunchConfiguration("jetson_user"),
                        "webrtc_port": LaunchConfiguration("webrtc_port"),
                        "start_remote_pipeline": LaunchConfiguration(
                            "start_remote_pipeline"
                        ),
                        "video_bitrate": LaunchConfiguration("video_bitrate"),
                        "video_fps": LaunchConfiguration("video_fps"),
                        "video_width": LaunchConfiguration("video_width"),
                        "video_height": LaunchConfiguration("video_height"),
                        "video_layout": LaunchConfiguration("video_layout"),
                        "monitor_stream_enabled": LaunchConfiguration(
                            "monitor_stream_enabled"
                        ),
                        "monitor_stream_host": LaunchConfiguration(
                            "monitor_stream_host"
                        ),
                        "monitor_stream_port": LaunchConfiguration(
                            "monitor_stream_port"
                        ),
                        "inference_stream_enabled": LaunchConfiguration(
                            "inference_stream_enabled"
                        ),
                        "inference_stream_host": LaunchConfiguration(
                            "inference_stream_host"
                        ),
                        "inference_stream_port": LaunchConfiguration(
                            "inference_stream_port"
                        ),
                        "quest_stream_enabled": LaunchConfiguration(
                            "quest_stream_enabled"
                        ),
                        "quest_stream_bind_host": LaunchConfiguration(
                            "quest_stream_bind_host"
                        ),
                        "quest_stream_port": LaunchConfiguration("quest_stream_port"),
                        "watchdog_enabled": LaunchConfiguration("watchdog_enabled"),
                        "watchdog_failure_threshold": LaunchConfiguration(
                            "watchdog_failure_threshold"
                        ),
                        "watchdog_restart_cooldown_s": LaunchConfiguration(
                            "watchdog_restart_cooldown_s"
                        ),
                        "browser_ui_enabled": LaunchConfiguration("browser_ui_enabled"),
                        "browser_ui_host": LaunchConfiguration("browser_ui_host"),
                        "browser_ui_port": LaunchConfiguration("browser_ui_port"),
                        "browser_ui_web_root": LaunchConfiguration(
                            "browser_ui_web_root"
                        ),
                        "status_topic": "/zed/status",
                    }
                ],
            ),
        ]
    )
