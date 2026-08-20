"""Launch the Quest/WebVR head target publisher."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            DeclareLaunchArgument("enabled", default_value="true"),
            DeclareLaunchArgument("input_source", default_value="auto"),
            DeclareLaunchArgument("base_frame", default_value="world"),
            DeclareLaunchArgument("head_frame", default_value="Head"),
            DeclareLaunchArgument(
                "fallback_head_frame", default_value="Head_uncalibrated"
            ),
            DeclareLaunchArgument("tf_topic", default_value="/quest/tf"),
            DeclareLaunchArgument("tf_static_topic", default_value="/quest/tf_static"),
            DeclareLaunchArgument("pose_topic", default_value="/quest/head_pose"),
            DeclareLaunchArgument(
                "orientation_topic", default_value="/quest/head_orientation"
            ),
            DeclareLaunchArgument("imu_topic", default_value="/quest/head_imu"),
            DeclareLaunchArgument("joy_topic", default_value="/quest/joy"),
            DeclareLaunchArgument("output_topic", default_value="/quest/head_joint_states"),
            DeclareLaunchArgument("status_topic", default_value="/quest/head_status"),
            DeclareLaunchArgument("publish_rate", default_value="100.0"),
            DeclareLaunchArgument("status_rate", default_value="2.0"),
            DeclareLaunchArgument("input_timeout", default_value="0.5"),
            DeclareLaunchArgument("warn_period", default_value="2.0"),
            DeclareLaunchArgument("auto_calibrate", default_value="true"),
            DeclareLaunchArgument("calibrate_button", default_value="4"),
            DeclareLaunchArgument("decalibrate_button", default_value="5"),
            DeclareLaunchArgument("yaw_sign", default_value="1.0"),
            DeclareLaunchArgument("pitch_sign", default_value="1.0"),
            DeclareLaunchArgument("yaw_limit", default_value="1.571"),
            DeclareLaunchArgument("pitch_limit", default_value="0.873"),
            DeclareLaunchArgument("yaw_joint_name", default_value="dof_pos/neckYaw"),
            DeclareLaunchArgument("pitch_joint_name", default_value="dof_pos/neckPitch"),
            DeclareLaunchArgument("frame_id", default_value="quest_head"),
            DeclareLaunchArgument("publish_zero_without_input", default_value="false"),
            Node(
                package="quest_node",
                executable="quest",
                name="quest",
                output="screen",
                parameters=[
                    {
                        "enabled": ParameterValue(
                            LaunchConfiguration("enabled"), value_type=bool
                        ),
                        "input_source": LaunchConfiguration("input_source"),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "head_frame": LaunchConfiguration("head_frame"),
                        "fallback_head_frame": LaunchConfiguration(
                            "fallback_head_frame"
                        ),
                        "pose_topic": LaunchConfiguration("pose_topic"),
                        "orientation_topic": LaunchConfiguration("orientation_topic"),
                        "imu_topic": LaunchConfiguration("imu_topic"),
                        "joy_topic": LaunchConfiguration("joy_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "status_topic": LaunchConfiguration("status_topic"),
                        "publish_rate": ParameterValue(
                            LaunchConfiguration("publish_rate"), value_type=float
                        ),
                        "status_rate": ParameterValue(
                            LaunchConfiguration("status_rate"), value_type=float
                        ),
                        "input_timeout": ParameterValue(
                            LaunchConfiguration("input_timeout"), value_type=float
                        ),
                        "warn_period": ParameterValue(
                            LaunchConfiguration("warn_period"), value_type=float
                        ),
                        "auto_calibrate": ParameterValue(
                            LaunchConfiguration("auto_calibrate"), value_type=bool
                        ),
                        "calibrate_button": ParameterValue(
                            LaunchConfiguration("calibrate_button"), value_type=int
                        ),
                        "decalibrate_button": ParameterValue(
                            LaunchConfiguration("decalibrate_button"), value_type=int
                        ),
                        "yaw_sign": ParameterValue(
                            LaunchConfiguration("yaw_sign"), value_type=float
                        ),
                        "pitch_sign": ParameterValue(
                            LaunchConfiguration("pitch_sign"), value_type=float
                        ),
                        "yaw_limit": ParameterValue(
                            LaunchConfiguration("yaw_limit"), value_type=float
                        ),
                        "pitch_limit": ParameterValue(
                            LaunchConfiguration("pitch_limit"), value_type=float
                        ),
                        "yaw_joint_name": LaunchConfiguration("yaw_joint_name"),
                        "pitch_joint_name": LaunchConfiguration("pitch_joint_name"),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "publish_zero_without_input": ParameterValue(
                            LaunchConfiguration("publish_zero_without_input"),
                            value_type=bool,
                        ),
                    }
                ],
                remappings=[
                    ("/tf", LaunchConfiguration("tf_topic")),
                    ("/tf_static", LaunchConfiguration("tf_static_topic")),
                ],
            ),
        ]
    )
