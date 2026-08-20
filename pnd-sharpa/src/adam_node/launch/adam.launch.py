"""Launch the Adam lowstate bridge and command output node."""

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
            DeclareLaunchArgument("lowstate_topic", default_value="/lowstate"),
            DeclareLaunchArgument(
                "bias_joint_states_topic",
                default_value="/adam_bias_command_joint_states",
            ),
            DeclareLaunchArgument(
                "command_joint_states_topic",
                default_value="/adam_command_joint_states",
            ),
            DeclareLaunchArgument(
                "physical_joint_states_topic",
                default_value="/adam_physical_joint_states",
            ),
            DeclareLaunchArgument("robot_states_topic", default_value="/robot_states"),
            DeclareLaunchArgument(
                "control_joint_states_topic",
                default_value="/joint_states",
            ),
            DeclareLaunchArgument("control_status_topic", default_value="/control_status"),
            DeclareLaunchArgument("publish_rate", default_value="100.0"),
            DeclareLaunchArgument("command_timeout", default_value="0.25"),
            DeclareLaunchArgument("lowstate_timeout", default_value="0.5"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("require_control_subscriber", default_value="true"),
            Node(
                package="adam_node",
                executable="adam",
                name="adam",
                output="screen",
                parameters=[
                    {
                        "lowstate_topic": LaunchConfiguration("lowstate_topic"),
                        "bias_joint_states_topic": LaunchConfiguration(
                            "bias_joint_states_topic"
                        ),
                        "command_joint_states_topic": LaunchConfiguration(
                            "command_joint_states_topic"
                        ),
                        "physical_joint_states_topic": LaunchConfiguration(
                            "physical_joint_states_topic"
                        ),
                        "robot_states_topic": LaunchConfiguration("robot_states_topic"),
                        "control_joint_states_topic": LaunchConfiguration(
                            "control_joint_states_topic"
                        ),
                        "control_status_topic": LaunchConfiguration(
                            "control_status_topic"
                        ),
                        "publish_rate": ParameterValue(
                            LaunchConfiguration("publish_rate"),
                            value_type=float,
                        ),
                        "command_timeout": ParameterValue(
                            LaunchConfiguration("command_timeout"),
                            value_type=float,
                        ),
                        "lowstate_timeout": ParameterValue(
                            LaunchConfiguration("lowstate_timeout"),
                            value_type=float,
                        ),
                        "dry_run": ParameterValue(
                            LaunchConfiguration("dry_run"),
                            value_type=bool,
                        ),
                        "require_control_subscriber": ParameterValue(
                            LaunchConfiguration("require_control_subscriber"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
