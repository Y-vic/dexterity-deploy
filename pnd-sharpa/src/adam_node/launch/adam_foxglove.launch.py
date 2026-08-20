"""Launch command retarget, Adam lowstate bridge, and Foxglove view."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    command_topic = LaunchConfiguration("command_joint_states_topic")
    physical_topic = LaunchConfiguration("physical_joint_states_topic")
    bias_topic = LaunchConfiguration("bias_joint_states_topic")
    sharpa_command_topic = LaunchConfiguration("sharpa_command_topic")

    noitom_launch = PathJoinSubstitution(
        [FindPackageShare("noitom_node"), "launch", "noitom_node.launch.py"]
    )
    manus_launch = PathJoinSubstitution(
        [FindPackageShare("manus_node"), "launch", "manus_node.launch.py"]
    )
    foxglove_launch = PathJoinSubstitution(
        [FindPackageShare("foxglove_node"), "launch", "foxglove_node.launch.py"]
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            DeclareLaunchArgument("start_noitom_node", default_value="true"),
            DeclareLaunchArgument("start_manus_node", default_value="false"),
            DeclareLaunchArgument("start_foxglove_node", default_value="true"),
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
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("require_control_subscriber", default_value="true"),
            DeclareLaunchArgument("noitom_fix_neck_waist", default_value="true"),
            DeclareLaunchArgument(
                "sharpa_command_topic",
                default_value="/sharpa_command_joint_states",
            ),
            DeclareLaunchArgument("publish_rate", default_value="100.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(noitom_launch),
                condition=IfCondition(LaunchConfiguration("start_noitom_node")),
                launch_arguments={
                    "retarget_output_topic": command_topic,
                    "bias_joint_states_topic": bias_topic,
                    "fix_neck_waist": LaunchConfiguration("noitom_fix_neck_waist"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(manus_launch),
                condition=IfCondition(LaunchConfiguration("start_manus_node")),
            ),
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
                        "command_joint_states_topic": command_topic,
                        "physical_joint_states_topic": physical_topic,
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
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(foxglove_launch),
                condition=IfCondition(LaunchConfiguration("start_foxglove_node")),
                launch_arguments={
                    "adam_joint_states_topic": command_topic,
                    "sharpa_joint_states_topic": sharpa_command_topic,
                }.items(),
            ),
        ]
    )
