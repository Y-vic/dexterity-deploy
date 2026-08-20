"""Launch Foxglove command-target visualization."""

from __future__ import annotations

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def default_urdf_path() -> str:
    try:
        share = get_package_share_directory("adam_sharpa_description")
    except PackageNotFoundError:
        return os.path.abspath(
            os.path.join(
                os.getcwd(),
                "src",
                "adam_sharpa_description",
                "urdf",
                "adam_pro_sharpa",
                "adam_pro_sharpa.urdf",
            )
        )
    return os.path.join(share, "urdf", "adam_pro_sharpa", "adam_pro_sharpa.urdf")


def read_urdf(path: str) -> str:
    with open(path, "r", encoding="utf-8") as infp:
        return infp.read()


def generate_launch_description() -> LaunchDescription:
    adam_topic = LaunchConfiguration("adam_joint_states_topic")
    sharpa_topic = LaunchConfiguration("sharpa_joint_states_topic")
    visualization_topic = LaunchConfiguration("visualization_joint_states_topic")
    status_topic = LaunchConfiguration("status_topic")
    urdf_path = LaunchConfiguration("urdf_path")
    publish_rate = LaunchConfiguration("publish_rate")
    bridge_port = LaunchConfiguration("bridge_port")
    bridge_address = LaunchConfiguration("bridge_address")
    world_frame = LaunchConfiguration("world_frame")
    robot_root_frame = LaunchConfiguration("robot_root_frame")

    def launch_nodes(context):
        resolved_urdf_path = urdf_path.perform(context)
        return [
            Node(
                package="foxglove_node",
                executable="foxglove_joint_state_merge",
                name="foxglove_joint_state_merge",
                output="screen",
                parameters=[
                    {
                        "adam_joint_states_topic": adam_topic,
                        "sharpa_joint_states_topic": sharpa_topic,
                        "output_topic": visualization_topic,
                        "status_topic": status_topic,
                        "urdf_path": resolved_urdf_path,
                        "publish_rate": ParameterValue(publish_rate, value_type=float),
                    }
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="foxglove_world_to_robot_root",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("start_world_static_transform")
                ),
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    world_frame,
                    robot_root_frame,
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="foxglove_robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": read_urdf(resolved_urdf_path),
                        "publish_frequency": ParameterValue(
                            publish_rate,
                            value_type=float,
                        ),
                    }
                ],
                remappings=[("joint_states", visualization_topic)],
            ),
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
                parameters=[
                    {
                        "port": ParameterValue(bridge_port, value_type=int),
                        "address": bridge_address,
                        "tls": False,
                        "topic_whitelist": [
                            "^/foxglove/joint_states$",
                            "^/foxglove/status$",
                            "^/_quest/retargeted_joint_states_raw$",
                            "^/_quest/tf$",
                            "^/_quest/joy$",
                            "^/quest/webvr_status$",
                            "^/quest/retarget_status$",
                            "^/quest/command_status$",
                            "^/adam_command_joint_states$",
                            "^/robot_description$",
                            "^/tf$",
                            "^/tf_static$",
                        ],
                        "service_whitelist": [".*"],
                        "param_whitelist": [".*"],
                        "client_topic_whitelist": [".*"],
                        "min_qos_depth": 1,
                        "max_qos_depth": 10,
                        "num_threads": 0,
                        "send_buffer_limit": 10000000,
                        "capabilities": [
                            "clientPublish",
                            "parameters",
                            "parametersSubscribe",
                            "services",
                            "connectionGraph",
                            "assets",
                        ],
                        "include_hidden": True,
                    }
                ],
            ),
        ]

    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            DeclareLaunchArgument(
                "adam_joint_states_topic",
                default_value="/adam_command_joint_states",
            ),
            DeclareLaunchArgument(
                "sharpa_joint_states_topic",
                default_value="/sharpa_command_joint_states",
            ),
            DeclareLaunchArgument(
                "visualization_joint_states_topic",
                default_value="/foxglove/joint_states",
            ),
            DeclareLaunchArgument("status_topic", default_value="/foxglove/status"),
            DeclareLaunchArgument("urdf_path", default_value=default_urdf_path()),
            DeclareLaunchArgument("publish_rate", default_value="60.0"),
            DeclareLaunchArgument("bridge_port", default_value="8765"),
            DeclareLaunchArgument("bridge_address", default_value="0.0.0.0"),
            DeclareLaunchArgument("start_world_static_transform", default_value="true"),
            DeclareLaunchArgument("world_frame", default_value="world"),
            DeclareLaunchArgument("robot_root_frame", default_value="pelvis"),
            OpaqueFunction(function=launch_nodes),
        ]
    )
