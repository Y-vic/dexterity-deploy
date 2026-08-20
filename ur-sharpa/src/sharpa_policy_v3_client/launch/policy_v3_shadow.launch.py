from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("sharpa_policy_v3_client"))
    default_config = str(package_share / "config" / "policy_v3_shadow.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument(
                "server_base_url",
                default_value="http://127.0.0.1:5500",
            ),
            Node(
                package="sharpa_policy_v3_client",
                executable="state_node",
                name="state_node",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
            Node(
                package="sharpa_policy_v3_client",
                executable="policy_node",
                name="policy_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {"server_base_url": LaunchConfiguration("server_base_url")},
                ],
            ),
        ]
    )
