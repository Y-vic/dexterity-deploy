from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package = "sharpa_policy_v3_client"
    package_share = Path(get_package_share_directory(package))
    default_config = str(package_share / "config" / "hardware_v3.yaml")
    config = LaunchConfiguration("config")
    enabled = ParameterValue(LaunchConfiguration("enable_execution"), value_type=bool)
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument(
                "server_base_url", default_value="http://127.0.0.1:5500"
            ),
            DeclareLaunchArgument("enable_execution", default_value="false"),
            DeclareLaunchArgument("ur_confirmation", default_value=""),
            DeclareLaunchArgument("sharpa_confirmation", default_value=""),
            DeclareLaunchArgument("action_confirmation", default_value=""),
            Node(
                package=package,
                executable="ur_node",
                name="ur_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "enable_execution": enabled,
                        "execution_confirmation": LaunchConfiguration(
                            "ur_confirmation"
                        ),
                    },
                ],
            ),
            Node(
                package=package,
                executable="sharpa_node",
                name="sharpa_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "enable_execution": enabled,
                        "execution_confirmation": LaunchConfiguration(
                            "sharpa_confirmation"
                        ),
                    },
                ],
            ),
            Node(
                package=package,
                executable="zed_node",
                name="zed_node",
                output="screen",
                parameters=[config],
            ),
            Node(
                package=package,
                executable="state_node",
                name="state_node",
                output="screen",
                parameters=[config],
            ),
            Node(
                package=package,
                executable="policy_node",
                name="policy_node",
                output="screen",
                parameters=[
                    config,
                    {"server_base_url": LaunchConfiguration("server_base_url")},
                ],
            ),
            Node(
                package=package,
                executable="action_node",
                name="action_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "enable_execution": enabled,
                        "execution_confirmation": LaunchConfiguration(
                            "action_confirmation"
                        ),
                    },
                ],
            ),
        ]
    )
