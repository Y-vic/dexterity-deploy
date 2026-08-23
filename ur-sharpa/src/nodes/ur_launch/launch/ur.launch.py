"""Compatibility selector for the explicit Teleop and Deploy entrypoints."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _include(filename: str, mode: str) -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("ur_launch"), "launch", filename])
        ),
        condition=LaunchConfigurationEquals("mode", mode),
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="deploy", choices=["teleop", "deploy"]),
        _include("teleop.launch.py", "teleop"),
        _include("deploy.launch.py", "deploy"),
    ])
