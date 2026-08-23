from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sdk_root", default_value=EnvironmentVariable("SHARPA_SDK_ROOT", default_value="")),
        Node(package="sharpa_node", executable="sharpa_node", name="sharpa_node", output="screen", parameters=[{"sdk_root": LaunchConfiguration("sdk_root")}]),
    ])
