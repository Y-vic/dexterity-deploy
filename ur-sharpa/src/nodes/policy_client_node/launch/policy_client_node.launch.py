from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("server_url", default_value="ws://127.0.0.1:5500/infer"),
        Node(package="policy_client_node", executable="policy_client_node", name="policy_client_node", output="screen", parameters=[{"server_url": LaunchConfiguration("server_url")}]),
    ])
