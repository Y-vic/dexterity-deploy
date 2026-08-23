from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="teleop"),
        DeclareLaunchArgument("web_enabled", default_value="false"),
        DeclareLaunchArgument("web_port", default_value="8080"),
        Node(package="status_node", executable="status_node", name="status_node", output="screen", parameters=[{
            "mode": LaunchConfiguration("mode"), "web_enabled": LaunchConfiguration("web_enabled"),
            "web_port": LaunchConfiguration("web_port"),
        }]),
    ])
