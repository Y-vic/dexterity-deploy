from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ego_source", default_value=""), DeclareLaunchArgument("left_wrist_source", default_value=""), DeclareLaunchArgument("right_wrist_source", default_value=""),
        Node(package="camera_node", executable="camera_node", name="camera_node", output="screen", parameters=[{"ego_source": LaunchConfiguration("ego_source"), "left_wrist_source": LaunchConfiguration("left_wrist_source"), "right_wrist_source": LaunchConfiguration("right_wrist_source")}]),
    ])
