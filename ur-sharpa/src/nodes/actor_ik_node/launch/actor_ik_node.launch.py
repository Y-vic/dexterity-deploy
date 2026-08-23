from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(
            package="actor_ik_node",
            executable="actor_ik_node",
            name="actor_ik_node",
            output="screen",
        )])
