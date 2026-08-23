from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="robot_states_node", executable="robot_states_node", name="robot_states_node", output="screen")
    ])
