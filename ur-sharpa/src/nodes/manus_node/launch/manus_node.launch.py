from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(package="manus_node", executable="manus_node", name="manus_node", output="screen")])
