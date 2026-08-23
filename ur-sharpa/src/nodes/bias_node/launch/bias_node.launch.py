from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ur_steps", default_value="100"),
        DeclareLaunchArgument("sharpa_steps", default_value="100"),
        DeclareLaunchArgument("enable_jump_gate", default_value="false"),
        DeclareLaunchArgument("enable_flip_hold", default_value="false"),
        DeclareLaunchArgument("enable_joint_clip", default_value="false"),
        DeclareLaunchArgument("bias_file", default_value="~/.config/ur_sharpa/teleop_bias.json"),
        Node(
            package="bias_node",
            executable="bias_node",
            name="bias_node",
            output="screen",
            parameters=[{
                "ur_steps": LaunchConfiguration("ur_steps"),
                "sharpa_steps": LaunchConfiguration("sharpa_steps"),
                "enable_jump_gate": LaunchConfiguration("enable_jump_gate"),
                "enable_flip_hold": LaunchConfiguration("enable_flip_hold"),
                "enable_joint_clip": LaunchConfiguration("enable_joint_clip"),
                "bias_file": LaunchConfiguration("bias_file"),
            }],
        ),
    ])
