from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("zmq_address", default_value="tcp://172.16.13.167:5555"),
        DeclareLaunchArgument("frequency", default_value="30.0"),
        DeclareLaunchArgument("pose_filter_enabled", default_value="true"),
        DeclareLaunchArgument("position_deadband_m", default_value="0.0015"),
        DeclareLaunchArgument("rotation_deadband_rad", default_value="0.0087266463"),
        DeclareLaunchArgument("position_min_cutoff_hz", default_value="1.5"),
        DeclareLaunchArgument("position_beta", default_value="5.0"),
        DeclareLaunchArgument("rotation_min_cutoff_hz", default_value="2.0"),
        DeclareLaunchArgument("rotation_beta", default_value="1.5"),
        DeclareLaunchArgument("derivative_cutoff_hz", default_value="1.0"),
        DeclareLaunchArgument("max_translation_delta_m", default_value="0.03"),
        DeclareLaunchArgument("max_rotation_delta_rad", default_value="0.12"),
        Node(
            package="vive_node",
            executable="vive_node",
            name="vive_node",
            output="screen",
            parameters=[{
                "zmq_address": LaunchConfiguration("zmq_address"),
                "frequency": ParameterValue(
                    LaunchConfiguration("frequency"), value_type=float
                ),
                "pose_filter_enabled": ParameterValue(
                    LaunchConfiguration("pose_filter_enabled"), value_type=bool
                ),
                "position_deadband_m": ParameterValue(
                    LaunchConfiguration("position_deadband_m"), value_type=float
                ),
                "rotation_deadband_rad": ParameterValue(
                    LaunchConfiguration("rotation_deadband_rad"), value_type=float
                ),
                "position_min_cutoff_hz": ParameterValue(
                    LaunchConfiguration("position_min_cutoff_hz"), value_type=float
                ),
                "position_beta": ParameterValue(
                    LaunchConfiguration("position_beta"), value_type=float
                ),
                "rotation_min_cutoff_hz": ParameterValue(
                    LaunchConfiguration("rotation_min_cutoff_hz"), value_type=float
                ),
                "rotation_beta": ParameterValue(
                    LaunchConfiguration("rotation_beta"), value_type=float
                ),
                "derivative_cutoff_hz": ParameterValue(
                    LaunchConfiguration("derivative_cutoff_hz"), value_type=float
                ),
                "max_translation_delta_m": ParameterValue(
                    LaunchConfiguration("max_translation_delta_m"), value_type=float
                ),
                "max_rotation_delta_rad": ParameterValue(
                    LaunchConfiguration("max_rotation_delta_rad"), value_type=float
                ),
            }],
        ),
    ])
