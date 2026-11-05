from ur_common.recording_config import DEFAULT_HIL_RECORDING_ROOT
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="8080"),
        DeclareLaunchArgument(
            "recording_root",
            default_value=DEFAULT_HIL_RECORDING_ROOT,
        ),
        DeclareLaunchArgument("reference_dataset_root", default_value=DEFAULT_HIL_RECORDING_ROOT),
        DeclareLaunchArgument("upload_host", default_value="BAAI"),
        DeclareLaunchArgument(
            "upload_path",
            default_value="/share/project/yenaisheng/datasets/postrain/ur-sharpa/raw_data",
        ),
        DeclareLaunchArgument("status_timeout_s", default_value="2.0"),
        Node(package="monitor_node", executable="monitor_node", name="monitor_node",
             output="screen", parameters=[{
                 "port": LaunchConfiguration("port"),
                 "recording_root": LaunchConfiguration("recording_root"),
                 "reference_dataset_root": LaunchConfiguration("reference_dataset_root"),
                 "upload_host": LaunchConfiguration("upload_host"),
                 "upload_path": LaunchConfiguration("upload_path"),
                 "status_timeout_s": LaunchConfiguration("status_timeout_s"),
             }]),
    ])
