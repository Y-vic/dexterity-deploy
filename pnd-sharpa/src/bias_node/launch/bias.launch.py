"""Launch the bias UI."""

from __future__ import annotations

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


BIAS_TARGET = "/adam_bias_command_joint_states"
PND_CONTROL_JOINT_STATES = "/adam_bias_command_joint_states"
ROBOT_STATES = "/adam_physical_joint_states"
DEFAULT_BIAS_PATH = "/home/pnd-humanoid/.adam/joint/bias_joints_set_with_init.json"


def default_urdf_path() -> str:
    try:
        share = get_package_share_directory("adam_sharpa_description")
    except PackageNotFoundError:
        return ""
    return os.path.join(share, "urdf", "adam_pro_sharpa", "adam_pro_sharpa.urdf")


def generate_launch_description() -> LaunchDescription:
    bind_host = LaunchConfiguration("bind_host")
    bind_port = LaunchConfiguration("bind_port")
    bias_path = LaunchConfiguration("bias_path")
    urdf_path = LaunchConfiguration("urdf_path")

    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            DeclareLaunchArgument(
                "bind_host",
                default_value="127.0.0.1",
                description="HTTP UI bind host.",
            ),
            DeclareLaunchArgument(
                "bind_port",
                default_value="18080",
                description="HTTP UI bind port.",
            ),
            DeclareLaunchArgument(
                "bias_path",
                default_value=DEFAULT_BIAS_PATH,
                description="Saved bias JSON file.",
            ),
            DeclareLaunchArgument(
                "startup_sequence_mode",
                default_value="bias_init_then_bias",
                choices=["bias_init_then_bias", "bias"],
            ),
            DeclareLaunchArgument(
                "urdf_path",
                default_value=default_urdf_path(),
                description="URDF file used for editable joint limits.",
            ),
            Node(
                package="bias_node",
                executable="bias",
                name="bias",
                output="screen",
                parameters=[
                    {
                        "bind_host": bind_host,
                        "bind_port": bind_port,
                        "bias_path": bias_path,
                        "urdf_path": urdf_path,
                        "output_topic": BIAS_TARGET,
                        "robot_state_topic": ROBOT_STATES,
                        "command_state_topic": PND_CONTROL_JOINT_STATES,
                        "control_status_topic": "/control_status",
                        "status_topic": "/bias/status",
                        "public_url_path": "/bias_joints",
                        "publish_rate_hz": 30.0,
                        "startup_publish_delay_s": 0.25,
                        "startup_on_t_init": True,
                        "startup_sequence_mode": LaunchConfiguration(
                            "startup_sequence_mode"
                        ),
                        "arrival_tolerance_deg": 8.0,
                        "robot_state_timeout_s": 0.5,
                        "sequence_step_timeout_s": 3.0,
                        "interpolation_threshold_deg": 10.0,
                        "interpolation_duration_s": 1.0,
                    }
                ],
            ),
        ]
    )
