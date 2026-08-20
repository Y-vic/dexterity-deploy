"""Launch the Sharpa hardware node."""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _repo_sdk_path(*parts: str) -> str:
    for root in _candidate_workspace_roots():
        sdk_root = root / "external" / "sharpa_control" / "sdk"
        if sdk_root.is_dir():
            return str(sdk_root.joinpath(*parts))
    searched = ", ".join(str(root) for root in _candidate_workspace_roots())
    raise FileNotFoundError(
        "Could not find external/sharpa_control/sdk from PND_WORKSPACE_DIR or cwd; "
        f"searched: {searched}"
    )


def _candidate_workspace_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("PND_WORKSPACE_DIR", "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    cwd = Path(os.getcwd()).resolve()
    roots.extend([cwd, *cwd.parents])
    return list(dict.fromkeys(roots))


def generate_launch_description() -> LaunchDescription:
    connect_sharpa = LaunchConfiguration("connect_sharpa")
    left_sn = LaunchConfiguration("left_sn")
    right_sn = LaunchConfiguration("right_sn")
    left_ip = LaunchConfiguration("left_ip")
    right_ip = LaunchConfiguration("right_ip")
    status_json_topic = LaunchConfiguration("status_json_topic")
    publish_tactile = LaunchConfiguration("publish_tactile")
    tactile_poll_warmup_s = LaunchConfiguration("tactile_poll_warmup_s")
    tactile_fresh_timeout_s = LaunchConfiguration("tactile_fresh_timeout_s")
    tactile_sensor_time_max_age_s = LaunchConfiguration("tactile_sensor_time_max_age_s")
    tactile_error_log_period_s = LaunchConfiguration("tactile_error_log_period_s")
    command_snapshot_topic = LaunchConfiguration("command_snapshot_topic")
    publish_command_snapshot = LaunchConfiguration("publish_command_snapshot")
    command_snapshot_max_hz = LaunchConfiguration("command_snapshot_max_hz")
    command_mode = LaunchConfiguration("command_mode")
    mit_torque_limit = LaunchConfiguration("mit_torque_limit")
    startup_zero_hold_s = LaunchConfiguration("startup_zero_hold_s")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "connect_sharpa",
                default_value="true",
                description="Connect Sharpa hardware on startup.",
            ),
            DeclareLaunchArgument(
                "left_sn",
                default_value=os.environ.get("LEFT_SN", "CF50953ACF51"),
                description="Expected left Sharpa serial number.",
            ),
            DeclareLaunchArgument(
                "right_sn",
                default_value=os.environ.get("RIGHT_SN", "C8559538C854"),
                description="Expected right Sharpa serial number.",
            ),
            DeclareLaunchArgument(
                "left_ip",
                default_value=os.environ.get("LEFT_HAND_IP", "10.10.10.201"),
                description="Expected left Sharpa device IP. Empty disables validation.",
            ),
            DeclareLaunchArgument(
                "right_ip",
                default_value=os.environ.get("RIGHT_HAND_IP", "10.10.10.202"),
                description="Expected right Sharpa device IP. Empty disables validation.",
            ),
            DeclareLaunchArgument(
                "status_json_topic",
                default_value="/teleop/status_json",
                description="Status JSON topic used to detect active recording.",
            ),
            DeclareLaunchArgument(
                "publish_tactile",
                default_value="true",
                description="Publish Sharpa tactile aggregate deform/force/contact topics.",
            ),
            DeclareLaunchArgument(
                "tactile_poll_warmup_s",
                default_value="5.0",
                description="Delay tactile fetch/summary calls after SDK start.",
            ),
            DeclareLaunchArgument(
                "tactile_fresh_timeout_s",
                default_value="0.25",
                description="Maximum age since a channel's last new tactile frame before publishing an empty entry.",
            ),
            DeclareLaunchArgument(
                "tactile_sensor_time_max_age_s",
                default_value="1.0",
                description="Maximum wall-clock age of SDK tactile sensor_time before treating the frame as stale.",
            ),
            DeclareLaunchArgument(
                "tactile_error_log_period_s",
                default_value="1.0",
                description="Minimum seconds between repeated recording tactile error logs.",
            ),
            DeclareLaunchArgument(
                "command_snapshot_topic",
                default_value="/sharpa_command_snapshot",
                description="Debug snapshot topic emitted when sharpa_node receives a command.",
            ),
            DeclareLaunchArgument(
                "publish_command_snapshot",
                default_value="true",
                description="Publish command-time q_cmd/q_exe/tactile snapshots.",
            ),
            DeclareLaunchArgument(
                "command_snapshot_max_hz",
                default_value="30.0",
                description="Maximum command snapshot publish rate. Use 0 for every received command.",
            ),
            DeclareLaunchArgument(
                "command_mode",
                default_value="position",
                description="Sharpa command mode: position, mit, or auto.",
            ),
            DeclareLaunchArgument(
                "mit_torque_limit",
                default_value="2.0",
                description="Absolute torque clamp for MIT command effort values.",
            ),
            DeclareLaunchArgument(
                "startup_zero_hold_s",
                default_value="0.0",
                description="Seconds to keep commanding both hands to zero after startup.",
            ),
            Node(
                package="sharpa_node",
                executable="sharpa",
                name="sharpa",
                output="screen",
                parameters=[
                    {
                        "sdk_python_path": os.environ.get(
                            "SHARPA_WAVE_SDK_PYTHON",
                            _repo_sdk_path("sharpa-wave-sdk", "python"),
                        ),
                        "left_sn": left_sn,
                        "right_sn": right_sn,
                        "left_ip": left_ip,
                        "right_ip": right_ip,
                        "status_topic": "/control_status",
                        "status_json_topic": status_json_topic,
                        "retargeted_joints_topic": "/sharpa_command_joint_states",
                        "joint_states_topic": "/sharpa_physical_joint_states",
                        "command_snapshot_topic": command_snapshot_topic,
                        "publish_command_snapshot": ParameterValue(
                            publish_command_snapshot,
                            value_type=bool,
                        ),
                        "command_snapshot_max_hz": ParameterValue(
                            command_snapshot_max_hz,
                            value_type=float,
                        ),
                        "sharpa_status_topic": "/sharpa_physical_status",
                        "tactile_topic_prefix": "/sharpa_physical_tactile",
                        "tactile_status_topic": "/sharpa_physical_tactile_status",
                        "connect_on_start": connect_sharpa,
                        "initial_mode": "zero",
                        "publish_tactile": publish_tactile,
                        "tactile_poll_warmup_s": tactile_poll_warmup_s,
                        "tactile_fresh_timeout_s": ParameterValue(
                            tactile_fresh_timeout_s,
                            value_type=float,
                        ),
                        "tactile_sensor_time_max_age_s": ParameterValue(
                            tactile_sensor_time_max_age_s,
                            value_type=float,
                        ),
                        "tactile_error_log_period_s": ParameterValue(
                            tactile_error_log_period_s,
                            value_type=float,
                        ),
                        "command_mode": command_mode,
                        "mit_torque_limit": ParameterValue(
                            mit_torque_limit,
                            value_type=float,
                        ),
                        "zero_on_shutdown": False,
                        "zero_when_target_stale": False,
                        "startup_zero_hold_s": ParameterValue(
                            startup_zero_hold_s,
                            value_type=float,
                        ),
                    }
                ],
            )
        ]
    )
