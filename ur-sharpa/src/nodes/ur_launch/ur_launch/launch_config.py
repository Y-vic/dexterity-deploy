"""Single-source launch composition for Teleop and Deploy."""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


MODES = ("teleop", "deploy")


def _workspace_root() -> Path:
    configured = os.environ.get("UR_SHARPA_ROOT", "").strip()
    starts = [Path(configured).expanduser()] if configured else []
    starts.extend((Path(__file__).resolve(), Path.cwd().resolve()))
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate.name == "ur-sharpa" and (candidate / "src" / "nodes").is_dir():
                return candidate
    raise RuntimeError(
        "cannot locate the ur-sharpa workspace; source scripts/setup_env.sh first"
    )


def _argument(name: str, default: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=default)


def _common_arguments(workspace_root: Path) -> list[DeclareLaunchArgument]:
    external_root = workspace_root / "external"
    return [
        _argument("hardware_python_paths", str(external_root / "realsense-python")),
        _argument("sharpa_sdk_root", str(external_root / "sharpa-wave-sdk-5.0.7")),
        _argument("sharpa_skip_tactile", "false"),
        _argument("enable_payload_configuration", "true"),
        _argument("left_payload_mass_kg", "1.3"),
        _argument("left_payload_cog_m", "[0.0, 0.0, 0.08]"),
        _argument("right_payload_mass_kg", "1.3"),
        _argument("right_payload_cog_m", "[0.0, 0.0, 0.08]"),
        _argument("camera_use_hardware", "true"),
        _argument("camera_start_delay_s", "5.0"),
        _argument("ego_zed_resolution", "HD720"),
        _argument("ego_zed_fps", "30"),
        _argument("ego_zed_serial_number", "0"),
        _argument("ego_first_frame_timeout_s", "5.0"),
        _argument("ego_frame_stall_timeout_s", "3.0"),
        _argument("camera_diagnostic_interval_s", "10.0"),
        _argument("left_wrist_serial", "230322270627"),
        _argument("right_wrist_serial", "218622279586"),
        _argument("wrist_camera_fps", "30"),
        _argument("wrist_camera_width", "480"),
        _argument("wrist_camera_height", "270"),
        _argument("camera_preview_fps", "10.0"),
        _argument("camera_shm_prefix", "ur_sharpa_camera_v1"),
        _argument("camera_shm_bytes", "4194304"),
        _argument("bias_file", "~/.config/ur_sharpa/teleop_bias.json"),
        _argument("interface_path", str(workspace_root.parent / "interface")),
        _argument("manus_python_path", str(external_root / "manus-python")),
        _argument(
            "pinocchio_python_path",
            str(external_root / "pinocchio-runtime" / "lib" / "python3.10" / "site-packages"),
        ),
        _argument("pinocchio_library_path", str(external_root / "pinocchio-runtime" / "lib")),
    ]


def _environment() -> list[SetEnvironmentVariable]:
    return [
        SetEnvironmentVariable(
            "PYTHONPATH",
            [
                LaunchConfiguration("manus_python_path"),
                ":",
                LaunchConfiguration("pinocchio_python_path"),
                ":",
                LaunchConfiguration("interface_path"),
                ":",
                EnvironmentVariable("PYTHONPATH", default_value=""),
            ],
        ),
        SetEnvironmentVariable(
            "LD_LIBRARY_PATH",
            [
                LaunchConfiguration("pinocchio_library_path"),
                ":",
                EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
            ],
        ),
        SetEnvironmentVariable(
            "UR_SHARPA_PYTHON_PATHS", LaunchConfiguration("hardware_python_paths")
        ),
    ]


def _shared_nodes(mode: str) -> list:
    camera = Node(
        package="camera_node",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=[{
            "use_hardware": ParameterValue(LaunchConfiguration("camera_use_hardware"), value_type=bool),
            "ego_zed_resolution": LaunchConfiguration("ego_zed_resolution"),
            "ego_zed_fps": ParameterValue(LaunchConfiguration("ego_zed_fps"), value_type=int),
            "ego_zed_serial_number": ParameterValue(
                LaunchConfiguration("ego_zed_serial_number"), value_type=int
            ),
            "ego_first_frame_timeout_s": ParameterValue(
                LaunchConfiguration("ego_first_frame_timeout_s"), value_type=float
            ),
            "ego_frame_stall_timeout_s": ParameterValue(
                LaunchConfiguration("ego_frame_stall_timeout_s"), value_type=float
            ),
            "diagnostic_interval_s": ParameterValue(
                LaunchConfiguration("camera_diagnostic_interval_s"), value_type=float
            ),
            "left_wrist_serial": ParameterValue(LaunchConfiguration("left_wrist_serial"), value_type=str),
            "right_wrist_serial": ParameterValue(LaunchConfiguration("right_wrist_serial"), value_type=str),
            "wrist_camera_fps": ParameterValue(LaunchConfiguration("wrist_camera_fps"), value_type=int),
            "wrist_camera_width": ParameterValue(LaunchConfiguration("wrist_camera_width"), value_type=int),
            "wrist_camera_height": ParameterValue(LaunchConfiguration("wrist_camera_height"), value_type=int),
            "preview_fps": ParameterValue(LaunchConfiguration("camera_preview_fps"), value_type=float),
            "shared_memory_prefix": LaunchConfiguration("camera_shm_prefix"),
            "shared_memory_bytes": ParameterValue(LaunchConfiguration("camera_shm_bytes"), value_type=int),
        }],
    )
    return [
        Node(
            package="ur_node",
            executable="ur_node",
            name="ur_node",
            output="screen",
            parameters=[{
                "enable_motion_limits": True,
                "enable_joint_step_limit": mode == "deploy",
                "enable_joint_clip": mode == "deploy",
                "enable_payload_configuration": ParameterValue(
                    LaunchConfiguration("enable_payload_configuration"), value_type=bool
                ),
                "left_payload_mass_kg": ParameterValue(
                    LaunchConfiguration("left_payload_mass_kg"), value_type=float
                ),
                "left_payload_cog_m": ParameterValue(
                    LaunchConfiguration("left_payload_cog_m"), value_type=list[float]
                ),
                "right_payload_mass_kg": ParameterValue(
                    LaunchConfiguration("right_payload_mass_kg"), value_type=float
                ),
                "right_payload_cog_m": ParameterValue(
                    LaunchConfiguration("right_payload_cog_m"), value_type=list[float]
                ),
            }],
        ),
        Node(
            package="sharpa_node",
            executable="sharpa_node",
            name="sharpa_node",
            output="screen",
            parameters=[{
                "sdk_root": LaunchConfiguration("sharpa_sdk_root"),
                "skip_tactile": ParameterValue(LaunchConfiguration("sharpa_skip_tactile"), value_type=bool),
            }],
        ),
        TimerAction(period=LaunchConfiguration("camera_start_delay_s"), actions=[camera]),
        Node(
            package="bias_node",
            executable="bias_node",
            name="bias_node",
            output="screen",
            parameters=[{
                "enable_jump_gate": True,
                "enable_flip_hold": True,
                "enable_joint_clip": True,
                "bias_file": LaunchConfiguration("bias_file"),
            }],
        ),
        Node(
            package="status_node",
            executable="status_node",
            name="status_node",
            output="screen",
            parameters=[{
                "mode": mode,
                "web_enabled": mode == "deploy",
                "web_port": 8080,
            }],
        ),
    ]


def _teleop_arguments(workspace_root: Path) -> list[DeclareLaunchArgument]:
    external_root = workspace_root / "external"
    return [
        _argument("recording_root", str(workspace_root / "episodes")),
        _argument("monitor_port", "8080"),
        _argument("vive_zmq_address", "tcp://172.16.13.167:5555"),
        _argument("vive_pose_filter_enabled", "true"),
        _argument("vive_position_deadband_m", "0.0015"),
        _argument("vive_rotation_deadband_rad", "0.0087266463"),
        _argument("vive_position_min_cutoff_hz", "1.5"),
        _argument("vive_position_beta", "5.0"),
        _argument("vive_rotation_min_cutoff_hz", "2.0"),
        _argument("vive_rotation_beta", "1.5"),
        _argument("vive_derivative_cutoff_hz", "1.0"),
        _argument("vive_max_translation_delta_m", "0.03"),
        _argument("vive_max_rotation_delta_rad", "0.12"),
        _argument("manus_sdk_root", str(external_root / "sharpa-manus-sdk")),
        _argument("manus_retarget_root", str(external_root / "manus-retarget")),
        _argument("manus_retarget_python", "/usr/bin/python3"),
        _argument("manus_start_hardware", "true"),
        _argument("manus_client_use_sudo", "false"),
        _argument("manus_restart_enabled", "true"),
        _argument("manus_restart_cooldown_s", "3.0"),
        _argument("manus_restart_max_attempts", "3"),
        _argument("manus_restart_window_s", "60.0"),
        _argument("manus_restart_slow_interval_s", "30.0"),
        _argument("manus_startup_grace_s", "8.0"),
        _argument("manus_raw_timeout_s", "2.0"),
        _argument("manus_incomplete_hand_timeout_s", "1.0"),
        _argument("manus_action_timeout_s", "2.0"),
        _argument("upload_host", "BAAI"),
        _argument(
            "upload_path",
            "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/raw_data",
        ),
        _argument("status_timeout_s", "2.0"),
    ]


def _teleop_nodes() -> list[Node]:
    return [
        Node(
            package="vive_node",
            executable="vive_node",
            name="vive_node",
            output="screen",
            parameters=[{
                "zmq_address": LaunchConfiguration("vive_zmq_address"),
                "pose_filter_enabled": ParameterValue(
                    LaunchConfiguration("vive_pose_filter_enabled"), value_type=bool
                ),
                "position_deadband_m": ParameterValue(
                    LaunchConfiguration("vive_position_deadband_m"), value_type=float
                ),
                "rotation_deadband_rad": ParameterValue(
                    LaunchConfiguration("vive_rotation_deadband_rad"), value_type=float
                ),
                "position_min_cutoff_hz": ParameterValue(
                    LaunchConfiguration("vive_position_min_cutoff_hz"), value_type=float
                ),
                "position_beta": ParameterValue(
                    LaunchConfiguration("vive_position_beta"), value_type=float
                ),
                "rotation_min_cutoff_hz": ParameterValue(
                    LaunchConfiguration("vive_rotation_min_cutoff_hz"), value_type=float
                ),
                "rotation_beta": ParameterValue(
                    LaunchConfiguration("vive_rotation_beta"), value_type=float
                ),
                "derivative_cutoff_hz": ParameterValue(
                    LaunchConfiguration("vive_derivative_cutoff_hz"), value_type=float
                ),
                "max_translation_delta_m": ParameterValue(
                    LaunchConfiguration("vive_max_translation_delta_m"), value_type=float
                ),
                "max_rotation_delta_rad": ParameterValue(
                    LaunchConfiguration("vive_max_rotation_delta_rad"), value_type=float
                ),
            }],
        ),
        Node(
            package="manus_node",
            executable="manus_node",
            name="manus_node",
            output="screen",
            parameters=[{
                "sdk_root": LaunchConfiguration("manus_sdk_root"),
                "retarget_root": LaunchConfiguration("manus_retarget_root"),
                "python_path": LaunchConfiguration("manus_python_path"),
                "retarget_python": LaunchConfiguration("manus_retarget_python"),
                "start_hardware": ParameterValue(LaunchConfiguration("manus_start_hardware"), value_type=bool),
                "client_use_sudo": ParameterValue(LaunchConfiguration("manus_client_use_sudo"), value_type=bool),
                "restart_enabled": ParameterValue(LaunchConfiguration("manus_restart_enabled"), value_type=bool),
                "restart_cooldown_s": ParameterValue(LaunchConfiguration("manus_restart_cooldown_s"), value_type=float),
                "restart_max_attempts": ParameterValue(LaunchConfiguration("manus_restart_max_attempts"), value_type=int),
                "restart_window_s": ParameterValue(LaunchConfiguration("manus_restart_window_s"), value_type=float),
                "restart_slow_interval_s": ParameterValue(LaunchConfiguration("manus_restart_slow_interval_s"), value_type=float),
                "startup_grace_s": ParameterValue(LaunchConfiguration("manus_startup_grace_s"), value_type=float),
                "raw_timeout_s": ParameterValue(LaunchConfiguration("manus_raw_timeout_s"), value_type=float),
                "incomplete_hand_timeout_s": ParameterValue(LaunchConfiguration("manus_incomplete_hand_timeout_s"), value_type=float),
                "action_timeout_s": ParameterValue(LaunchConfiguration("manus_action_timeout_s"), value_type=float),
            }],
        ),
        Node(
            package="recording_node",
            executable="recording_node",
            name="recording_node",
            output="screen",
            parameters=[{
                "recording_root": LaunchConfiguration("recording_root"),
                "frequency": 30.0,
                "shared_memory_prefix": LaunchConfiguration("camera_shm_prefix"),
            }],
        ),
        Node(
            package="monitor_node",
            executable="monitor_node",
            name="monitor_node",
            output="screen",
            parameters=[{
                "port": ParameterValue(LaunchConfiguration("monitor_port"), value_type=int),
                "recording_root": LaunchConfiguration("recording_root"),
                "upload_host": LaunchConfiguration("upload_host"),
                "upload_path": LaunchConfiguration("upload_path"),
                "status_timeout_s": ParameterValue(LaunchConfiguration("status_timeout_s"), value_type=float),
                "shared_memory_prefix": LaunchConfiguration("camera_shm_prefix"),
            }],
        ),
    ]


def _deploy_arguments() -> list[DeclareLaunchArgument]:
    return [
        _argument("dashboard_port", "8088"),
        _argument("policy_server_url", "ws://127.0.0.1:5500/infer"),
        _argument("policy_prompt", ""),
    ]


def _deploy_nodes() -> list[Node]:
    return [
        Node(
            package="dashboard_node",
            executable="dashboard_node",
            name="dashboard_node",
            output="screen",
            parameters=[{
                "port": ParameterValue(LaunchConfiguration("dashboard_port"), value_type=int)
            }],
        ),
        Node(
            package="robot_states_node",
            executable="robot_states_node",
            name="robot_states_node",
            output="screen",
        ),
        Node(
            package="policy_client_node",
            executable="policy_client_node",
            name="policy_client_node",
            output="screen",
            parameters=[{
                "server_url": LaunchConfiguration("policy_server_url"),
                "prompt": LaunchConfiguration("policy_prompt"),
            }],
        ),
        Node(
            package="actor_ik_node",
            executable="actor_ik_node",
            name="actor_ik_node",
            output="screen",
        ),
    ]


def build_launch_description(mode: str) -> LaunchDescription:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, received {mode!r}")
    workspace_root = _workspace_root()
    mode_arguments = (
        _teleop_arguments(workspace_root) if mode == "teleop" else _deploy_arguments()
    )
    mode_nodes = _teleop_nodes() if mode == "teleop" else _deploy_nodes()
    return LaunchDescription([
        *_common_arguments(workspace_root),
        *mode_arguments,
        *_environment(),
        *_shared_nodes(mode),
        *mode_nodes,
    ])
