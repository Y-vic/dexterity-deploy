"""Single-source launch composition for Teleop and Deploy."""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription, Substitution
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction, OpaqueFunction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ur_common.recording_config import DEFAULT_HIL_RECORDING_ROOT, RecordingPreferences
from ur_common.protection_config import ENVIRONMENT_VARIABLE, protection_path, load_protection_config


MODES = ("teleop", "deploy", "workbench")


def _small_math_environment():
    """Small robot matrices do not benefit from a BLAS pool per ROS process.

    Apply before native libraries load, only to input/hardware/recording nodes.
    Preserve explicit operator settings; model service processes have their own environment.
    """
    return {name: os.environ.get(name) or '1'
            for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}


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


PROTECTION_ARGUMENTS = {
    'ur_max_abs_joint_rad': ('ur', 'max_abs_joint_rad'),
    'ur_max_joint_step_rad': ('ur', 'max_joint_step_rad'),
    'ego_first_frame_timeout_s': ('camera', 'ego_first_frame_timeout_s'),
    'ego_frame_stall_timeout_s': ('camera', 'ego_frame_stall_timeout_s'),
    'vive_max_translation_delta_m': ('vive', 'max_translation_delta_m'),
    'vive_max_rotation_delta_rad': ('vive', 'max_rotation_delta_rad'),
    'ae01_data_timeout_s': ('ae01', 'data_timeout_s'),
    'hil_return_duration_s': ('hil', 'return_duration_s'),
}


class ProtectionDefault(Substitution):
    def __init__(self, section, key):
        super().__init__()
        self.section, self.key = section, key

    def perform(self, context):
        path = LaunchConfiguration('protection_config', default=str(protection_path())).perform(context)
        return str(load_protection_config(path)[self.section][self.key])

    def describe(self):
        return f'protection_config:{self.section}.{self.key}'


def _validate_protection(context):
    path = LaunchConfiguration('protection_config').perform(context)
    load_protection_config(path)  # Validate before any hardware process starts.
    return [LogInfo(msg=f'Protection config: {Path(path).expanduser().resolve()}')]


class RecordingRootDefault(Substitution):
    def __init__(self, fallback):
        super().__init__()
        self.fallback = fallback

    def perform(self, context):
        return RecordingPreferences().read(self.fallback)

    def describe(self):
        return f'last applied recording directory (fallback: {self.fallback})'


def _argument(name: str, default=None) -> DeclareLaunchArgument:
    if name in PROTECTION_ARGUMENTS:
        default = ProtectionDefault(*PROTECTION_ARGUMENTS[name])
    elif name == 'recording_root':
        default = RecordingRootDefault(default)
    return DeclareLaunchArgument(name, default_value=default)


def _default_bias_file(workspace_root: Path, mode: str) -> str:
    return '~/.config/ur_sharpa/teleop_bias.json'


def _common_arguments(workspace_root: Path, mode: str = "teleop") -> list[DeclareLaunchArgument]:
    external_root = workspace_root / "external"
    return [
        _argument('protection_config', str(protection_path())),
        _argument("hardware_python_paths", str(external_root / "realsense-python")),
        _argument("sharpa_sdk_root", str(external_root / "sharpa-wave-sdk-5.0.7")),
        _argument("sharpa_skip_tactile", "false"),
        _argument("ur_max_abs_joint_rad"),
        _argument("ur_max_joint_step_rad"),
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
        _argument("ego_first_frame_timeout_s"),
        _argument("ego_frame_stall_timeout_s"),
        _argument("camera_diagnostic_interval_s", "10.0"),
        _argument("left_wrist_serial", "230322270627"),
        _argument("wrist_camera_fps", "30"),
        _argument("wrist_camera_width", "480"),
        _argument("wrist_camera_height", "270"),
        _argument("camera_preview_fps", "10.0"),
        _argument("reference_dataset_root", "/home/yns/data/0916_ppp"),
        _argument("camera_shm_prefix", "ur_sharpa_camera_v1"),
        _argument("deformation_shm_prefix", "ur_sharpa_deformation_v1"),
        _argument("recording_alignment_wait_frames", "2.0"),
        _argument("camera_shm_bytes", "4194304"),
        _argument("camera_raw_queue_size", "64"),
        _argument("camera_encoded_queue_size", "256"),
        _argument("camera_shm_slots", "64"),
        _argument("acquisition_log_dir", ""),
        _argument("video_backend", "hevc_nvenc"),
        _argument("ego_video_bitrate_mbps", "30.0"),
        _argument("wrist_video_bitrate_mbps", "12.0"),
        _argument("bias_file", _default_bias_file(workspace_root, mode)),
        _argument(
            "fastdds_ack_profiles_file",
            str(workspace_root / "config" / "fastdds_ack_profiles.xml"),
        ),
        _argument("interface_path", str(workspace_root.parent / "interface")),
        _argument(
            "pinocchio_python_path",
            str(external_root / "pinocchio-runtime" / "lib" / "python3.10" / "site-packages"),
        ),
        _argument("pinocchio_library_path", str(external_root / "pinocchio-runtime" / "lib")),
    ]


def _environment() -> list[SetEnvironmentVariable]:
    return [
        SetEnvironmentVariable(ENVIRONMENT_VARIABLE, LaunchConfiguration('protection_config')),
        SetEnvironmentVariable(
            "PYTHONPATH",
            [
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
        SetEnvironmentVariable(
            "FASTDDS_DEFAULT_PROFILES_FILE",
            LaunchConfiguration("fastdds_ack_profiles_file"),
        ),
        # Keep rmw_fastrtps' existing history memory, synchronous publication,
        # and data-sharing behavior.  The XML contains only exact-topic writer
        # timing profiles; enabling full XML QoS would change unrelated topics.
        SetEnvironmentVariable("RMW_FASTRTPS_USE_QOS_FROM_XML", "0"),
        SetEnvironmentVariable("UR_SHARPA_FASTDDS_HEARTBEAT_MS", "50"),
    ]


def _ur_motion_parameters(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, received {mode!r}")
    return {
        "enable_motion_limits": True,
        "enable_joint_step_limit": True,
        "enable_joint_clip": True,
        "max_abs_joint_rad": ParameterValue(
            LaunchConfiguration("ur_max_abs_joint_rad"), value_type=float
        ),
        "max_joint_step_rad": ParameterValue(
            LaunchConfiguration("ur_max_joint_step_rad"), value_type=float
        ),
    }


def _shared_nodes(mode: str) -> list:
    camera = Node(
        package="camera_node",
        executable="camera_node",
        name="camera_node",
        additional_env=_small_math_environment(),
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
            "wrist_camera_fps": ParameterValue(LaunchConfiguration("wrist_camera_fps"), value_type=int),
            "wrist_camera_width": ParameterValue(LaunchConfiguration("wrist_camera_width"), value_type=int),
            "wrist_camera_height": ParameterValue(LaunchConfiguration("wrist_camera_height"), value_type=int),
            "preview_fps": ParameterValue(LaunchConfiguration("camera_preview_fps"), value_type=float),
            "shared_memory_prefix": LaunchConfiguration("camera_shm_prefix"),
            "shared_memory_bytes": ParameterValue(LaunchConfiguration("camera_shm_bytes"), value_type=int),
            "raw_queue_size": ParameterValue(LaunchConfiguration("camera_raw_queue_size"), value_type=int),
            "encoded_queue_size": ParameterValue(LaunchConfiguration("camera_encoded_queue_size"), value_type=int),
            "shared_memory_slots": ParameterValue(LaunchConfiguration("camera_shm_slots"), value_type=int),
            "acquisition_log_dir": LaunchConfiguration("acquisition_log_dir"),
            "video_backend": LaunchConfiguration("video_backend"),
        }],
    )
    return [
        Node(
            package="ur_node",
            executable="ur_node",
            name="ur_node",
            additional_env=_small_math_environment(),
            output="screen",
            parameters=[{
                **_ur_motion_parameters(mode),
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
            additional_env=_small_math_environment(),
            output="screen",
            parameters=[{
                "sdk_root": LaunchConfiguration("sharpa_sdk_root"),
                "skip_tactile": ParameterValue(LaunchConfiguration("sharpa_skip_tactile"), value_type=bool),
                "deformation_shm_prefix": LaunchConfiguration("deformation_shm_prefix"),
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
                "mode": "teleop" if mode == "workbench" else mode,
                "allow_mode_switch": mode == "workbench",
                "enable_hil": False if mode == "teleop" else ParameterValue(
                    LaunchConfiguration("enable_hil"), value_type=bool),
                "web_enabled": mode != "teleop",
                "web_port": 8080,
            }],
        ),
    ]


def _teleop_arguments(workspace_root: Path) -> list[DeclareLaunchArgument]:
    return [
        _argument("recording_root", "/media/yns/T9/0824_yns_unscrew_cap"),
        _argument("monitor_port", "8080"),
        _argument("vive_zmq_address", "tcp://172.16.13.201:5555"),
        _argument("vive_pose_filter_enabled", "true"),
        _argument('vive_hil_initial_step_guard', 'true'),
        _argument("vive_position_deadband_m", "0.0015"),
        _argument("vive_rotation_deadband_rad", "0.0087266463"),
        _argument("vive_position_min_cutoff_hz", "1.5"),
        _argument("vive_position_beta", "5.0"),
        _argument("vive_rotation_min_cutoff_hz", "2.0"),
        _argument("vive_rotation_beta", "1.5"),
        _argument("vive_derivative_cutoff_hz", "1.0"),
        _argument("vive_max_translation_delta_m"),
        _argument("vive_max_rotation_delta_rad"),
        _argument("ae01_sdk_root", os.environ.get("AE01_SDK_ROOT", "/opt/avatar-sdk")),
        _argument("ae01_sdk_config", [LaunchConfiguration("ae01_sdk_root"), "/share/sdk_config.json"]),
        _argument("ae01_transport", "wireless"),
        _argument("ae01_start_hardware", "true"),
        _argument("ae01_feedback_enabled", "true"),
        _argument("ae01_data_timeout_s"),
        _argument("upload_host", "BAAI"),
        _argument(
            "upload_path",
            "/share/project/yenaisheng/datasets/postrain/ur-sharpa/raw_data",
        ),
        _argument("status_timeout_s", "2.0"),
    ]


def _teleop_nodes(hil=False, workbench=False) -> list[Node]:
    return [
        Node(
            package="vive_node",
            executable="vive_node",
            name="vive_node",
            additional_env=_small_math_environment(),
            output="screen",
            **({'condition': IfCondition(LaunchConfiguration('enable_hil'))} if hil and not workbench else {}),
            parameters=[{
                "hil_mode": hil,
                "workbench_mode": workbench,
                "hil_initial_step_guard": ParameterValue(
                    LaunchConfiguration('vive_hil_initial_step_guard'), value_type=bool),
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
                "max_joint_step_rad": ParameterValue(
                    LaunchConfiguration("ur_max_joint_step_rad"), value_type=float
                ),
            }],
        ),
        Node(
            package="ae01_node",
            executable="ae01_node",
            name="ae01_node",
            additional_env=_small_math_environment(),
            output="screen",
            **({'condition': IfCondition(LaunchConfiguration('enable_hil'))} if hil and not workbench else {}),
            parameters=[{
                "hil_mode": hil,
                "workbench_mode": workbench,
                "sdk_root": LaunchConfiguration("ae01_sdk_root"),
                "sdk_config": LaunchConfiguration("ae01_sdk_config"),
                "transport": LaunchConfiguration("ae01_transport"),
                "start_hardware": ParameterValue(LaunchConfiguration("ae01_start_hardware"), value_type=bool),
                "feedback_enabled": ParameterValue(LaunchConfiguration("ae01_feedback_enabled"), value_type=bool),
                "data_timeout_s": ParameterValue(LaunchConfiguration("ae01_data_timeout_s"), value_type=float),
            }],
        ),
        Node(
            package="recording_node",
            executable="recording_node",
            name="recording_node",
            additional_env=_small_math_environment(),
            output="screen",
            parameters=[{
                "auto_record_rollout": hil,
                "workbench_mode": workbench,
                "recording_root": LaunchConfiguration("recording_root"),
                "skip_tactile": ParameterValue(LaunchConfiguration("sharpa_skip_tactile"), value_type=bool),
                "deformation_shm_prefix": LaunchConfiguration("deformation_shm_prefix"),
                "alignment_wait_frames": ParameterValue(LaunchConfiguration("recording_alignment_wait_frames"), value_type=float),
                "video_backend": LaunchConfiguration("video_backend"),
                "ego_video_bitrate_mbps": ParameterValue(LaunchConfiguration("ego_video_bitrate_mbps"), value_type=float),
                "wrist_video_bitrate_mbps": ParameterValue(LaunchConfiguration("wrist_video_bitrate_mbps"), value_type=float),
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
                "reference_dataset_root": LaunchConfiguration("reference_dataset_root"),
                "upload_host": LaunchConfiguration("upload_host"),
                "upload_path": LaunchConfiguration("upload_path"),
                "status_timeout_s": ParameterValue(LaunchConfiguration("status_timeout_s"), value_type=float),
                "shared_memory_prefix": LaunchConfiguration("camera_shm_prefix"),
            }],
        ),
    ]


def _default_model_config(workspace_root: Path) -> Path:
    local = workspace_root / "config" / "model_servers.local.json"
    return local if local.is_file() else workspace_root / "config" / "model_servers.json"


def _deploy_arguments(workspace_root: Path, workbench=False) -> list[DeclareLaunchArgument]:
    return [
        _argument('enable_hil', 'false'),
        _argument("hil_return_duration_s"),
        _argument('upload_host', 'BAAI'),
        _argument('upload_path', '/share/project/yenaisheng/datasets/postrain/ur-sharpa/raw_data'),
        _argument('recording_root', DEFAULT_HIL_RECORDING_ROOT),
        *[argument for argument in _teleop_arguments(workspace_root)
          if argument.name.startswith(('vive_', 'ae01_'))],
        _argument("dashboard_port", "8088"),
        _argument("demo_port", "8089"),
        _argument("demo_camera_serial", ""),
        _argument("demo_camera_width", "1920"),
        _argument("demo_camera_height", "1080"),
        _argument("demo_camera_fps", "30"),
        _argument("demo_recording_root", str(workspace_root / "demo_recordings")),
        _argument(
            "model_config",
            str(_default_model_config(workspace_root)),
        ),
        _argument(
            "replay_sample_dir",
            "",
        ),
        _argument("replay_execution_steps", "40"),
        _argument("replay_frequency_hz", "30.0"),
        _argument("replay_joint_source", "measured"),
        _argument("replay_loop", "false"),
        _argument("replay_dry_run", "true"),
        _argument("execution_log_root", str(workspace_root / "log" / "deploy")),
        _argument("execution_log_retention_days", "7"),
        _argument("execution_log_max_runs", "100"),
        _argument("execution_log_max_total_bytes", str(5 * 1024 * 1024 * 1024)),
    ]


def _deploy_nodes(workbench=False) -> list[Node]:
    return [
        Node(package="policy_client_node", executable="policy_client_node", name="policy_client_node",
             output="screen", parameters=[{"camera_shm_prefix": LaunchConfiguration("camera_shm_prefix"),
                 "enable_hil": ParameterValue(LaunchConfiguration("enable_hil"), value_type=bool)}]),
        Node(
            package="dashboard_node",
            executable="dashboard_node",
            name="dashboard_node",
            output="screen",
            parameters=[{
                "port": ParameterValue(LaunchConfiguration("dashboard_port"), value_type=int),
                "workbench_mode": workbench,
                "recording_root": LaunchConfiguration("recording_root"),
                "upload_host": LaunchConfiguration("upload_host"),
                "upload_path": LaunchConfiguration("upload_path"),
                "model_config": LaunchConfiguration("model_config"),
                "enable_hil": ParameterValue(LaunchConfiguration("enable_hil"), value_type=bool),
                "reference_dataset_root": LaunchConfiguration("reference_dataset_root"),
                "shared_memory_prefix": LaunchConfiguration("camera_shm_prefix"),
                "demo_api_url": ["http://127.0.0.1:", LaunchConfiguration("demo_port")],
            }],
        ),
        Node(
            package="recording_node",
            executable="demo_node",
            name="demo_node",
            output="screen",
            parameters=[{
                "port": ParameterValue(LaunchConfiguration("demo_port"), value_type=int),
                "serial": ParameterValue(LaunchConfiguration("demo_camera_serial"), value_type=str),
                "width": ParameterValue(LaunchConfiguration("demo_camera_width"), value_type=int),
                "height": ParameterValue(LaunchConfiguration("demo_camera_height"), value_type=int),
                "fps": ParameterValue(LaunchConfiguration("demo_camera_fps"), value_type=int),
                "root": LaunchConfiguration("demo_recording_root"),
                "shared_memory_prefix": LaunchConfiguration("camera_shm_prefix"),
                "left_wrist_serial": ParameterValue(LaunchConfiguration("left_wrist_serial"), value_type=str),
            }],
        ),
        Node(
            package="actor_ik_node",
            executable="actor_ik_node",
            name="actor_ik_node",
            output="screen",
            parameters=[{
                "deformation_shm_prefix": LaunchConfiguration("deformation_shm_prefix"),
                "enable_hil": ParameterValue(LaunchConfiguration("enable_hil"), value_type=bool),
                "hil_return_duration_s": ParameterValue(LaunchConfiguration("hil_return_duration_s"), value_type=float),
                "camera_shm_prefix": LaunchConfiguration("camera_shm_prefix"),
                "execution_log_root": LaunchConfiguration("execution_log_root"),
                "execution_log_retention_days": ParameterValue(
                    LaunchConfiguration("execution_log_retention_days"),
                    value_type=int,
                ),
                "execution_log_max_runs": ParameterValue(
                    LaunchConfiguration("execution_log_max_runs"), value_type=int
                ),
                "execution_log_max_total_bytes": ParameterValue(
                    LaunchConfiguration("execution_log_max_total_bytes"),
                    value_type=int,
                ),
            }],
        ),
        Node(
            package="actor_ik_node",
            executable="hand_tracking_node",
            name="hand_tracking_node",
            output="screen",
        ),
        Node(
            package="replay_node",
            executable="replay_node",
            name="replay_node",
            output="screen",
            parameters=[{
                "sample_dir": LaunchConfiguration("replay_sample_dir"),
                "execution_steps": ParameterValue(
                    LaunchConfiguration("replay_execution_steps"), value_type=int
                ),
                "frequency_hz": ParameterValue(
                    LaunchConfiguration("replay_frequency_hz"), value_type=float
                ),
                "joint_source": LaunchConfiguration("replay_joint_source"),
                "loop": ParameterValue(
                    LaunchConfiguration("replay_loop"), value_type=bool
                ),
                "dry_run": ParameterValue(
                    LaunchConfiguration("replay_dry_run"), value_type=bool
                ),
            }],
        ),
    ]


def build_launch_description(mode: str) -> LaunchDescription:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, received {mode!r}")
    workspace_root = _workspace_root()
    mode_arguments = (
        _teleop_arguments(workspace_root)
        if mode == "teleop"
        else _deploy_arguments(workspace_root, workbench=mode == "workbench")
    )
    mode_nodes = _teleop_nodes() if mode == "teleop" else [
        *_deploy_nodes(workbench=mode == "workbench"),
        *_teleop_nodes(hil=True, workbench=mode == "workbench")[:3],
    ]
    return LaunchDescription([
        *_common_arguments(workspace_root, mode),
        *mode_arguments,
        OpaqueFunction(function=_validate_protection),
        *_environment(),
        *_shared_nodes(mode),
        *mode_nodes,
    ])
