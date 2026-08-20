"""Launch the teleoperation stack with Adam as the only body publisher."""

from __future__ import annotations

import os

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


DEFAULT_BIAS_PATH = "/home/pnd-humanoid/.adam/joint/bias_joints_set_with_init.json"


def teleop_component_enabled(mode, enabled) -> PythonExpression:
    return PythonExpression(
        [
            "'",
            mode,
            "'.lower() == 'teleop' and '",
            enabled,
            "'.lower() == 'true'",
        ]
    )


def default_urdf_path() -> str:
    try:
        share = get_package_share_directory("adam_sharpa_description")
    except PackageNotFoundError:
        return ""
    return os.path.join(share, "urdf", "adam_pro_sharpa", "adam_pro_sharpa.urdf")


def generate_launch_description() -> LaunchDescription:
    noitom_launch = PathJoinSubstitution(
        [FindPackageShare("noitom_node"), "launch", "noitom_node.launch.py"]
    )
    manus_launch = PathJoinSubstitution(
        [FindPackageShare("manus_node"), "launch", "manus_node.launch.py"]
    )
    sharpa_launch = PathJoinSubstitution(
        [FindPackageShare("sharpa_node"), "launch", "sharpa_node.launch.py"]
    )
    zed_launch = PathJoinSubstitution(
        [FindPackageShare("zed_node"), "launch", "zed.launch.py"]
    )
    quest_launch = PathJoinSubstitution(
        [FindPackageShare("quest_node"), "launch", "quest_teleop.launch.py"]
    )
    monitor_launch = PathJoinSubstitution(
        [FindPackageShare("monitor_node"), "launch", "monitor_node.launch.py"]
    )
    foxglove_launch = PathJoinSubstitution(
        [FindPackageShare("foxglove_node"), "launch", "foxglove_node.launch.py"]
    )
    mode = LaunchConfiguration("mode")
    teleop_source = LaunchConfiguration("teleop_source")
    start_manus = LaunchConfiguration("start_manus")
    start_actor_node = LaunchConfiguration("start_actor_node")
    start_obs_node = LaunchConfiguration("start_obs_node")
    sharpa_command_mode = LaunchConfiguration("sharpa_command_mode")
    sharpa_startup_zero_hold_s = LaunchConfiguration("sharpa_startup_zero_hold_s")
    zed_inference_stream_enabled = LaunchConfiguration("zed_inference_stream_enabled")

    noitom_enabled = PythonExpression(
        [
            "'",
            mode,
            "'.lower() == 'teleop' and '",
            teleop_source,
            "'.lower() == 'noitom'",
        ]
    )
    quest_enabled = PythonExpression(
        [
            "'",
            mode,
            "'.lower() == 'teleop' and '",
            teleop_source,
            "'.lower() == 'quest'",
        ]
    )
    manus_enabled = teleop_component_enabled(mode, start_manus)
    actor_node_enabled = PythonExpression(
        [
            "'true' if ('",
            mode,
            "'.lower() == 'deploy' or '",
            start_actor_node,
            "'.lower() == 'true' and '",
            teleop_source,
            "'.lower() != 'quest') else 'false'",
        ]
    )
    obs_node_enabled = PythonExpression(
        [
            "'true' if ('",
            mode,
            "'.lower() == 'deploy' or '",
            start_obs_node,
            "'.lower() == 'true') else 'false'",
        ]
    )
    sharpa_command_mode_effective = PythonExpression(
        ["'auto' if '", mode, "'.lower() == 'deploy' else '", sharpa_command_mode, "'"]
    )
    zed_inference_stream_enabled_effective = PythonExpression(
        [
            "'true' if '",
            mode,
            "'.lower() == 'deploy' else '",
            zed_inference_stream_enabled,
            "'",
        ]
    )
    zed_quest_stream_enabled_effective = PythonExpression(
        [
            "'true' if ('",
            mode,
            "'.lower() == 'teleop' and '",
            teleop_source,
            "'.lower() == 'quest') else '",
            LaunchConfiguration("zed_quest_stream_enabled"),
            "'",
        ]
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            DeclareLaunchArgument(
                "mode",
                default_value="teleop",
                description="High-level mode: teleop starts the selected input source; deploy starts actor/obs.",
            ),
            DeclareLaunchArgument(
                "teleop_source",
                default_value="noitom",
                description="Upper-body teleoperation source: noitom or quest.",
                choices=["noitom", "quest"],
            ),
            DeclareLaunchArgument("start_bias", default_value="true"),
            DeclareLaunchArgument("start_status", default_value="true"),
            DeclareLaunchArgument("start_adam", default_value="true"),
            DeclareLaunchArgument(
                "start_manus",
                default_value="true",
                description="Start Manus hand tracking in either teleop source mode.",
            ),
            DeclareLaunchArgument("start_sharpa", default_value="true"),
            DeclareLaunchArgument("start_zed", default_value="true"),
            DeclareLaunchArgument("start_monitor", default_value="true"),
            DeclareLaunchArgument("start_actor_node", default_value="false"),
            DeclareLaunchArgument("start_obs_node", default_value="false"),
            DeclareLaunchArgument("start_foxglove", default_value="true"),
            DeclareLaunchArgument("inference_host", default_value="10.10.20.110"),
            DeclareLaunchArgument("deploy_action_port", default_value="15010"),
            DeclareLaunchArgument("deploy_state_port", default_value="15020"),
            DeclareLaunchArgument("deploy_tactile_bulk_port", default_value="15021"),
            DeclareLaunchArgument("deploy_state_rate_hz", default_value="60.0"),
            DeclareLaunchArgument("connect_sharpa", default_value="true"),
            DeclareLaunchArgument("publish_tactile", default_value="true"),
            DeclareLaunchArgument("tactile_poll_warmup_s", default_value="5.0"),
            DeclareLaunchArgument("tactile_fresh_timeout_s", default_value="0.25"),
            DeclareLaunchArgument("tactile_sensor_time_max_age_s", default_value="1.0"),
            DeclareLaunchArgument("tactile_error_log_period_s", default_value="1.0"),
            DeclareLaunchArgument(
                "command_snapshot_topic", default_value="/sharpa_command_snapshot"
            ),
            DeclareLaunchArgument("publish_command_snapshot", default_value="true"),
            DeclareLaunchArgument("command_snapshot_max_hz", default_value="30.0"),
            DeclareLaunchArgument("sharpa_command_mode", default_value="position"),
            DeclareLaunchArgument("sharpa_mit_torque_limit", default_value="2.0"),
            DeclareLaunchArgument("sharpa_startup_zero_hold_s", default_value="2.0"),
            DeclareLaunchArgument("bias_bind_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("bias_bind_port", default_value="18080"),
            DeclareLaunchArgument("bias_path", default_value=DEFAULT_BIAS_PATH),
            DeclareLaunchArgument(
                "bias_startup_sequence_mode",
                default_value="bias_init_then_bias",
                choices=["bias_init_then_bias", "bias"],
            ),
            DeclareLaunchArgument("urdf_path", default_value=default_urdf_path()),
            DeclareLaunchArgument("recording_root", default_value="/mnt/t9/recordings"),
            DeclareLaunchArgument("require_recording_root_mount", default_value="true"),
            DeclareLaunchArgument("recording_sample_rate_hz", default_value="30.0"),
            DeclareLaunchArgument(
                "require_tactile_fresh_on_start", default_value="true"
            ),
            DeclareLaunchArgument(
                "block_recording_on_tactile_error", default_value="false"
            ),
            DeclareLaunchArgument("tactile_start_max_age_ms", default_value="500.0"),
            DeclareLaunchArgument("record_zed_video", default_value="true"),
            DeclareLaunchArgument("zed_monitor_stream_enabled", default_value="true"),
            DeclareLaunchArgument(
                "zed_monitor_stream_host", default_value="10.10.20.127"
            ),
            DeclareLaunchArgument("zed_monitor_stream_port", default_value="5600"),
            DeclareLaunchArgument(
                "zed_inference_stream_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "zed_inference_stream_host", default_value="10.10.20.110"
            ),
            DeclareLaunchArgument("zed_inference_stream_port", default_value="5601"),
            DeclareLaunchArgument("zed_quest_stream_enabled", default_value="false"),
            DeclareLaunchArgument(
                "zed_quest_stream_bind_host", default_value="0.0.0.0"
            ),
            DeclareLaunchArgument("zed_quest_stream_port", default_value="5602"),
            DeclareLaunchArgument("zed_video_ffmpeg", default_value="ffmpeg"),
            DeclareLaunchArgument("zed_video_stop_timeout_s", default_value="5.0"),
            DeclareLaunchArgument("zed_video_bitrate", default_value="8000000"),
            DeclareLaunchArgument("zed_video_fps", default_value="30"),
            DeclareLaunchArgument("zed_video_width", default_value="1280"),
            DeclareLaunchArgument("zed_video_height", default_value="720"),
            DeclareLaunchArgument("zed_video_layout", default_value="mono"),
            DeclareLaunchArgument("zed_watchdog_enabled", default_value="true"),
            DeclareLaunchArgument("zed_watchdog_failure_threshold", default_value="5"),
            DeclareLaunchArgument(
                "zed_watchdog_restart_cooldown_s", default_value="20.0"
            ),
            DeclareLaunchArgument("zed_browser_ui_enabled", default_value="true"),
            DeclareLaunchArgument("zed_browser_ui_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("zed_browser_ui_port", default_value="12100"),
            DeclareLaunchArgument("zed_browser_ui_web_root", default_value=""),
            DeclareLaunchArgument("dpad_y_up_sign", default_value="-1"),
            DeclareLaunchArgument("xbox_device", default_value="/dev/input/js0"),
            DeclareLaunchArgument("rt_axis", default_value="4"),
            DeclareLaunchArgument("rt_button", default_value="9"),
            DeclareLaunchArgument("rt_axis_threshold", default_value="0.5"),
            DeclareLaunchArgument("rt_pressed_when", default_value="above"),
            DeclareLaunchArgument("adam_dry_run", default_value="false"),
            DeclareLaunchArgument(
                "adam_require_control_subscriber", default_value="true"
            ),
            DeclareLaunchArgument("noitom_fix_neck_waist", default_value="true"),
            DeclareLaunchArgument("noitom_retarget_backend", default_value="pinocchio"),
            DeclareLaunchArgument(
                "noitom_mink_config_path",
                default_value=EnvironmentVariable(
                    "PND_MINK_CONFIG_PATH",
                    default_value=(
                        "/opt/pnd/pnd_teleop/install/adam_mink/share/"
                        "adam_mink/config/adam_pro_noitom_mink_cfg.yaml"
                    ),
                ),
            ),
            DeclareLaunchArgument(
                "noitom_mink_model_path",
                default_value=EnvironmentVariable(
                    "PND_MINK_MODEL_PATH",
                    default_value=(
                        "/opt/pnd/pnd_teleop/install/adam_description/share/"
                        "adam_description/urdf/adam_pro/adam_pro.xml"
                    ),
                ),
            ),
            DeclareLaunchArgument("noitom_mink_mujoco_sim", default_value="false"),
            DeclareLaunchArgument("noitom_mink_ik_iter_max", default_value="1"),
            DeclareLaunchArgument("noitom_mink_ik_damping", default_value="0.1"),
            DeclareLaunchArgument("noitom_mink_ik_solver", default_value="daqp"),
            DeclareLaunchArgument(
                "noitom_mink_python_venv_bin",
                default_value=EnvironmentVariable(
                    "PND_MINK_VENV_BIN",
                    default_value="/opt/pnd/pnd_teleop/.venv/bin",
                ),
            ),
            DeclareLaunchArgument(
                "noitom_gmr_repo_path",
                default_value=EnvironmentVariable(
                    "PND_GMR_REPO",
                    default_value="/home/pnd-humanoid/Deploy/GMR-master",
                ),
            ),
            DeclareLaunchArgument(
                "noitom_gmr_python",
                default_value=EnvironmentVariable(
                    "NOITOM_GMR_PYTHON",
                    default_value="/home/pnd-humanoid/Deploy/.venv-gmr/bin/python",
                ),
            ),
            DeclareLaunchArgument("noitom_gmr_solver", default_value="daqp"),
            DeclareLaunchArgument("noitom_gmr_damping", default_value="0.3"),
            DeclareLaunchArgument("noitom_gmr_lock_root", default_value="false"),
            DeclareLaunchArgument("noitom_gmr_reset_each_frame", default_value="false"),
            DeclareLaunchArgument("noitom_gmr_posture_cost", default_value="0.05"),
            DeclareLaunchArgument(
                "noitom_gmr_apply_pnd_coordinate_transform",
                default_value="false",
            ),
            DeclareLaunchArgument("quest_http_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("quest_http_port", default_value="8443"),
            DeclareLaunchArgument(
                "quest_websocket_host",
                default_value="127.0.0.1",
            ),
            DeclareLaunchArgument("quest_websocket_port", default_value="8442"),
            DeclareLaunchArgument(
                "quest_access_token",
                default_value=EnvironmentVariable(
                    "PND_QUEST_ACCESS_TOKEN",
                    default_value="",
                ),
            ),
            DeclareLaunchArgument(
                "quest_authentication_timeout",
                default_value="3.0",
            ),
            DeclareLaunchArgument(
                "quest_public_web_url",
                default_value="https://10.10.20.127/webvr/",
            ),
            DeclareLaunchArgument("quest_web_root", default_value=""),
            DeclareLaunchArgument("quest_video_swap_eyes", default_value="false"),
            DeclareLaunchArgument("quest_video_enabled", default_value="false"),
            DeclareLaunchArgument("quest_turn_enabled", default_value="false"),
            DeclareLaunchArgument("quest_input_timeout", default_value="0.2"),
            DeclareLaunchArgument(
                "quest_calibration_stale_timeout",
                default_value="1.0",
            ),
            DeclareLaunchArgument("quest_tracking_timeout", default_value="0.2"),
            DeclareLaunchArgument("quest_bias_state_timeout", default_value="0.5"),
            DeclareLaunchArgument("quest_fix_neck_waist", default_value="false"),
            DeclareLaunchArgument("quest_enable_neck", default_value="true"),
            DeclareLaunchArgument(
                "quest_retarget_method",
                default_value="nonlinear_ik",
                description=(
                    "Quest arm IK method: local_qp, shoulder_prior, "
                    "nonlinear_ik, or elbow_pole."
                ),
            ),
            DeclareLaunchArgument("quest_retarget_rate", default_value="50.0"),
            DeclareLaunchArgument("quest_position_scale", default_value="1.0"),
            DeclareLaunchArgument("quest_ik_iterations", default_value="5"),
            DeclareLaunchArgument("quest_ik_solve_dt", default_value="0.05"),
            DeclareLaunchArgument("quest_ik_solver", default_value="daqp"),
            DeclareLaunchArgument("quest_ik_damping", default_value="0.0"),
            DeclareLaunchArgument("quest_ik_lm_damping", default_value="1.0"),
            DeclareLaunchArgument("quest_ik_line_search_steps", default_value="10"),
            DeclareLaunchArgument("quest_wrist_position_cost", default_value="50.0"),
            DeclareLaunchArgument("quest_wrist_orientation_cost", default_value="1.0"),
            DeclareLaunchArgument("quest_elbow_position_cost", default_value="10.0"),
            DeclareLaunchArgument("quest_smoothness_cost", default_value="0.2"),
            DeclareLaunchArgument("quest_posture_cost", default_value="0.05"),
            DeclareLaunchArgument(
                "quest_shoulder_prior_wrist_position_cost", default_value="20.0"
            ),
            DeclareLaunchArgument(
                "quest_shoulder_prior_wrist_orientation_cost", default_value="18.0"
            ),
            DeclareLaunchArgument(
                "quest_shoulder_prior_orientation_cost", default_value="2.0"
            ),
            DeclareLaunchArgument(
                "quest_nonlinear_translation_cost", default_value="50.0"
            ),
            DeclareLaunchArgument(
                "quest_nonlinear_rotation_cost", default_value="1.0"
            ),
            DeclareLaunchArgument(
                "quest_nonlinear_posture_cost", default_value="0.02"
            ),
            DeclareLaunchArgument(
                "quest_nonlinear_smoothness_cost", default_value="0.1"
            ),
            DeclareLaunchArgument(
                "quest_nonlinear_filter_enabled", default_value="true"
            ),
            DeclareLaunchArgument(
                "quest_model_path",
                default_value=EnvironmentVariable(
                    "PND_QUEST_MODEL_PATH",
                    default_value=(
                        "/opt/pnd/pnd_teleop/install/adam_description/share/"
                        "adam_description/urdf/adam_pro/adam_pro.urdf"
                    ),
                ),
            ),
            DeclareLaunchArgument(
                "quest_python",
                default_value=EnvironmentVariable(
                    "PND_QUEST_PYTHON",
                    default_value="/opt/pnd/pnd_teleop/.venv/bin/python3",
                ),
            ),
            Node(
                package="bias_node",
                executable="bias",
                name="bias",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_bias")),
                parameters=[
                    {
                        "bind_host": LaunchConfiguration("bias_bind_host"),
                        "bind_port": ParameterValue(
                            LaunchConfiguration("bias_bind_port"), value_type=int
                        ),
                        "bias_path": LaunchConfiguration("bias_path"),
                        "urdf_path": LaunchConfiguration("urdf_path"),
                        "output_topic": "/adam_bias_command_joint_states",
                        "robot_state_topic": "/adam_physical_joint_states",
                        "command_state_topic": "/adam_bias_command_joint_states",
                        "control_status_topic": "/control_status",
                        "status_topic": "/bias/status",
                        "public_url_path": "/bias_joints",
                        "publish_rate_hz": 30.0,
                        "startup_publish_delay_s": 0.25,
                        "startup_on_t_init": True,
                        "startup_sequence_mode": LaunchConfiguration(
                            "bias_startup_sequence_mode"
                        ),
                        "arrival_tolerance_deg": 8.0,
                        "robot_state_timeout_s": 0.5,
                        "sequence_step_timeout_s": 3.0,
                        "interpolation_threshold_deg": 10.0,
                        "interpolation_duration_s": 1.0,
                    }
                ],
            ),
            Node(
                package="status_node",
                executable="status",
                name="status",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_status")),
                parameters=[
                    {
                        "status_topic": "/control_status",
                        "status_json_topic": "/teleop/status_json",
                        "joy_topics": ["/xbox/joy"],
                        "device": LaunchConfiguration("xbox_device"),
                        "device_backend": "auto",
                        "read_device": True,
                        "publish_joy": True,
                        "joy_output_topic": "/joy",
                        "axis_count": 8,
                        "button_count": 15,
                        "poll_period": 0.005,
                        "dpad_x_axis": 6,
                        "dpad_y_axis": 7,
                        "dpad_y_up_sign": ParameterValue(
                            LaunchConfiguration("dpad_y_up_sign"), value_type=int
                        ),
                        "lb_button": 6,
                        "b_button": 1,
                        "lt_axis": 5,
                        "lt_button": 8,
                        "lt_axis_threshold": 0.5,
                        "lt_pressed_when": "above",
                        "rt_axis": ParameterValue(
                            LaunchConfiguration("rt_axis"), value_type=int
                        ),
                        "rt_button": ParameterValue(
                            LaunchConfiguration("rt_button"), value_type=int
                        ),
                        "rt_axis_threshold": ParameterValue(
                            LaunchConfiguration("rt_axis_threshold"), value_type=float
                        ),
                        "rt_pressed_when": LaunchConfiguration("rt_pressed_when"),
                    }
                ],
            ),
            Node(
                package="adam_node",
                executable="adam",
                name="adam",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_adam")),
                parameters=[
                    {
                        "lowstate_topic": "/lowstate",
                        "bias_joint_states_topic": "/adam_bias_command_joint_states",
                        "command_joint_states_topic": "/adam_command_joint_states",
                        "physical_joint_states_topic": "/adam_physical_joint_states",
                        "robot_states_topic": "/robot_states",
                        "control_joint_states_topic": "/joint_states",
                        "control_status_topic": "/control_status",
                        "status_topic": "/adam/status",
                        "publish_rate": 100.0,
                        "dry_run": ParameterValue(
                            LaunchConfiguration("adam_dry_run"), value_type=bool
                        ),
                        "require_control_subscriber": ParameterValue(
                            LaunchConfiguration("adam_require_control_subscriber"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(noitom_launch),
                condition=IfCondition(noitom_enabled),
                launch_arguments={
                    "retarget_backend": LaunchConfiguration("noitom_retarget_backend"),
                    "retarget_output_topic": "/adam_command_joint_states",
                    "bias_joint_states_topic": "/adam_bias_command_joint_states",
                    "fix_neck_waist": LaunchConfiguration("noitom_fix_neck_waist"),
                    "mink_config_path": LaunchConfiguration("noitom_mink_config_path"),
                    "mink_model_path": LaunchConfiguration("noitom_mink_model_path"),
                    "mink_mujoco_sim": LaunchConfiguration("noitom_mink_mujoco_sim"),
                    "mink_ik_iter_max": LaunchConfiguration("noitom_mink_ik_iter_max"),
                    "mink_ik_damping": LaunchConfiguration("noitom_mink_ik_damping"),
                    "mink_ik_solver": LaunchConfiguration("noitom_mink_ik_solver"),
                    "mink_python_venv_bin": LaunchConfiguration(
                        "noitom_mink_python_venv_bin"
                    ),
                    "gmr_repo_path": LaunchConfiguration("noitom_gmr_repo_path"),
                    "gmr_python": LaunchConfiguration("noitom_gmr_python"),
                    "gmr_solver": LaunchConfiguration("noitom_gmr_solver"),
                    "gmr_damping": LaunchConfiguration("noitom_gmr_damping"),
                    "gmr_lock_root": LaunchConfiguration("noitom_gmr_lock_root"),
                    "gmr_reset_each_frame": LaunchConfiguration(
                        "noitom_gmr_reset_each_frame"
                    ),
                    "gmr_posture_cost": LaunchConfiguration("noitom_gmr_posture_cost"),
                    "gmr_apply_pnd_coordinate_transform": LaunchConfiguration(
                        "noitom_gmr_apply_pnd_coordinate_transform"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(manus_launch),
                condition=IfCondition(manus_enabled),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sharpa_launch),
                condition=IfCondition(LaunchConfiguration("start_sharpa")),
                launch_arguments={
                    "connect_sharpa": LaunchConfiguration("connect_sharpa"),
                    "status_json_topic": "/teleop/status_json",
                    "publish_tactile": LaunchConfiguration("publish_tactile"),
                    "tactile_poll_warmup_s": LaunchConfiguration(
                        "tactile_poll_warmup_s"
                    ),
                    "tactile_fresh_timeout_s": LaunchConfiguration(
                        "tactile_fresh_timeout_s"
                    ),
                    "tactile_sensor_time_max_age_s": LaunchConfiguration(
                        "tactile_sensor_time_max_age_s"
                    ),
                    "tactile_error_log_period_s": LaunchConfiguration(
                        "tactile_error_log_period_s"
                    ),
                    "command_snapshot_topic": LaunchConfiguration(
                        "command_snapshot_topic"
                    ),
                    "publish_command_snapshot": LaunchConfiguration(
                        "publish_command_snapshot"
                    ),
                    "command_snapshot_max_hz": LaunchConfiguration(
                        "command_snapshot_max_hz"
                    ),
                    "command_mode": sharpa_command_mode_effective,
                    "mit_torque_limit": LaunchConfiguration("sharpa_mit_torque_limit"),
                    "startup_zero_hold_s": sharpa_startup_zero_hold_s,
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(zed_launch),
                condition=IfCondition(LaunchConfiguration("start_zed")),
                launch_arguments={
                    "video_bitrate": LaunchConfiguration("zed_video_bitrate"),
                    "video_fps": LaunchConfiguration("zed_video_fps"),
                    "video_width": LaunchConfiguration("zed_video_width"),
                    "video_height": LaunchConfiguration("zed_video_height"),
                    "video_layout": LaunchConfiguration("zed_video_layout"),
                    "monitor_stream_enabled": LaunchConfiguration(
                        "zed_monitor_stream_enabled"
                    ),
                    "monitor_stream_host": LaunchConfiguration(
                        "zed_monitor_stream_host"
                    ),
                    "monitor_stream_port": LaunchConfiguration(
                        "zed_monitor_stream_port"
                    ),
                    "inference_stream_enabled": zed_inference_stream_enabled_effective,
                    "inference_stream_host": LaunchConfiguration(
                        "zed_inference_stream_host"
                    ),
                    "inference_stream_port": LaunchConfiguration(
                        "zed_inference_stream_port"
                    ),
                    "quest_stream_enabled": zed_quest_stream_enabled_effective,
                    "quest_stream_bind_host": LaunchConfiguration(
                        "zed_quest_stream_bind_host"
                    ),
                    "quest_stream_port": LaunchConfiguration("zed_quest_stream_port"),
                    "watchdog_enabled": LaunchConfiguration("zed_watchdog_enabled"),
                    "watchdog_failure_threshold": LaunchConfiguration(
                        "zed_watchdog_failure_threshold"
                    ),
                    "watchdog_restart_cooldown_s": LaunchConfiguration(
                        "zed_watchdog_restart_cooldown_s"
                    ),
                    "browser_ui_enabled": LaunchConfiguration("zed_browser_ui_enabled"),
                    "browser_ui_host": LaunchConfiguration("zed_browser_ui_host"),
                    "browser_ui_port": LaunchConfiguration("zed_browser_ui_port"),
                    "browser_ui_web_root": LaunchConfiguration(
                        "zed_browser_ui_web_root"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(quest_launch),
                condition=IfCondition(quest_enabled),
                launch_arguments={
                    "http_host": LaunchConfiguration("quest_http_host"),
                    "http_port": LaunchConfiguration("quest_http_port"),
                    "websocket_host": LaunchConfiguration("quest_websocket_host"),
                    "websocket_port": LaunchConfiguration("quest_websocket_port"),
                    "access_token": LaunchConfiguration("quest_access_token"),
                    "authentication_timeout": LaunchConfiguration(
                        "quest_authentication_timeout"
                    ),
                    "public_web_url": LaunchConfiguration("quest_public_web_url"),
                    "web_root": LaunchConfiguration("quest_web_root"),
                    "video_layout": LaunchConfiguration("zed_video_layout"),
                    "video_swap_eyes": LaunchConfiguration("quest_video_swap_eyes"),
                    "video_enabled": LaunchConfiguration("quest_video_enabled"),
                    "turn_enabled": LaunchConfiguration("quest_turn_enabled"),
                    "input_timeout": LaunchConfiguration("quest_input_timeout"),
                    "calibration_stale_timeout": LaunchConfiguration(
                        "quest_calibration_stale_timeout"
                    ),
                    "tracking_timeout": LaunchConfiguration("quest_tracking_timeout"),
                    "bias_state_timeout": LaunchConfiguration(
                        "quest_bias_state_timeout"
                    ),
                    "fix_neck_waist": LaunchConfiguration("quest_fix_neck_waist"),
                    "enable_neck": LaunchConfiguration("quest_enable_neck"),
                    "retarget_method": LaunchConfiguration(
                        "quest_retarget_method"
                    ),
                    "retarget_rate": LaunchConfiguration("quest_retarget_rate"),
                    "position_scale": LaunchConfiguration("quest_position_scale"),
                    "ik_iterations": LaunchConfiguration("quest_ik_iterations"),
                    "ik_solve_dt": LaunchConfiguration("quest_ik_solve_dt"),
                    "ik_solver": LaunchConfiguration("quest_ik_solver"),
                    "ik_damping": LaunchConfiguration("quest_ik_damping"),
                    "ik_lm_damping": LaunchConfiguration("quest_ik_lm_damping"),
                    "ik_line_search_steps": LaunchConfiguration(
                        "quest_ik_line_search_steps"
                    ),
                    "wrist_position_cost": LaunchConfiguration(
                        "quest_wrist_position_cost"
                    ),
                    "wrist_orientation_cost": LaunchConfiguration(
                        "quest_wrist_orientation_cost"
                    ),
                    "elbow_position_cost": LaunchConfiguration(
                        "quest_elbow_position_cost"
                    ),
                    "smoothness_cost": LaunchConfiguration("quest_smoothness_cost"),
                    "posture_cost": LaunchConfiguration("quest_posture_cost"),
                    "shoulder_prior_wrist_position_cost": LaunchConfiguration(
                        "quest_shoulder_prior_wrist_position_cost"
                    ),
                    "shoulder_prior_wrist_orientation_cost": LaunchConfiguration(
                        "quest_shoulder_prior_wrist_orientation_cost"
                    ),
                    "shoulder_prior_orientation_cost": LaunchConfiguration(
                        "quest_shoulder_prior_orientation_cost"
                    ),
                    "nonlinear_translation_cost": LaunchConfiguration(
                        "quest_nonlinear_translation_cost"
                    ),
                    "nonlinear_rotation_cost": LaunchConfiguration(
                        "quest_nonlinear_rotation_cost"
                    ),
                    "nonlinear_posture_cost": LaunchConfiguration(
                        "quest_nonlinear_posture_cost"
                    ),
                    "nonlinear_smoothness_cost": LaunchConfiguration(
                        "quest_nonlinear_smoothness_cost"
                    ),
                    "nonlinear_filter_enabled": LaunchConfiguration(
                        "quest_nonlinear_filter_enabled"
                    ),
                    "model_path": LaunchConfiguration("quest_model_path"),
                    "python": LaunchConfiguration("quest_python"),
                    "output_topic": "/adam_command_joint_states",
                    "bias_joint_states_topic": "/adam_bias_command_joint_states",
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(monitor_launch),
                condition=IfCondition(LaunchConfiguration("start_monitor")),
                launch_arguments={
                    "recording_root": LaunchConfiguration("recording_root"),
                    "require_recording_root_mount": LaunchConfiguration(
                        "require_recording_root_mount"
                    ),
                    "sample_rate_hz": LaunchConfiguration("recording_sample_rate_hz"),
                    "require_tactile_fresh_on_start": LaunchConfiguration(
                        "require_tactile_fresh_on_start"
                    ),
                    "block_recording_on_tactile_error": LaunchConfiguration(
                        "block_recording_on_tactile_error"
                    ),
                    "tactile_start_max_age_ms": LaunchConfiguration(
                        "tactile_start_max_age_ms"
                    ),
                    "tactile_error_log_period_s": LaunchConfiguration(
                        "tactile_error_log_period_s"
                    ),
                    "record_zed_video": LaunchConfiguration("record_zed_video"),
                    "zed_video_rtp_port": LaunchConfiguration(
                        "zed_monitor_stream_port"
                    ),
                    "zed_video_ffmpeg": LaunchConfiguration("zed_video_ffmpeg"),
                    "zed_video_stop_timeout_s": LaunchConfiguration(
                        "zed_video_stop_timeout_s"
                    ),
                }.items(),
            ),
            Node(
                package="actor_node",
                executable="actor_node",
                name="actor_node",
                output="screen",
                condition=IfCondition(actor_node_enabled),
                parameters=[
                    {
                        "server_host": LaunchConfiguration("inference_host"),
                        "action_port": ParameterValue(
                            LaunchConfiguration("deploy_action_port"),
                            value_type=int,
                        ),
                        "adam_command_topic": "/adam_command_joint_states",
                        "sharpa_command_topic": "/sharpa_command_joint_states",
                        "status_topic": "/actor_node/status",
                    }
                ],
            ),
            Node(
                package="obs_node",
                executable="obs_node",
                name="obs_node",
                output="screen",
                condition=IfCondition(obs_node_enabled),
                parameters=[
                    {
                        "server_host": LaunchConfiguration("inference_host"),
                        "state_port": ParameterValue(
                            LaunchConfiguration("deploy_state_port"),
                            value_type=int,
                        ),
                        "tactile_bulk_port": ParameterValue(
                            LaunchConfiguration("deploy_tactile_bulk_port"),
                            value_type=int,
                        ),
                        "state_rate_hz": ParameterValue(
                            LaunchConfiguration("deploy_state_rate_hz"),
                            value_type=float,
                        ),
                        "adam_state_topic": "/adam_physical_joint_states",
                        "sharpa_state_topic": "/sharpa_physical_joint_states",
                        "tactile_topic_prefix": "/sharpa_physical_tactile",
                        "zed_status_topic": "/zed/status",
                        "status_topic": "/obs_node/status",
                    }
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(foxglove_launch),
                condition=IfCondition(LaunchConfiguration("start_foxglove")),
                launch_arguments={
                    "adam_joint_states_topic": "/adam_command_joint_states",
                    "sharpa_joint_states_topic": "/sharpa_command_joint_states",
                    "visualization_joint_states_topic": "/foxglove/joint_states",
                    "urdf_path": LaunchConfiguration("urdf_path"),
                }.items(),
            ),
        ]
    )
