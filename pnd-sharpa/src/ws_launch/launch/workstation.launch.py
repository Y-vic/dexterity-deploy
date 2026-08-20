"""Launch the workstation deploy pipeline."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _int_arg(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def _float_arg(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _bool_arg(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("adam_sharpa_description"))
    default_model_xml = str(
        description_share / "mujoco" / "adam_pro_sharpa_kinematics.xml"
    )
    default_dashboard_model_xml = str(
        description_share / "mujoco" / "adam_pro_sharpa_scene.xml"
    )
    bind_host = LaunchConfiguration("bind_host")
    policy_provider = LaunchConfiguration("policy_provider")
    policy_protocol = LaunchConfiguration("policy_protocol")
    policy_server_url = LaunchConfiguration("policy_server_url")
    policy_ssh_host = LaunchConfiguration("policy_ssh_host")
    policy_ssh_remote_host = LaunchConfiguration("policy_ssh_remote_host")
    policy_prompt = LaunchConfiguration("policy_prompt")
    policy_session_id = LaunchConfiguration("policy_session_id")
    execution_mode = LaunchConfiguration("execution_mode")
    sharpa_control_mode = LaunchConfiguration("sharpa_control_mode")
    model_xml = LaunchConfiguration("model_xml")
    dashboard_model_xml = LaunchConfiguration("dashboard_model_xml")
    pnd_bias_ssh_host = LaunchConfiguration("pnd_bias_ssh_host")
    pnd_bias_remote_path = LaunchConfiguration("pnd_bias_remote_path")
    pnd_bias_local_path = LaunchConfiguration("pnd_bias_local_path")
    dry_run = _bool_arg("dry_run")
    policy_dry_run = _bool_arg("policy_dry_run")
    enable_adam = _bool_arg("enable_adam")
    enable_sharpa = _bool_arg("enable_sharpa")
    allow_zero_wrist_fallback = _bool_arg("allow_zero_wrist_fallback")
    include_waist = _bool_arg("include_waist")
    include_neck = _bool_arg("include_neck")
    start_policy_pipeline = LaunchConfiguration("start_policy_pipeline")

    state_port = _int_arg("state_port")
    tactile_port = _int_arg("tactile_port")
    action_port = _int_arg("action_port")
    action_ttl_ms = _int_arg("action_ttl_ms")
    vision_rtp_port = _int_arg("vision_rtp_port")
    dashboard_port = _int_arg("dashboard_port")
    action_horizon = _int_arg("action_horizon")
    policy_window_frames = _int_arg("policy_window_frames")
    policy_window_stride = _int_arg("policy_window_stride")
    gcc_history_frames = _int_arg("gcc_history_frames")
    ik_max_nfev = _int_arg("ik_max_nfev")
    actor_send_hz = _float_arg("actor_send_hz")
    obs_rate_hz = _float_arg("obs_rate_hz")
    gcc_history_max_gap_s = _float_arg("gcc_history_max_gap_s")
    gcc_joint_max_age_ms = _float_arg("gcc_joint_max_age_ms")
    gcc_wrench_max_age_ms = _float_arg("gcc_wrench_max_age_ms")
    gcc_deformation_max_age_ms = _float_arg("gcc_deformation_max_age_ms")
    baseline_history_max_gap_s = _float_arg("baseline_history_max_gap_s")
    baseline_image_max_age_ms = _float_arg("baseline_image_max_age_ms")
    baseline_wrench_max_age_ms = _float_arg("baseline_wrench_max_age_ms")
    baseline_deformation_max_age_ms = _float_arg(
        "baseline_deformation_max_age_ms"
    )
    v3_history_max_gap_s = _float_arg("v3_history_max_gap_s")
    v3_image_max_age_ms = _float_arg("v3_image_max_age_ms")
    v3_joint_max_age_ms = _float_arg("v3_joint_max_age_ms")
    v3_wrench_max_age_ms = _float_arg("v3_wrench_max_age_ms")
    v3_deformation_max_age_ms = _float_arg("v3_deformation_max_age_ms")
    policy_request_timeout_s = _float_arg("policy_request_timeout_s")
    policy_ssh_remote_port = _int_arg("policy_ssh_remote_port")
    ik_pos_weight = _float_arg("ik_pos_weight")
    ik_rot_weight = _float_arg("ik_rot_weight")
    ik_reg_weight = _float_arg("ik_reg_weight")
    ik_smooth_weight = _float_arg("ik_smooth_weight")
    ik_diff_step = _float_arg("ik_diff_step")

    status_topics = [
        "/ws/status",
        "/ws/robot_states/status",
        "/ws/robot_tactile/status",
        "/ws/robot_vision/status",
        "/ws/obs_sync/status",
        "/ws/obs/debug",
        "/ws/policy_client/status",
        "/ws/action_ik/status",
        "/ws/replay/status",
        "/ws/action_execute/status",
        "/ws/action_execute/plan_debug",
        "/ws/action_execute/safety",
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("bind_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("state_port", default_value="15020"),
            DeclareLaunchArgument("tactile_port", default_value="15021"),
            DeclareLaunchArgument("action_port", default_value="15010"),
            DeclareLaunchArgument("action_ttl_ms", default_value="120"),
            DeclareLaunchArgument("vision_rtp_port", default_value="5601"),
            DeclareLaunchArgument(
                "policy_provider",
                default_value="dreamzero",
                description=(
                    "Expected policy family: dreamzero, cgp, trex, "
                    "vitacformer, groot, gcc, or pace. sharpa_v3 verifies "
                    "this against server metadata."
                ),
            ),
            DeclareLaunchArgument(
                "policy_protocol",
                default_value="sharpa_v3",
                description=(
                    "Policy wire protocol. sharpa_v3 uses metadata/reset/infer; "
                    "set legacy explicitly for old servers."
                ),
            ),
            DeclareLaunchArgument(
                "policy_server_url",
                default_value="ws://127.0.0.1:5500/infer",
            ),
            DeclareLaunchArgument("policy_ssh_host", default_value="BAAI2"),
            DeclareLaunchArgument(
                "policy_ssh_remote_host",
                default_value="127.0.0.1",
            ),
            DeclareLaunchArgument("policy_ssh_remote_port", default_value="5500"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument(
                "policy_dry_run",
                default_value=LaunchConfiguration("dry_run"),
                description=(
                    "Skip the remote policy call and publish placeholder "
                    "predictions. Defaults to dry_run for compatibility. "
                    "For shadow inference use dry_run:=true and "
                    "policy_dry_run:=false."
                ),
            ),
            DeclareLaunchArgument("start_policy_pipeline", default_value="true"),
            DeclareLaunchArgument("enable_adam", default_value="true"),
            DeclareLaunchArgument("enable_sharpa", default_value="true"),
            DeclareLaunchArgument(
                "action_horizon",
                default_value="24",
                description=(
                    "Legacy/dry-run action rows. sharpa_v3 uses the server "
                    "execution action_length and slice."
                ),
            ),
            DeclareLaunchArgument(
                "actor_send_hz",
                default_value="15.0",
                description=(
                    "Expected execution rate: 15 Hz for "
                    "DreamZero/ViTacFormer, 30 Hz for Groot/CGP/T-Rex/GCC/PACE; "
                    "must match sharpa_v3 execution.frequency_hz."
                ),
            ),
            DeclareLaunchArgument(
                "execute_steps",
                default_value="8",
                description=(
                    "Compatibility-only; the server-provided execute slice is used."
                ),
            ),
            DeclareLaunchArgument("execution_mode", default_value="synchronous"),
            DeclareLaunchArgument("policy_window_frames", default_value="4"),
            DeclareLaunchArgument("policy_window_stride", default_value="2"),
            DeclareLaunchArgument(
                "baseline_history_max_gap_s", default_value="0.25"
            ),
            DeclareLaunchArgument(
                "baseline_image_max_age_ms", default_value="150.0"
            ),
            DeclareLaunchArgument(
                "baseline_wrench_max_age_ms", default_value="150.0"
            ),
            DeclareLaunchArgument(
                "baseline_deformation_max_age_ms", default_value="150.0"
            ),
            DeclareLaunchArgument("v3_history_max_gap_s", default_value="0.25"),
            DeclareLaunchArgument("v3_image_max_age_ms", default_value="150.0"),
            DeclareLaunchArgument("v3_joint_max_age_ms", default_value="150.0"),
            DeclareLaunchArgument("v3_wrench_max_age_ms", default_value="150.0"),
            DeclareLaunchArgument(
                "v3_deformation_max_age_ms", default_value="150.0"
            ),
            DeclareLaunchArgument("gcc_history_frames", default_value="9"),
            DeclareLaunchArgument("gcc_history_max_gap_s", default_value="0.25"),
            DeclareLaunchArgument("gcc_joint_max_age_ms", default_value="150.0"),
            DeclareLaunchArgument("gcc_wrench_max_age_ms", default_value="150.0"),
            DeclareLaunchArgument(
                "gcc_deformation_max_age_ms",
                default_value="150.0",
            ),
            DeclareLaunchArgument("obs_rate_hz", default_value="30.0"),
            DeclareLaunchArgument("sharpa_control_mode", default_value="position"),
            DeclareLaunchArgument("policy_request_timeout_s", default_value="90.0"),
            DeclareLaunchArgument(
                "policy_prompt",
                default_value="",
                description=(
                    "Optional legacy prompt. sharpa_v3 uses metadata.prompt and "
                    "rejects a conflicting non-empty value."
                ),
            ),
            DeclareLaunchArgument("policy_session_id", default_value=""),
            DeclareLaunchArgument("allow_zero_wrist_fallback", default_value="false"),
            DeclareLaunchArgument(
                "model_xml",
                default_value=default_model_xml,
            ),
            DeclareLaunchArgument(
                "dashboard_model_xml",
                default_value=default_dashboard_model_xml,
            ),
            DeclareLaunchArgument("include_waist", default_value="false"),
            DeclareLaunchArgument("include_neck", default_value="false"),
            DeclareLaunchArgument("ik_max_nfev", default_value="80"),
            DeclareLaunchArgument("ik_pos_weight", default_value="45.0"),
            DeclareLaunchArgument("ik_rot_weight", default_value="3.5"),
            DeclareLaunchArgument("ik_reg_weight", default_value="0.08"),
            DeclareLaunchArgument("ik_smooth_weight", default_value="0.04"),
            DeclareLaunchArgument("ik_diff_step", default_value="0.0001"),
            DeclareLaunchArgument("dashboard_port", default_value="8088"),
            DeclareLaunchArgument("pnd_bias_ssh_host", default_value="pnd"),
            DeclareLaunchArgument(
                "pnd_bias_remote_path",
                default_value="/home/pnd-humanoid/.adam/joint/bias_joints_set_with_init.json",
            ),
            DeclareLaunchArgument(
                "pnd_bias_local_path",
                default_value=(
                    "/home/ps/Deploy-v2/pnd-sharpa/deploy/runtime/pnd_bias/"
                    "bias_joints_set_with_init.json"
                ),
            ),
            DeclareLaunchArgument("vision_decode", default_value="true"),
            DeclareLaunchArgument("vision_decode_backend", default_value="auto"),
            DeclareLaunchArgument("vision_gst_launch", default_value="gst-launch-1.0"),
            Node(
                package="ws_io",
                executable="robot_states",
                name="robot_states",
                output="screen",
                parameters=[
                    {
                        "host": bind_host,
                        "port": state_port,
                        "state_topic": "/ws/robot_states",
                        "raw_state_topic": "/ws/robot_states/raw",
                        "status_topic": "/ws/robot_states/status",
                        "pnd_bias_ssh_host": pnd_bias_ssh_host,
                        "pnd_bias_remote_path": pnd_bias_remote_path,
                        "pnd_bias_local_path": pnd_bias_local_path,
                    }
                ],
            ),
            Node(
                package="ws_io",
                executable="robot_tactile",
                name="robot_tactile",
                output="screen",
                parameters=[
                    {
                        "host": bind_host,
                        "port": tactile_port,
                        "tactile_topic": "/ws/robot_tactile",
                        "status_topic": "/ws/robot_tactile/status",
                    }
                ],
            ),
            Node(
                package="ws_io",
                executable="robot_vision",
                name="robot_vision",
                output="screen",
                parameters=[
                    {
                        "rtp_port": vision_rtp_port,
                        "enable_decode": _bool_arg("vision_decode"),
                        "decode_backend": LaunchConfiguration("vision_decode_backend"),
                        "gst_launch": LaunchConfiguration("vision_gst_launch"),
                        "output_width": 320,
                        "output_height": 160,
                        "model_image_topic": "/ws/robot_vision",
                        "status_topic": "/ws/robot_vision/status",
                    }
                ],
            ),
            Node(
                package="ws_core",
                executable="obs_sync",
                name="obs_sync",
                output="screen",
                parameters=[
                    {
                        "robot_states_topic": "/ws/robot_states",
                        "robot_tactile_topic": "/ws/robot_tactile",
                        "model_image_topic": "/ws/robot_vision",
                        "obs_topic": "/ws/obs",
                        "status_topic": "/ws/obs_sync/status",
                        "debug_topic": "/ws/obs/debug",
                        "obs_rate_hz": obs_rate_hz,
                        "model_xml": model_xml,
                        "require_fk": True,
                    }
                ],
            ),
            Node(
                package="ws_core",
                executable="policy_client",
                name="policy_client",
                output="screen",
                condition=IfCondition(start_policy_pipeline),
                parameters=[
                    {
                        "provider": policy_provider,
                        "policy_protocol": policy_protocol,
                        "server_url": policy_server_url,
                        "ssh_host": policy_ssh_host,
                        "ssh_remote_host": policy_ssh_remote_host,
                        "ssh_remote_port": policy_ssh_remote_port,
                        "request_timeout_s": policy_request_timeout_s,
                        "prompt": policy_prompt,
                        "session_id": policy_session_id,
                        "allow_zero_wrist_fallback": allow_zero_wrist_fallback,
                        "obs_topic": "/ws/obs",
                        "pred_topic": "/ws/pred",
                        "execution_done_topic": "/ws/execution_done",
                        "status_topic": "/ws/policy_client/status",
                        "dry_run": policy_dry_run,
                        "action_horizon": action_horizon,
                        "dry_run_horizon": action_horizon,
                        "actor_send_hz": actor_send_hz,
                        "obs_rate_hz": obs_rate_hz,
                        "policy_window_frames": policy_window_frames,
                        "policy_window_stride": policy_window_stride,
                        "baseline_history_max_gap_s": (
                            baseline_history_max_gap_s
                        ),
                        "baseline_image_max_age_ms": (
                            baseline_image_max_age_ms
                        ),
                        "baseline_wrench_max_age_ms": (
                            baseline_wrench_max_age_ms
                        ),
                        "baseline_deformation_max_age_ms": (
                            baseline_deformation_max_age_ms
                        ),
                        "v3_history_max_gap_s": v3_history_max_gap_s,
                        "v3_image_max_age_ms": v3_image_max_age_ms,
                        "v3_joint_max_age_ms": v3_joint_max_age_ms,
                        "v3_wrench_max_age_ms": v3_wrench_max_age_ms,
                        "v3_deformation_max_age_ms": (
                            v3_deformation_max_age_ms
                        ),
                        "gcc_history_frames": gcc_history_frames,
                        "gcc_history_max_gap_s": gcc_history_max_gap_s,
                        "gcc_joint_max_age_ms": gcc_joint_max_age_ms,
                        "gcc_wrench_max_age_ms": gcc_wrench_max_age_ms,
                        "gcc_deformation_max_age_ms": (
                            gcc_deformation_max_age_ms
                        ),
                    }
                ],
            ),
            Node(
                package="ws_core",
                executable="action_ik",
                name="action_ik",
                output="screen",
                condition=IfCondition(start_policy_pipeline),
                parameters=[
                    {
                        "pred_topic": "/ws/pred",
                        "action_plan_topic": "/ws/action_plan",
                        "status_topic": "/ws/action_ik/status",
                        "model_xml": model_xml,
                        "include_waist": include_waist,
                        "include_neck": include_neck,
                        "enable_adam": enable_adam,
                        "enable_sharpa": enable_sharpa,
                        "ik_max_nfev": ik_max_nfev,
                        "ik_pos_weight": ik_pos_weight,
                        "ik_rot_weight": ik_rot_weight,
                        "ik_reg_weight": ik_reg_weight,
                        "ik_smooth_weight": ik_smooth_weight,
                        "ik_diff_step": ik_diff_step,
                    }
                ],
            ),
            Node(
                package="ws_core",
                executable="action_execute",
                name="action_execute",
                output="screen",
                parameters=[
                    {
                        "listen_host": bind_host,
                        "listen_port": action_port,
                        "action_ttl_ms": action_ttl_ms,
                        "action_plan_topic": "/ws/action_plan",
                        "execution_done_topic": "/ws/execution_done",
                        "action_topic": "/ws/action",
                        "status_topic": "/ws/action_execute/status",
                        "plan_debug_topic": "/ws/action_execute/plan_debug",
                        "safety_topic": "/ws/action_execute/safety",
                        "dry_run": dry_run,
                        "enable_adam": enable_adam,
                        "enable_sharpa": enable_sharpa,
                        "actor_send_hz": actor_send_hz,
                        "execution_mode": execution_mode,
                        "sharpa_control_mode": sharpa_control_mode,
                    }
                ],
            ),
            Node(
                package="ws_dashboard",
                executable="dashboard",
                name="dashboard",
                output="screen",
                parameters=[
                    {
                        "http_host": bind_host,
                        "http_port": dashboard_port,
                        "status_file": "deploy/runs/ws_dashboard/latest_status.json",
                        "robot_states_topic": "/ws/robot_states/raw",
                        "robot_tactile_topic": "/ws/robot_tactile",
                        "model_image_topic": "/ws/robot_vision",
                        "obs_topic": "/ws/obs",
                        "pred_topic": "/ws/pred",
                        "action_topic": "/ws/action",
                        "status_topics": status_topics,
                        "model_xml": dashboard_model_xml,
                        "task_prompt": policy_prompt,
                        "waist_body": "pelvis",
                    }
                ],
            ),
        ]
    )
