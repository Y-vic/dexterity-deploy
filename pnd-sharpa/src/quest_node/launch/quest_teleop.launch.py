"""Launch Quest tracking, retargeting, and the Adam command gate."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


DEFAULT_MODEL = (
    "/opt/pnd/pnd_teleop/install/adam_description/share/adam_description/"
    "urdf/adam_pro/adam_pro.urdf"
)


def generate_launch_description() -> LaunchDescription:
    tf_topic = LaunchConfiguration("tf_topic")
    tf_static_topic = LaunchConfiguration("tf_static_topic")
    joy_topic = LaunchConfiguration("joy_topic")
    tracking_status_topic = LaunchConfiguration("tracking_status_topic")
    retarget_raw_topic = LaunchConfiguration("retarget_raw_topic")

    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            DeclareLaunchArgument("start_webvr", default_value="true"),
            DeclareLaunchArgument("start_retarget", default_value="true"),
            DeclareLaunchArgument("start_gate", default_value="true"),
            DeclareLaunchArgument("http_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("http_port", default_value="8443"),
            DeclareLaunchArgument("websocket_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("websocket_port", default_value="8442"),
            DeclareLaunchArgument(
                "access_token",
                default_value=EnvironmentVariable(
                    "PND_QUEST_ACCESS_TOKEN",
                    default_value="",
                ),
            ),
            DeclareLaunchArgument("authentication_timeout", default_value="3.0"),
            DeclareLaunchArgument(
                "public_web_url",
                default_value="https://10.10.20.127/webvr/",
            ),
            DeclareLaunchArgument("web_root", default_value=""),
            DeclareLaunchArgument("video_layout", default_value="mono"),
            DeclareLaunchArgument("video_swap_eyes", default_value="false"),
            DeclareLaunchArgument("video_enabled", default_value="false"),
            DeclareLaunchArgument("turn_enabled", default_value="false"),
            DeclareLaunchArgument("turn_host", default_value="10.10.20.127"),
            DeclareLaunchArgument("turn_port", default_value="3478"),
            DeclareLaunchArgument("turn_username", default_value="quest-video"),
            DeclareLaunchArgument("turn_password", default_value=""),
            DeclareLaunchArgument("turn_relay_min_port", default_value="49160"),
            DeclareLaunchArgument("turn_relay_max_port", default_value="49200"),
            DeclareLaunchArgument("tf_topic", default_value="/_quest/tf"),
            DeclareLaunchArgument(
                "tf_static_topic",
                default_value="/_quest/tf_static",
            ),
            DeclareLaunchArgument("joy_topic", default_value="/_quest/joy"),
            DeclareLaunchArgument(
                "tracking_status_topic",
                default_value="/quest/webvr_status",
            ),
            DeclareLaunchArgument(
                "command_status_topic",
                default_value="/quest/command_status",
            ),
            DeclareLaunchArgument(
                "calibration_service",
                default_value="/quest/calibrate",
            ),
            DeclareLaunchArgument(
                "retarget_raw_topic",
                default_value="/_quest/retargeted_joint_states_raw",
            ),
            DeclareLaunchArgument(
                "output_topic",
                default_value="/adam_command_joint_states",
            ),
            DeclareLaunchArgument(
                "bias_joint_states_topic",
                default_value="/adam_bias_command_joint_states",
            ),
            DeclareLaunchArgument("input_timeout", default_value="0.2"),
            DeclareLaunchArgument("calibration_stale_timeout", default_value="1.0"),
            DeclareLaunchArgument("tracking_recovery_frames", default_value="3"),
            DeclareLaunchArgument(
                "tracking_recovery_motion_threshold", default_value="0.03"
            ),
            DeclareLaunchArgument("tracking_timeout", default_value="0.2"),
            DeclareLaunchArgument("bias_state_timeout", default_value="0.5"),
            DeclareLaunchArgument("fix_neck_waist", default_value="false"),
            DeclareLaunchArgument("enable_neck", default_value="true"),
            DeclareLaunchArgument(
                "retarget_method",
                default_value="nonlinear_ik",
                description=(
                    "Arm IK method: local_qp, shoulder_prior, nonlinear_ik, "
                    "or elbow_pole."
                ),
            ),
            DeclareLaunchArgument("webvr_poll_rate", default_value="100.0"),
            DeclareLaunchArgument("webvr_status_rate", default_value="2.0"),
            DeclareLaunchArgument("robot_arm_length", default_value="0.53"),
            DeclareLaunchArgument("retarget_rate", default_value="50.0"),
            DeclareLaunchArgument("position_scale", default_value="1.0"),
            DeclareLaunchArgument("ik_iterations", default_value="5"),
            DeclareLaunchArgument("ik_solve_dt", default_value="0.05"),
            DeclareLaunchArgument("ik_solver", default_value="daqp"),
            DeclareLaunchArgument("ik_damping", default_value="0.0"),
            DeclareLaunchArgument("ik_lm_damping", default_value="1.0"),
            DeclareLaunchArgument("ik_line_search_steps", default_value="10"),
            DeclareLaunchArgument("wrist_position_cost", default_value="50.0"),
            DeclareLaunchArgument("wrist_orientation_cost", default_value="1.0"),
            DeclareLaunchArgument("elbow_position_cost", default_value="10.0"),
            DeclareLaunchArgument("smoothness_cost", default_value="0.2"),
            DeclareLaunchArgument("posture_cost", default_value="0.05"),
            DeclareLaunchArgument(
                "shoulder_prior_wrist_position_cost", default_value="20.0"
            ),
            DeclareLaunchArgument(
                "shoulder_prior_wrist_orientation_cost", default_value="18.0"
            ),
            DeclareLaunchArgument(
                "shoulder_prior_orientation_cost", default_value="2.0"
            ),
            DeclareLaunchArgument("nonlinear_translation_cost", default_value="50.0"),
            DeclareLaunchArgument("nonlinear_rotation_cost", default_value="1.0"),
            DeclareLaunchArgument("nonlinear_posture_cost", default_value="0.02"),
            DeclareLaunchArgument("nonlinear_smoothness_cost", default_value="0.1"),
            DeclareLaunchArgument("nonlinear_filter_enabled", default_value="true"),
            DeclareLaunchArgument(
                "model_path",
                default_value=EnvironmentVariable(
                    "PND_QUEST_MODEL_PATH",
                    default_value=DEFAULT_MODEL,
                ),
            ),
            DeclareLaunchArgument(
                "python",
                default_value=EnvironmentVariable(
                    "PND_QUEST_PYTHON",
                    default_value="/opt/pnd/pnd_teleop/.venv/bin/python3",
                ),
            ),
            Node(
                package="quest_node",
                executable="quest_webvr",
                name="_quest_webvr",
                output="screen",
                emulate_tty=True,
                condition=IfCondition(LaunchConfiguration("start_webvr")),
                parameters=[
                    {
                        "http_host": LaunchConfiguration("http_host"),
                        "http_port": ParameterValue(
                            LaunchConfiguration("http_port"), value_type=int
                        ),
                        "websocket_host": LaunchConfiguration("websocket_host"),
                        "websocket_port": ParameterValue(
                            LaunchConfiguration("websocket_port"), value_type=int
                        ),
                        "access_token": LaunchConfiguration("access_token"),
                        "authentication_timeout": ParameterValue(
                            LaunchConfiguration("authentication_timeout"),
                            value_type=float,
                        ),
                        "public_web_url": LaunchConfiguration("public_web_url"),
                        "web_root": LaunchConfiguration("web_root"),
                        "video_layout": LaunchConfiguration("video_layout"),
                        "video_swap_eyes": ParameterValue(
                            LaunchConfiguration("video_swap_eyes"), value_type=bool
                        ),
                        "video_enabled": ParameterValue(
                            LaunchConfiguration("video_enabled"), value_type=bool
                        ),
                        "turn_enabled": ParameterValue(
                            LaunchConfiguration("turn_enabled"), value_type=bool
                        ),
                        "turn_host": LaunchConfiguration("turn_host"),
                        "turn_port": ParameterValue(
                            LaunchConfiguration("turn_port"), value_type=int
                        ),
                        "turn_username": LaunchConfiguration("turn_username"),
                        "turn_password": LaunchConfiguration("turn_password"),
                        "turn_relay_min_port": ParameterValue(
                            LaunchConfiguration("turn_relay_min_port"), value_type=int
                        ),
                        "turn_relay_max_port": ParameterValue(
                            LaunchConfiguration("turn_relay_max_port"), value_type=int
                        ),
                        "joy_topic": joy_topic,
                        "status_topic": tracking_status_topic,
                        "calibration_service": LaunchConfiguration(
                            "calibration_service"
                        ),
                        "poll_rate": ParameterValue(
                            LaunchConfiguration("webvr_poll_rate"), value_type=float
                        ),
                        "status_rate": ParameterValue(
                            LaunchConfiguration("webvr_status_rate"), value_type=float
                        ),
                        "input_timeout": ParameterValue(
                            LaunchConfiguration("input_timeout"), value_type=float
                        ),
                        "calibration_stale_timeout": ParameterValue(
                            LaunchConfiguration("calibration_stale_timeout"),
                            value_type=float,
                        ),
                        "tracking_recovery_frames": ParameterValue(
                            LaunchConfiguration("tracking_recovery_frames"),
                            value_type=int,
                        ),
                        "tracking_recovery_motion_threshold": ParameterValue(
                            LaunchConfiguration(
                                "tracking_recovery_motion_threshold"
                            ),
                            value_type=float,
                        ),
                        "robot_arm_length": ParameterValue(
                            LaunchConfiguration("robot_arm_length"), value_type=float
                        ),
                        "calibration_mode": "zero_pose",
                        "retarget_warm_start_service": "",
                    }
                ],
                remappings=[("/tf", tf_topic)],
            ),
            Node(
                package="quest_node",
                executable="quest_retarget",
                name="_quest_retarget",
                output="screen",
                emulate_tty=True,
                condition=IfCondition(LaunchConfiguration("start_retarget")),
                prefix=LaunchConfiguration("python"),
                additional_env={"PYTHONNOUSERSITE": "1"},
                parameters=[
                    {
                        "base_frame": "world",
                        "model_path": LaunchConfiguration("model_path"),
                        "output_topic": retarget_raw_topic,
                        "bias_joint_states_topic": LaunchConfiguration(
                            "bias_joint_states_topic"
                        ),
                        "tracking_status_topic": tracking_status_topic,
                        "joy_topic": joy_topic,
                        "tracking_timeout": ParameterValue(
                            LaunchConfiguration("tracking_timeout"), value_type=float
                        ),
                        "bias_state_timeout": ParameterValue(
                            LaunchConfiguration("bias_state_timeout"), value_type=float
                        ),
                        "enable_neck": ParameterValue(
                            LaunchConfiguration("enable_neck"), value_type=bool
                        ),
                        "retarget_method": LaunchConfiguration("retarget_method"),
                        "control_loop_rate": ParameterValue(
                            LaunchConfiguration("retarget_rate"), value_type=float
                        ),
                        "position_scale": ParameterValue(
                            LaunchConfiguration("position_scale"), value_type=float
                        ),
                        "iterations": ParameterValue(
                            LaunchConfiguration("ik_iterations"), value_type=int
                        ),
                        "solve_dt": ParameterValue(
                            LaunchConfiguration("ik_solve_dt"), value_type=float
                        ),
                        "solver": LaunchConfiguration("ik_solver"),
                        "damping": ParameterValue(
                            LaunchConfiguration("ik_damping"), value_type=float
                        ),
                        "lm_damping": ParameterValue(
                            LaunchConfiguration("ik_lm_damping"), value_type=float
                        ),
                        "line_search_steps": ParameterValue(
                            LaunchConfiguration("ik_line_search_steps"), value_type=int
                        ),
                        "wrist_position_cost": ParameterValue(
                            LaunchConfiguration("wrist_position_cost"), value_type=float
                        ),
                        "wrist_orientation_cost": ParameterValue(
                            LaunchConfiguration("wrist_orientation_cost"),
                            value_type=float,
                        ),
                        "elbow_position_cost": ParameterValue(
                            LaunchConfiguration("elbow_position_cost"),
                            value_type=float,
                        ),
                        "smoothness_cost": ParameterValue(
                            LaunchConfiguration("smoothness_cost"), value_type=float
                        ),
                        "posture_cost": ParameterValue(
                            LaunchConfiguration("posture_cost"), value_type=float
                        ),
                        "shoulder_prior_wrist_position_cost": ParameterValue(
                            LaunchConfiguration(
                                "shoulder_prior_wrist_position_cost"
                            ),
                            value_type=float,
                        ),
                        "shoulder_prior_wrist_orientation_cost": ParameterValue(
                            LaunchConfiguration(
                                "shoulder_prior_wrist_orientation_cost"
                            ),
                            value_type=float,
                        ),
                        "shoulder_prior_orientation_cost": ParameterValue(
                            LaunchConfiguration("shoulder_prior_orientation_cost"),
                            value_type=float,
                        ),
                        "nonlinear_translation_cost": ParameterValue(
                            LaunchConfiguration("nonlinear_translation_cost"),
                            value_type=float,
                        ),
                        "nonlinear_rotation_cost": ParameterValue(
                            LaunchConfiguration("nonlinear_rotation_cost"),
                            value_type=float,
                        ),
                        "nonlinear_posture_cost": ParameterValue(
                            LaunchConfiguration("nonlinear_posture_cost"),
                            value_type=float,
                        ),
                        "nonlinear_smoothness_cost": ParameterValue(
                            LaunchConfiguration("nonlinear_smoothness_cost"),
                            value_type=float,
                        ),
                        "nonlinear_filter_enabled": ParameterValue(
                            LaunchConfiguration("nonlinear_filter_enabled"),
                            value_type=bool,
                        ),
                    }
                ],
                remappings=[("/tf", tf_topic), ("/tf_static", tf_static_topic)],
            ),
            Node(
                package="quest_node",
                executable="quest_command",
                name="quest",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_gate")),
                parameters=[
                    {
                        "input_topic": retarget_raw_topic,
                        "tracking_status_topic": tracking_status_topic,
                        "output_topic": LaunchConfiguration("output_topic"),
                        "bias_joint_states_topic": LaunchConfiguration(
                            "bias_joint_states_topic"
                        ),
                        "command_status_topic": LaunchConfiguration(
                            "command_status_topic"
                        ),
                        "tracking_timeout": ParameterValue(
                            LaunchConfiguration("tracking_timeout"), value_type=float
                        ),
                        "bias_state_timeout": ParameterValue(
                            LaunchConfiguration("bias_state_timeout"), value_type=float
                        ),
                        "fix_neck_waist": ParameterValue(
                            LaunchConfiguration("fix_neck_waist"), value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
