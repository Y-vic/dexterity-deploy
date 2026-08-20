"""Launch Noitom mocap and selectable Adam retarget backends."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("noitom_node")
    retarget_config_dir = PathJoinSubstitution(
        [
            package_share,
            "vendor",
            "share",
            "adam_retarget",
            "opti_config",
            "noitom",
            "adam_sp_pro",
        ]
    )
    retarget_config = PathJoinSubstitution(
        [retarget_config_dir, "Adam_SPPro_Noitom_Deploy_opti.json"]
    )
    noitom_lib_dir = PathJoinSubstitution(
        [package_share, "vendor", "lib", "noitom_mocap"]
    )
    mink_config = EnvironmentVariable(
        "PND_MINK_CONFIG_PATH",
        default_value=(
            "/opt/pnd/pnd_teleop/install/adam_mink/share/adam_mink/config/"
            "adam_pro_noitom_mink_cfg.yaml"
        ),
    )
    mink_model = EnvironmentVariable(
        "PND_MINK_MODEL_PATH",
        default_value=(
            "/opt/pnd/pnd_teleop/install/adam_description/share/"
            "adam_description/urdf/adam_pro/adam_pro.xml"
        ),
    )

    tf_topic = LaunchConfiguration("tf_topic")
    tf_static_topic = LaunchConfiguration("tf_static_topic")
    start_pinocchio_retarget = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("start_retarget"),
                "'.lower() == 'true' and '",
                LaunchConfiguration("retarget_backend"),
                "'.lower() in ('pinocchio', 'vendor')",
            ]
        )
    )
    start_mink_retarget = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("start_retarget"),
                "'.lower() == 'true' and '",
                LaunchConfiguration("retarget_backend"),
                "'.lower() == 'mink'",
            ]
        )
    )
    start_gmr_retarget = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("start_retarget"),
                "'.lower() == 'true' and '",
                LaunchConfiguration("retarget_backend"),
                "'.lower() == 'gmr'",
            ]
        )
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable("PYTHONNOUSERSITE", "1"),
            SetEnvironmentVariable(
                "LD_LIBRARY_PATH",
                [
                    noitom_lib_dir,
                    ":",
                    retarget_config_dir,
                    ":",
                    EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
                ],
            ),
            DeclareLaunchArgument("start_mocap", default_value="true"),
            DeclareLaunchArgument("start_retarget", default_value="true"),
            DeclareLaunchArgument("start_gate", default_value="true"),
            DeclareLaunchArgument("start_static_transform", default_value="true"),
            DeclareLaunchArgument("retarget_backend", default_value="pinocchio"),
            DeclareLaunchArgument("tf_topic", default_value="/_noitom/tf"),
            DeclareLaunchArgument("tf_static_topic", default_value="/_noitom/tf_static"),
            DeclareLaunchArgument(
                "retarget_raw_topic",
                default_value="/_noitom/retargeted_joint_states_raw",
            ),
            DeclareLaunchArgument(
                "retarget_output_topic",
                default_value="/adam_command_joint_states",
            ),
            DeclareLaunchArgument(
                "bias_joint_states_topic",
                default_value="/adam_bias_command_joint_states",
            ),
            DeclareLaunchArgument("fix_neck_waist", default_value="true"),
            DeclareLaunchArgument("bias_state_timeout", default_value="0.5"),
            DeclareLaunchArgument("base_frame", default_value="world_zup"),
            DeclareLaunchArgument("control_loop_rate", default_value="100.0"),
            DeclareLaunchArgument("mink_config_path", default_value=mink_config),
            DeclareLaunchArgument("mink_model_path", default_value=mink_model),
            DeclareLaunchArgument("mink_mujoco_sim", default_value="false"),
            DeclareLaunchArgument("mink_ik_iter_max", default_value="1"),
            DeclareLaunchArgument("mink_ik_damping", default_value="0.1"),
            DeclareLaunchArgument("mink_ik_solver", default_value="daqp"),
            DeclareLaunchArgument(
                "mink_python_venv_bin",
                default_value=EnvironmentVariable(
                    "PND_MINK_VENV_BIN",
                    default_value="/opt/pnd/pnd_teleop/.venv/bin",
                ),
            ),
            DeclareLaunchArgument(
                "gmr_repo_path",
                default_value=EnvironmentVariable(
                    "PND_GMR_REPO",
                    default_value="/home/pnd-humanoid/Deploy/GMR-master",
                ),
            ),
            DeclareLaunchArgument(
                "gmr_python",
                default_value=EnvironmentVariable(
                    "NOITOM_GMR_PYTHON",
                    default_value="/home/pnd-humanoid/Deploy/.venv-gmr/bin/python",
                ),
            ),
            DeclareLaunchArgument("gmr_solver", default_value="daqp"),
            DeclareLaunchArgument("gmr_damping", default_value="0.3"),
            DeclareLaunchArgument("gmr_use_velocity_limit", default_value="false"),
            DeclareLaunchArgument("gmr_upper_body_only", default_value="true"),
            DeclareLaunchArgument("gmr_lock_root", default_value="false"),
            DeclareLaunchArgument("gmr_reset_each_frame", default_value="false"),
            DeclareLaunchArgument("gmr_posture_cost", default_value="0.05"),
            DeclareLaunchArgument(
                "gmr_apply_pnd_coordinate_transform",
                default_value="false",
            ),
            DeclareLaunchArgument("gmr_offset_to_ground", default_value="false"),
            DeclareLaunchArgument("gmr_status_period", default_value="2.0"),
            SetEnvironmentVariable(
                "NOITOM_GMR_PYTHON",
                LaunchConfiguration("gmr_python"),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="_noitom_world_static_transform",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_static_transform")),
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0.707106781",
                    "0",
                    "0",
                    "0.707106781",
                    "world_zup",
                    "world",
                ],
                remappings=[("/tf", tf_topic), ("/tf_static", tf_static_topic)],
            ),
            Node(
                package="noitom_node",
                executable="noitom_mocap",
                name="_noitom_mocap",
                output="screen",
                emulate_tty=True,
                condition=IfCondition(LaunchConfiguration("start_mocap")),
                remappings=[("/tf", tf_topic)],
            ),
            Node(
                package="noitom_node",
                executable="noitom_retarget",
                name="_noitom_retarget",
                output="screen",
                emulate_tty=True,
                condition=start_pinocchio_retarget,
                cwd=retarget_config_dir,
                parameters=[
                    {
                        "base_frame": LaunchConfiguration("base_frame"),
                        "control_loop_rate": ParameterValue(
                            LaunchConfiguration("control_loop_rate"),
                            value_type=float,
                        ),
                        "config_json_path": retarget_config,
                        "custom_weight": {},
                    }
                ],
                remappings=[
                    ("/joint_states", LaunchConfiguration("retarget_raw_topic")),
                    ("/tf", tf_topic),
                    ("/tf_static", tf_static_topic),
                ],
            ),
            Node(
                package="noitom_node",
                executable="noitom_mink_retarget",
                name="_noitom_retarget",
                output="screen",
                emulate_tty=True,
                condition=start_mink_retarget,
                parameters=[
                    {
                        "adam_mink_cfg": LaunchConfiguration("mink_config_path"),
                        "adam_model_path": LaunchConfiguration("mink_model_path"),
                        "mujoco_sim": ParameterValue(
                            LaunchConfiguration("mink_mujoco_sim"),
                            value_type=bool,
                        ),
                        "ik_iter_max": ParameterValue(
                            LaunchConfiguration("mink_ik_iter_max"),
                            value_type=int,
                        ),
                        "ik_damping": ParameterValue(
                            LaunchConfiguration("mink_ik_damping"),
                            value_type=float,
                        ),
                        "ik_solver": LaunchConfiguration("mink_ik_solver"),
                    }
                ],
                additional_env={
                    "PATH": [
                        LaunchConfiguration("mink_python_venv_bin"),
                        ":",
                        EnvironmentVariable("PATH", default_value=""),
                    ],
                    "PYTHONNOUSERSITE": "1",
                },
                remappings=[
                    ("/joint_states", LaunchConfiguration("retarget_raw_topic")),
                    ("/tf", tf_topic),
                    ("/tf_static", tf_static_topic),
                ],
            ),
            Node(
                package="noitom_node",
                executable="noitom_gmr_retarget",
                name="_noitom_retarget",
                output="screen",
                emulate_tty=True,
                condition=start_gmr_retarget,
                parameters=[
                    {
                        "base_frame": LaunchConfiguration("base_frame"),
                        "control_loop_rate": ParameterValue(
                            LaunchConfiguration("control_loop_rate"),
                            value_type=float,
                        ),
                        "gmr_repo_path": LaunchConfiguration("gmr_repo_path"),
                        "solver": LaunchConfiguration("gmr_solver"),
                        "damping": ParameterValue(
                            LaunchConfiguration("gmr_damping"),
                            value_type=float,
                        ),
                        "use_velocity_limit": ParameterValue(
                            LaunchConfiguration("gmr_use_velocity_limit"),
                            value_type=bool,
                        ),
                        "upper_body_only": ParameterValue(
                            LaunchConfiguration("gmr_upper_body_only"),
                            value_type=bool,
                        ),
                        "lock_root": ParameterValue(
                            LaunchConfiguration("gmr_lock_root"),
                            value_type=bool,
                        ),
                        "reset_each_frame": ParameterValue(
                            LaunchConfiguration("gmr_reset_each_frame"),
                            value_type=bool,
                        ),
                        "posture_cost": ParameterValue(
                            LaunchConfiguration("gmr_posture_cost"),
                            value_type=float,
                        ),
                        "apply_pnd_coordinate_transform": ParameterValue(
                            LaunchConfiguration(
                                "gmr_apply_pnd_coordinate_transform"
                            ),
                            value_type=bool,
                        ),
                        "offset_to_ground": ParameterValue(
                            LaunchConfiguration("gmr_offset_to_ground"),
                            value_type=bool,
                        ),
                        "status_period": ParameterValue(
                            LaunchConfiguration("gmr_status_period"),
                            value_type=float,
                        ),
                    }
                ],
                remappings=[
                    ("/joint_states", LaunchConfiguration("retarget_raw_topic")),
                    ("/tf", tf_topic),
                    ("/tf_static", tf_static_topic),
                ],
            ),
            Node(
                package="noitom_node",
                executable="noitom",
                name="noitom",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_gate")),
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("retarget_raw_topic"),
                        "output_topic": LaunchConfiguration("retarget_output_topic"),
                        "bias_joint_states_topic": LaunchConfiguration(
                            "bias_joint_states_topic"
                        ),
                        "bias_state_timeout": ParameterValue(
                            LaunchConfiguration("bias_state_timeout"),
                            value_type=float,
                        ),
                        "fix_neck_waist": ParameterValue(
                            LaunchConfiguration("fix_neck_waist"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
