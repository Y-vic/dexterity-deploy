"""Launch the Manus acquisition and retargeting node."""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
    start_manus_client = LaunchConfiguration("start_manus_client")
    start_retarget = LaunchConfiguration("start_retarget")

    sdk_root = _repo_sdk_path("sharpa-manus-sdk")
    retarget_dir = _repo_sdk_path(
        "sharpa-manus-sdk", "retargeting_alg_release_V4.0"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_manus_client",
                default_value="true",
                description="Start the Manus Core client process if available.",
            ),
            DeclareLaunchArgument(
                "start_retarget",
                default_value="true",
                description="Start the Manus-to-Sharpa retarget process.",
            ),
            Node(
                package="manus_node",
                executable="manus",
                name="manus",
                output="screen",
                parameters=[
                    {
                        "sdk_root": sdk_root,
                        "manus_client_path": os.path.join(
                            sdk_root, "client", "SharpaManusClient.out"
                        ),
                        "retarget_script": os.path.join(
                            retarget_dir, "retargeting_manus_demo_multiprocess.py"
                        ),
                        "proto_path": os.path.join(
                            retarget_dir, "include", "proto_hand"
                        ),
                        "retarget_cwd": retarget_dir,
                        "mocap_address": "tcp://127.0.0.1:2044",
                        "hand_action_bind_address": "tcp://*:6668",
                        "hand_action_monitor_address": "tcp://127.0.0.1:6668",
                        "output_topic": "/sharpa_command_joint_states",
                        "status_topic": "/manus/status",
                        "start_manus_client": start_manus_client,
                        "start_retarget": start_retarget,
                    }
                ],
            )
        ]
    )
