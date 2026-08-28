from ur_launch.launch_config import _deploy_nodes, _ur_motion_parameters


def test_teleop_enables_final_ur_motion_guards():
    parameters = _ur_motion_parameters("teleop")

    assert parameters["enable_motion_limits"] is True
    assert parameters["enable_joint_step_limit"] is True
    assert parameters["enable_joint_clip"] is True
    assert "max_abs_joint_rad" in parameters
    assert "max_joint_step_rad" in parameters


def test_deploy_graph_builds_observations_inside_policy_client():
    nodes = _deploy_nodes()
    executables = [
        (node.node_package, node.node_executable) for node in nodes
    ]

    assert executables == [
        ("dashboard_node", "dashboard_node"),
        ("policy_client_node", "policy_client_node"),
        ("actor_ik_node", "actor_ik_node"),
        ("actor_ik_node", "hand_tracking_node"),
        ("replay_node", "replay_node"),
    ]
    assert "robot_states_node" not in [node.node_package for node in nodes]
