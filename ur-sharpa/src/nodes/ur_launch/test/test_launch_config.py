from pathlib import Path
import xml.etree.ElementTree as ET

from ur_launch.launch_config import (
    _deploy_nodes,
    _environment,
    _ur_motion_parameters,
)


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
        ("recording_node", "demo_node"),
        ("policy_client_node", "policy_client_node"),
        ("actor_ik_node", "actor_ik_node"),
        ("actor_ik_node", "hand_tracking_node"),
        ("replay_node", "replay_node"),
    ]
    assert "robot_states_node" not in [node.node_package for node in nodes]


def test_fastdds_ack_profiles_are_exact_topic_writers_only():
    workspace = Path(__file__).resolve().parents[4]
    root = ET.parse(workspace / "config" / "fastdds_ack_profiles.xml").getroot()
    namespace = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
    publishers = root.findall("f:profiles/f:publisher", namespace)

    assert {publisher.attrib["profile_name"] for publisher in publishers} == {
        "/ur/command_result",
        "/sharpa/command_result",
    }
    assert not any(
        publisher.attrib.get("is_default_profile") == "true"
        for publisher in publishers
    )
    for publisher in publishers:
        period = publisher.find("f:times/f:heartbeatPeriod", namespace)
        assert period is not None
        assert period.findtext("f:sec", namespaces=namespace) == "0"
        assert period.findtext("f:nanosec", namespaces=namespace) == "50000000"


def test_launch_loads_ack_profiles_without_enabling_global_xml_qos():
    actions = _environment()
    names = {
        "".join(part.text for part in action._SetEnvironmentVariable__name): action
        for action in actions
    }

    assert "FASTDDS_DEFAULT_PROFILES_FILE" in names
    assert "RMW_FASTRTPS_USE_QOS_FROM_XML" in names
    assert "UR_SHARPA_FASTDDS_HEARTBEAT_MS" in names
    assert names["RMW_FASTRTPS_USE_QOS_FROM_XML"]._SetEnvironmentVariable__value[
        0
    ].text == "0"
    assert names["UR_SHARPA_FASTDDS_HEARTBEAT_MS"]._SetEnvironmentVariable__value[
        0
    ].text == "50"
