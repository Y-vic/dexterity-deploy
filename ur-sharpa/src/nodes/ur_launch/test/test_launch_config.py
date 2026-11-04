from pathlib import Path
import xml.etree.ElementTree as ET

from ur_launch.launch_config import (
    _default_model_config,
    _deploy_nodes,
    _environment,
    _ur_motion_parameters,
)


def test_remote_model_manifest_is_default_when_present(tmp_path):
    assert _default_model_config(tmp_path) == tmp_path / "config/model_servers.json"
    local = tmp_path / "config/model_servers.local.json"
    local.parent.mkdir()
    local.write_text('{}')
    assert _default_model_config(tmp_path) == local


def test_tactile_acquisition_recording_and_ae01_feedback_defaults_and_overrides(tmp_path):
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters
    from ur_launch.launch_config import _common_arguments, _teleop_arguments, _shared_nodes, _teleop_nodes
    context = LaunchContext()
    for argument in [*_common_arguments(tmp_path), *_teleop_arguments(tmp_path)]:
        argument.execute(context)
    def parameters(executable):
        nodes = [*_shared_nodes('teleop'), *_teleop_nodes()]
        node = next(node for node in nodes if getattr(node, 'node_executable', None) == executable)
        return evaluate_parameters(context, node._Node__parameters)[0]
    assert parameters('sharpa_node')['skip_tactile'] is False
    assert parameters('recording_node')['skip_tactile'] is False
    assert parameters('ae01_node')['feedback_enabled'] is True
    context.launch_configurations.update(sharpa_skip_tactile='true', ae01_feedback_enabled='false')
    assert parameters('sharpa_node')['skip_tactile'] is True
    assert parameters('recording_node')['skip_tactile'] is True
    assert parameters('ae01_node')['feedback_enabled'] is False


def test_teleop_enables_final_ur_motion_guards():
    parameters = _ur_motion_parameters("teleop")

    assert parameters["enable_motion_limits"] is True
    assert parameters["enable_joint_step_limit"] is True
    assert parameters["enable_joint_clip"] is True
    assert "max_abs_joint_rad" in parameters
    assert "max_joint_step_rad" in parameters


def test_deploy_graph_runs_remote_policy_client():
    nodes = _deploy_nodes()
    executables = [
        (node.node_package, node.node_executable) for node in nodes
    ]

    assert executables == [
        ("policy_client_node", "policy_client_node"),
        ("dashboard_node", "dashboard_node"),
        ("recording_node", "demo_node"),
        ("actor_ik_node", "actor_ik_node"),
        ("actor_ik_node", "hand_tracking_node"),
        ("replay_node", "replay_node"),
    ]
    assert "robot_states_node" not in [node.node_package for node in nodes]


def test_fastdds_ack_and_small_feedback_profiles_are_exact_topic_writers_only():
    workspace = Path(__file__).resolve().parents[4]
    root = ET.parse(workspace / "config" / "fastdds_ack_profiles.xml").getroot()
    namespace = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
    publishers = root.findall("f:profiles/f:publisher", namespace)

    assert {publisher.attrib["profile_name"] for publisher in publishers} == {
        "/ur/command_result",
        "/sharpa/command_result",
        "/ur/state", "/sharpa/state", "/sharpa/tau", "/sharpa/wrench",
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


def test_small_math_thread_defaults_preserve_explicit_operator_settings(monkeypatch):
    from ur_launch.launch_config import _small_math_environment
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.delenv(key, raising=False)
    assert set(_small_math_environment().values()) == {'1'}
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '2')
    assert _small_math_environment()['OPENBLAS_NUM_THREADS'] == '2'


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


def test_ppp_reset_only_applies_to_ppp_deployment(tmp_path):
    import json
    from ur_launch.launch_config import _default_bias_file
    config=tmp_path/'config';config.mkdir()
    (config/'ppp_reset.json').write_text('{}')
    manifest=config/'model_servers.local.json'
    manifest.write_text(json.dumps({'models':[{'task':'pick_place_push'}]}))
    assert _default_bias_file(tmp_path,'deploy') == '~/.config/ur_sharpa/teleop_bias.json'
    assert _default_bias_file(tmp_path,'teleop') == '~/.config/ur_sharpa/teleop_bias.json'
    manifest.write_text(json.dumps({'models':[{'task':'twist_tube'}]}))
    assert _default_bias_file(tmp_path,'deploy') == '~/.config/ur_sharpa/teleop_bias.json'


def test_hil_inputs_are_opt_in_and_share_existing_teleop_arguments(tmp_path):
    from launch import LaunchContext
    from ur_launch.launch_config import _deploy_arguments, _teleop_nodes
    context = LaunchContext()
    candidates = _teleop_nodes(hil=True)[:3]
    assert [node.node_package for node in candidates] == ['vive_node','ae01_node','recording_node']
    context.launch_configurations['enable_hil'] = 'false'
    assert all(not node.condition.evaluate(context) for node in candidates[:2])
    context.launch_configurations['enable_hil'] = 'true'
    assert all(node.condition.evaluate(context) for node in candidates[:2])
    assert candidates[2].condition is None  # Policy may opt into saving without HIL devices.
    names = [argument.name for argument in _deploy_arguments(tmp_path)]
    assert len(names)==len(set(names)) and 'enable_hil' in names
    assert 'recording_root' in names


def test_deploy_recording_root_defaults_to_test_and_remains_overridable(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    from launch import LaunchContext
    from ur_launch.launch_config import _deploy_arguments, _teleop_nodes
    argument = next(arg for arg in _deploy_arguments(tmp_path) if arg.name == 'recording_root')
    context = LaunchContext()
    argument.execute(context)
    assert context.launch_configurations['recording_root'] == '/home/yns/data/test'
    recorder = _teleop_nodes(hil=True)[2]
    params = recorder._Node__parameters[0]
    root = next(value for key, value in params.items() if key[0].text == 'recording_root')
    assert root[0].perform(context) == '/home/yns/data/test'
    context.launch_configurations['recording_root'] = '/tmp/custom-hil'
    assert root[0].perform(context) == '/tmp/custom-hil'


def test_all_launch_modes_restore_recording_directory_but_explicit_argument_wins(tmp_path, monkeypatch):
    from launch import LaunchContext
    from ur_common.recording_config import RecordingPreferences
    from ur_launch.launch_config import _teleop_arguments, _deploy_arguments
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'settings'))
    selected = str(tmp_path / 'remembered')
    RecordingPreferences().save(selected)
    for arguments in (_teleop_arguments(tmp_path), _deploy_arguments(tmp_path),
                      _deploy_arguments(tmp_path, workbench=True)):
        argument = next(arg for arg in arguments if arg.name == 'recording_root')
        context = LaunchContext()
        argument.execute(context)
        assert context.launch_configurations['recording_root'] == selected
        context = LaunchContext()
        context.launch_configurations['recording_root'] = '/explicit'
        argument.execute(context)
        assert context.launch_configurations['recording_root'] == '/explicit'
    assert RecordingPreferences().read('/fallback') == selected


def test_workbench_graph_has_one_instance_of_each_hardware_input_and_recorder(tmp_path):
    from launch import LaunchContext
    from ur_launch.launch_config import _deploy_arguments, _teleop_nodes, _shared_nodes
    context = LaunchContext()
    arguments = _deploy_arguments(tmp_path, workbench=True)
    assert len({arg.name for arg in arguments}) == len(arguments)
    next(arg for arg in arguments if arg.name == 'enable_hil').execute(context)
    assert context.launch_configurations['enable_hil'] == 'false'
    nodes = [*_deploy_nodes(workbench=True), *_teleop_nodes(hil=True, workbench=True)[:3]]
    executables = [node.node_executable for node in nodes]
    for name in ('dashboard_node', 'vive_node', 'ae01_node', 'recording_node', 'actor_ik_node'):
        assert executables.count(name) == 1
    assert 'monitor_node' not in executables
    assert all(n.condition is None for n in _teleop_nodes(hil=True, workbench=True)[:2])
    status = next(node for node in _shared_nodes('workbench') if getattr(node, 'node_executable', None) == 'status_node')
    params = {key[0].text: value for key,value in status._Node__parameters[0].items()}
    assert params['allow_mode_switch'] is True
    from launch_ros.utilities import evaluate_parameters
    assert evaluate_parameters(context, status._Node__parameters)[0]['mode'] == 'teleop'


def test_selected_protection_file_drives_launch_defaults_and_explicit_override(tmp_path, monkeypatch):
    import os
    import yaml
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters, normalize_parameters
    from ur_common.protection_config import load_protection_config, ENVIRONMENT_VARIABLE
    from ur_launch.launch_config import _common_arguments, _deploy_arguments, _validate_protection
    data = load_protection_config()
    data['ur']['max_joint_step_rad'] = .31
    data['hil']['return_duration_s'] = 1.5
    data['ae01']['data_timeout_s'] = .2
    path = tmp_path/'protection.yaml'
    path.write_text(yaml.safe_dump(data))
    # LaunchContext defaults to os.environ; isolate SetEnvironmentVariable actions.
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    context = LaunchContext()
    context.launch_configurations['protection_config'] = str(path)
    _validate_protection(context)
    for argument in [*_common_arguments(tmp_path), *_deploy_arguments(tmp_path)]:
        argument.execute(context)
    params = evaluate_parameters(context, normalize_parameters([_ur_motion_parameters('deploy')]))[0]
    assert params['max_joint_step_rad'] == .31
    assert context.launch_configurations['hil_return_duration_s'] == '1.5'
    assert context.launch_configurations['ae01_data_timeout_s'] == '0.2'
    for action in _environment():
        action.execute(context)
    assert context.environment[ENVIRONMENT_VARIABLE] == str(path)
    context.launch_configurations['ur_max_joint_step_rad'] = '.29'
    assert evaluate_parameters(context, normalize_parameters([_ur_motion_parameters('deploy')]))[0]['max_joint_step_rad'] == .29


def test_authoritative_status_receives_hil_capability_in_every_launch_mode():
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters
    from ur_launch.launch_config import _shared_nodes
    for mode in ('teleop', 'deploy', 'workbench'):
        for enabled in ('false', 'true'):
            context = LaunchContext()
            context.launch_configurations['enable_hil'] = enabled
            node = next(node for node in _shared_nodes(mode)
                        if getattr(node, 'node_executable', None) == 'status_node')
            parameters = evaluate_parameters(context, node._Node__parameters)[0]
            assert parameters['enable_hil'] is (mode != 'teleop' and enabled == 'true')
