from pathlib import Path

from launch import LaunchContext
from ur_common.paths import workspace_root
from ur_launch.launch_config import _common_arguments, _deploy_arguments


def test_checkout_discovery_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv('DEPLOY_ROOT', raising=False)
    monkeypatch.delenv('UR_SHARPA_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)
    assert workspace_root() == Path(__file__).resolve().parents[4]


def test_launch_paths_stay_in_the_selected_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'runtime/config'))
    context = LaunchContext()
    for argument in [*_common_arguments(tmp_path), *_deploy_arguments(tmp_path)]:
        argument.execute(context)
    for key, relative in {
        'interface_path': 'interface',
        'recording_root': 'data/episodes',
        'reference_dataset_root': 'data/episodes',
        'demo_recording_root': 'data/demo_recordings',
        'execution_log_root': 'runtime/log/execution',
        'bias_file': 'runtime/config/teleop_bias.json',
    }.items():
        assert context.launch_configurations[key] == str(tmp_path / relative)
