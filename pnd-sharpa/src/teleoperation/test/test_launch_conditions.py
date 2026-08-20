from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import IncludeLaunchDescription


LAUNCH_PATH = Path(__file__).parents[1] / "launch" / "teleoperation.launch.py"


def _load_launch_module():
    spec = spec_from_file_location("teleoperation_launch", LAUNCH_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _manus_enabled(mode: str, teleop_source: str, start_manus: str) -> bool:
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "mode": mode,
            "teleop_source": teleop_source,
            "start_manus": start_manus,
        }
    )
    manus_include = next(
        action
        for action in module.generate_launch_description().entities
        if isinstance(action, IncludeLaunchDescription)
        and "manus_node" in repr(action.launch_description_source.location)
    )
    assert manus_include.condition is not None
    return manus_include.condition.evaluate(context)


def test_quest_teleop_starts_manus_by_default():
    assert _manus_enabled("teleop", "quest", "true")


def test_noitom_teleop_starts_manus_by_default():
    assert _manus_enabled("teleop", "noitom", "true")


def test_manus_can_be_disabled_explicitly():
    assert not _manus_enabled("teleop", "quest", "false")


def test_deploy_does_not_start_manus_pipeline():
    assert not _manus_enabled("deploy", "quest", "true")
