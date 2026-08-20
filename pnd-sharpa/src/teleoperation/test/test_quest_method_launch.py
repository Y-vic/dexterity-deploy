from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node


REPOSITORY_ROOT = Path(__file__).parents[3]
TELEOPERATION_LAUNCH = (
    REPOSITORY_ROOT / "src" / "teleoperation" / "launch" / "teleoperation.launch.py"
)
QUEST_LAUNCH = (
    REPOSITORY_ROOT / "src" / "quest_node" / "launch" / "quest_teleop.launch.py"
)
QUEST_TEST_LAUNCH = (
    REPOSITORY_ROOT / "src" / "teleoperation" / "launch" / "quest_test.launch.py"
)


def _load_launch(path: Path):
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def _declared_argument(launch_description, name: str) -> DeclareLaunchArgument:
    return next(
        action
        for action in launch_description.entities
        if isinstance(action, DeclareLaunchArgument) and action.name == name
    )


def _default_value(argument: DeclareLaunchArgument) -> str:
    context = LaunchContext()
    return "".join(value.perform(context) for value in argument.default_value)


def _quest_include(launch_description) -> IncludeLaunchDescription:
    return next(
        action
        for action in launch_description.entities
        if isinstance(action, IncludeLaunchDescription)
        and "quest_node" in repr(action.launch_description_source.location)
    )


def _teleoperation_include(launch_description) -> IncludeLaunchDescription:
    return next(
        action
        for action in launch_description.entities
        if isinstance(action, IncludeLaunchDescription)
        and "teleoperation" in repr(action.launch_description_source.location)
    )


def _quest_retarget_node(launch_description) -> Node:
    return next(
        action
        for action in launch_description.entities
        if isinstance(action, Node) and action.node_executable == "quest_retarget"
    )


def _node_parameter(node: Node, name: str):
    for parameter_group in node._Node__parameters:
        if not isinstance(parameter_group, dict):
            continue
        for normalized_name, value in parameter_group.items():
            if "".join(part.text for part in normalized_name) == name:
                return value
    raise AssertionError(f"node parameter {name!r} was not found")


def test_nonlinear_ik_is_the_default_quest_retarget_method():
    for launch_path, argument_name in (
        (TELEOPERATION_LAUNCH, "quest_retarget_method"),
        (QUEST_LAUNCH, "retarget_method"),
        (QUEST_TEST_LAUNCH, "quest_retarget_method"),
    ):
        argument = _declared_argument(_load_launch(launch_path), argument_name)
        assert _default_value(argument) == "nonlinear_ik"
        assert all(
            method in argument.description
            for method in (
                "local_qp",
                "shoulder_prior",
                "nonlinear_ik",
                "elbow_pole",
            )
        )


def test_teleoperation_forwards_each_quest_method_to_quest_launch():
    launch_description = _load_launch(TELEOPERATION_LAUNCH)
    method_value = dict(_quest_include(launch_description).launch_arguments)[
        "retarget_method"
    ]

    for method in ("local_qp", "shoulder_prior", "nonlinear_ik", "elbow_pole"):
        context = LaunchContext()
        context.launch_configurations["quest_retarget_method"] = method
        assert method_value.perform(context) == method


def test_quest_dry_run_forwards_method_and_nonlinear_filter():
    launch_description = _load_launch(QUEST_TEST_LAUNCH)
    launch_arguments = dict(
        _teleoperation_include(launch_description).launch_arguments
    )
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "quest_retarget_method": "nonlinear_ik",
            "quest_nonlinear_filter_enabled": "false",
        }
    )

    assert (
        launch_arguments["quest_retarget_method"].perform(context)
        == "nonlinear_ik"
    )
    assert (
        launch_arguments["quest_nonlinear_filter_enabled"].perform(context)
        == "false"
    )


def test_quest_launch_forwards_method_specific_parameters_to_retarget_node():
    retarget_node = _quest_retarget_node(_load_launch(QUEST_LAUNCH))
    expected_launch_configurations = {
        "retarget_method": "retarget_method",
        "shoulder_prior_wrist_position_cost": (
            "shoulder_prior_wrist_position_cost"
        ),
        "shoulder_prior_wrist_orientation_cost": (
            "shoulder_prior_wrist_orientation_cost"
        ),
        "shoulder_prior_orientation_cost": "shoulder_prior_orientation_cost",
        "nonlinear_translation_cost": "nonlinear_translation_cost",
        "nonlinear_rotation_cost": "nonlinear_rotation_cost",
        "nonlinear_posture_cost": "nonlinear_posture_cost",
        "nonlinear_smoothness_cost": "nonlinear_smoothness_cost",
        "nonlinear_filter_enabled": "nonlinear_filter_enabled",
    }

    for parameter_name, launch_configuration in expected_launch_configurations.items():
        value = _node_parameter(retarget_node, parameter_name)
        substitution = (
            value.value[0]
            if hasattr(value, "value")
            else value[0]
        )
        assert substitution.variable_name[0].text == launch_configuration
