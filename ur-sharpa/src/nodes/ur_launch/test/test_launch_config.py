from ur_launch.launch_config import _ur_motion_parameters


def test_teleop_enables_final_ur_motion_guards():
    parameters = _ur_motion_parameters("teleop")

    assert parameters["enable_motion_limits"] is True
    assert parameters["enable_joint_step_limit"] is True
    assert parameters["enable_joint_clip"] is True
    assert "max_abs_joint_rad" in parameters
    assert "max_joint_step_rad" in parameters
