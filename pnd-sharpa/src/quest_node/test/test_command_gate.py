import pytest

from quest_node.command_gate import (
    ADAM_COMMAND_JOINTS_19,
    ARM_JOINTS,
    NECK_WAIST_JOINTS,
    TrackingStatus,
    TrackingWatchdog,
    make_command_positions,
    positions_from_joint_arrays,
)


def arm_positions():
    return {name: float(index + 1) for index, name in enumerate(ARM_JOINTS)}


def test_command_matches_noitom_19d_order_and_bias_behavior():
    bias = {name: -float(index + 1) for index, name in enumerate(NECK_WAIST_JOINTS)}

    command, source = make_command_positions(
        arm_positions(),
        fix_neck_waist=True,
        bias_positions=bias,
        bias_fresh=True,
    )

    assert source == "bias_command"
    assert list(command) == list(ADAM_COMMAND_JOINTS_19)
    assert [command[name] for name in NECK_WAIST_JOINTS] == [-1, -2, -3, -4, -5]
    assert [command[name] for name in ARM_JOINTS] == list(range(1, 15))


def test_stale_bias_falls_back_to_zero_like_noitom():
    command, source = make_command_positions(
        arm_positions(),
        fix_neck_waist=True,
        bias_positions={name: 9.0 for name in NECK_WAIST_JOINTS},
        bias_fresh=False,
    )

    assert source == "zero_fallback"
    assert [command[name] for name in NECK_WAIST_JOINTS] == [0.0] * 5


def test_retarget_mode_forwards_all_neck_waist_and_arm_positions():
    raw = {name: float(index + 1) for index, name in enumerate(ADAM_COMMAND_JOINTS_19)}

    command, source = make_command_positions(
        raw,
        fix_neck_waist=False,
        bias_positions={name: -1.0 for name in NECK_WAIST_JOINTS},
        bias_fresh=True,
    )

    assert source == "retarget"
    assert list(command) == list(ADAM_COMMAND_JOINTS_19)
    assert command == raw


def test_joint_names_accept_mink_prefix_and_reject_non_finite_values():
    names = [name.removeprefix("dof_pos/") for name in ARM_JOINTS]
    positions = list(range(len(names)))

    parsed = positions_from_joint_arrays(names, positions)

    assert list(parsed) == list(ARM_JOINTS)
    positions[3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        positions_from_joint_arrays(names, positions)


def test_tracking_watchdog_only_refreshes_on_new_real_frame_sequence():
    watchdog = TrackingWatchdog(0.2)
    frame = TrackingStatus(
        event="frame",
        sequence=10,
        connected=True,
        calibrated=True,
        tracking_fresh=True,
    )
    watchdog.observe(frame, 1.0)
    assert watchdog.is_fresh(1.19)

    watchdog.observe(frame, 1.19)
    assert not watchdog.is_fresh(1.21)

    watchdog.observe(
        TrackingStatus(
            event="frame",
            sequence=11,
            connected=True,
            calibrated=True,
            tracking_fresh=True,
        ),
        1.21,
    )
    assert watchdog.is_fresh(1.4)


def test_uncalibrated_status_blocks_tracking_immediately():
    watchdog = TrackingWatchdog(0.2)
    watchdog.observe(
        TrackingStatus("frame", 1, True, True, True),
        1.0,
    )
    watchdog.observe(
        TrackingStatus("status", 1, True, False, True),
        1.05,
    )

    assert not watchdog.is_fresh(1.06)
