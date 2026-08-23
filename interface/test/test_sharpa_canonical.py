from __future__ import annotations

from sharpa_interface import sharpa


def test_canonical_root_and_wrist_axis_contract() -> None:
    assert sharpa.CANONICAL_EEF_FRAME == "canonical"
    assert sharpa.CANONICAL_ROOT_AXIS_ORDER == ("x", "y", "z")
    assert sharpa.CANONICAL_ROOT_AXES == ("left", "up", "forward")
    assert dict(sharpa.CANONICAL_ROOT_AXIS_DIRECTIONS) == {
        "x": "left",
        "y": "up",
        "z": "forward",
    }
    assert sharpa.CANONICAL_WRIST_SIDE_ORDER == ("left", "right")
    assert tuple(sharpa.CANONICAL_WRIST_FRAME_BY_SIDE.values()) == (
        "left_hand_C_MC",
        "right_hand_C_MC",
    )
    assert sharpa.CANONICAL_WRIST_AXIS_ORDER == ("x", "y", "z")


def test_canonical_pose9_matches_pnd_column_order() -> None:
    assert sharpa.CANONICAL_ROT6D_CONVENTION == "first_two_matrix_columns"
    assert sharpa.CANONICAL_EEF_POSE9_ORDER == (
        "x",
        "y",
        "z",
        "r00",
        "r10",
        "r20",
        "r01",
        "r11",
        "r21",
    )


def test_interface_contains_no_ur_mount_bias() -> None:
    public_names = dir(sharpa)
    assert not any("29_5" in name or "FLANGE" in name or "TCP_OFFSET" in name
                   for name in public_names)
