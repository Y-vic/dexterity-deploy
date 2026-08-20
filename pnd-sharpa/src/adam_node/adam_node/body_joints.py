"""Adam joint-name contracts and explicit topic mappings."""

LOWSTATE_REAL_JOINTS_31 = [
    "dof_pos/hipPitch_Left",
    "dof_pos/hipRoll_Left",
    "dof_pos/hipYaw_Left",
    "dof_pos/kneePitch_Left",
    "dof_pos/anklePitch_Left",
    "dof_pos/ankleRoll_Left",
    "dof_pos/hipPitch_Right",
    "dof_pos/hipRoll_Right",
    "dof_pos/hipYaw_Right",
    "dof_pos/kneePitch_Right",
    "dof_pos/anklePitch_Right",
    "dof_pos/ankleRoll_Right",
    "dof_pos/waistRoll",
    "dof_pos/waistPitch",
    "dof_pos/waistYaw",
    "dof_pos/neckYaw",
    "dof_pos/neckPitch",
    "dof_pos/shoulderPitch_Left",
    "dof_pos/shoulderRoll_Left",
    "dof_pos/shoulderYaw_Left",
    "dof_pos/elbow_Left",
    "dof_pos/wristYaw_Left",
    "dof_pos/wristPitch_Left",
    "dof_pos/wristRoll_Left",
    "dof_pos/shoulderPitch_Right",
    "dof_pos/shoulderRoll_Right",
    "dof_pos/shoulderYaw_Right",
    "dof_pos/elbow_Right",
    "dof_pos/wristYaw_Right",
    "dof_pos/wristPitch_Right",
    "dof_pos/wristRoll_Right",
]

ADAM_PHYSICAL_JOINTS_31 = LOWSTATE_REAL_JOINTS_31

WAIST_JOINTS = [
    "dof_pos/waistRoll",
    "dof_pos/waistPitch",
    "dof_pos/waistYaw",
]
NECK_JOINTS = [
    "dof_pos/neckYaw",
    "dof_pos/neckPitch",
]
LEFT_ARM_JOINTS = [
    "dof_pos/shoulderPitch_Left",
    "dof_pos/shoulderRoll_Left",
    "dof_pos/shoulderYaw_Left",
    "dof_pos/elbow_Left",
    "dof_pos/wristYaw_Left",
    "dof_pos/wristPitch_Left",
    "dof_pos/wristRoll_Left",
]
RIGHT_ARM_JOINTS = [
    "dof_pos/shoulderPitch_Right",
    "dof_pos/shoulderRoll_Right",
    "dof_pos/shoulderYaw_Right",
    "dof_pos/elbow_Right",
    "dof_pos/wristYaw_Right",
    "dof_pos/wristPitch_Right",
    "dof_pos/wristRoll_Right",
]
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

ADAM_COMMAND_JOINTS_19 = WAIST_JOINTS + NECK_JOINTS + ARM_JOINTS

CONTROL_HAND_JOINTS_12 = [
    "dof_pos/hand_pinky_Left",
    "dof_pos/hand_ring_Left",
    "dof_pos/hand_middle_Left",
    "dof_pos/hand_index_Left",
    "dof_pos/hand_thumb_1_Left",
    "dof_pos/hand_thumb_2_Left",
    "dof_pos/hand_pinky_Right",
    "dof_pos/hand_ring_Right",
    "dof_pos/hand_middle_Right",
    "dof_pos/hand_index_Right",
    "dof_pos/hand_thumb_1_Right",
    "dof_pos/hand_thumb_2_Right",
]

PND_CONTROL_HAND_PLACEHOLDER_VALUE = 1000.0
PND_CONTROL_ROOT_HEIGHT_NAME = "root_pos/z"
PND_CONTROL_ROOT_HEIGHT_VALUE = 1.0

ROBOT_STATE_HAND_JOINTS_12 = [
    "dof_pos/L_pinky_joint",
    "dof_pos/L_ring_joint",
    "dof_pos/L_middle_joint",
    "dof_pos/L_index_joint",
    "dof_pos/L_thumb_PIP_joint",
    "dof_pos/L_thumb_MCP_joint",
    "dof_pos/R_pinky_joint",
    "dof_pos/R_ring_joint",
    "dof_pos/R_middle_joint",
    "dof_pos/R_index_joint",
    "dof_pos/R_thumb_PIP_joint",
    "dof_pos/R_thumb_MCP_joint",
]

ADAM_CONTROL_JOINTS_32 = (
    ADAM_COMMAND_JOINTS_19
    + [PND_CONTROL_ROOT_HEIGHT_NAME]
    + CONTROL_HAND_JOINTS_12
)
ADAM_ROBOT_STATE_JOINTS_31 = ADAM_COMMAND_JOINTS_19 + ROBOT_STATE_HAND_JOINTS_12

LOWSTATE_INDEX_BY_JOINT = {
    name: index for index, name in enumerate(LOWSTATE_REAL_JOINTS_31)
}

KNOWN_JOINTS = (
    set(LOWSTATE_REAL_JOINTS_31)
    | set(ADAM_CONTROL_JOINTS_32)
    | set(ADAM_ROBOT_STATE_JOINTS_31)
)


def canonical_body_name(name: str) -> str:
    """Return the dof_pos joint name used by the Adam ROS contracts."""
    if name in KNOWN_JOINTS:
        return name
    prefixed = f"dof_pos/{name}"
    if prefixed in KNOWN_JOINTS:
        return prefixed
    return name
