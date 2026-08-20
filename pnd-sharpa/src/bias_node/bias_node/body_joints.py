"""Adam command JointState contract used by the Bias UI."""

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

ADAM_COMMAND_JOINTS_19 = (
    WAIST_JOINTS + NECK_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
)

DEFAULT_JOINT_VALUES = {name: 0.0 for name in ADAM_COMMAND_JOINTS_19}

UPPER_BODY_EDITABLE_GROUPS = [
    ("Neck", NECK_JOINTS),
    ("Left Arm", LEFT_ARM_JOINTS),
    ("Right Arm", RIGHT_ARM_JOINTS),
    ("Waist", WAIST_JOINTS),
]

UPPER_BODY_EDITABLE_JOINTS = [
    joint for _, joints in UPPER_BODY_EDITABLE_GROUPS for joint in joints
]


def canonical_body_name(name: str) -> str:
    """Return the Adam command joint name, accepting unprefixed input."""
    if name in ADAM_COMMAND_JOINTS_19:
        return name
    prefixed = f"dof_pos/{name}"
    if prefixed in ADAM_COMMAND_JOINTS_19:
        return prefixed
    return name
