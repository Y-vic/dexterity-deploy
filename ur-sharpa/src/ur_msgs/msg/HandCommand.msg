# SharpA dual-hand joint position command, one control step.
#
# Layout is the FROZEN 44-DOF concatenation from
# interface/sharpa_interface/sharpa.py: LEFT hand [0:22] then RIGHT hand
# [22:44]. Within a hand the order is thumb, index, middle, ring, pinky as
# listed in JOINT_NAMES_PER_HAND. Never index this array with magic offsets;
# use SharpAJointIndex.

# SOURCE_* values for the `source` field. These mirror ArmCommand so both
# hardware outlets use the same control-state arbitration contract.
uint8 SOURCE_UNKNOWN = 0
uint8 SOURCE_BIAS = 1
uint8 SOURCE_VIVE = 2
uint8 SOURCE_ACTOR = 3

std_msgs/Header header

# Producer of this step; one of SOURCE_*.
uint8 source

# Identity passthrough. Empty action_id for teleoperation sources.
string action_id
uint64 revision
uint32 step_index          # index within the current slice, 0-based

# Target joint positions, radians. Shape (44,), left hand first.
float32[44] q_cmd

# Per-hand enable. A hand with enable=false keeps its last commanded pose and
# its 22-DOF slice of q_cmd is ignored.
bool left_enable
bool right_enable

# Validate and acknowledge this command without calling the SharpA SDK.
# Only Actor commands may use dry-run.
bool dry_run
