# Dual-arm UR joint target, one control step.
#
# This is the ONLY arm command type on the wire. IK lives upstream in
# action_ik, so the payload is joint space, never EEF: whoever publishes here
# has already resolved Cartesian intent into UR joint angles.
#
# `source` says which producer owns this step, so ur_node can arbitrate
# priority and reject a step whose producer is not currently gated in.

# SOURCE_* values for the `source` field.
uint8 SOURCE_UNKNOWN = 0
uint8 SOURCE_BIAS = 1    # bias / homing node: move to a fixed calibration pose
uint8 SOURCE_VIVE = 2    # teleoperation tracker stream
uint8 SOURCE_ACTOR = 3   # policy execution (action_execute)

std_msgs/Header header

# Producer of this step; one of SOURCE_*.
uint8 source

# Identity passthrough. Empty action_id for non-policy sources (bias, vive).
string action_id
uint64 revision
uint32 step_index          # index within the current slice, 0-based

# Nominal duration of this step, seconds. Drives the UR servo interpolation.
float64 period_s

# Joint targets, radians, UR base-to-wrist order (shoulder_pan, shoulder_lift,
# elbow, wrist_1, wrist_2, wrist_3). Shape (6,) per arm, fixed by UR5 hardware.
float32[6] left_joint
float32[6] right_joint

# Per-arm enable. A disabled arm ignores its joint slice and is not sent a
# servoJ command, allowing the other teleop arm to continue independently.
bool left_enable
bool right_enable

# Validate and acknowledge this command without writing to the UR controller.
# Only Actor commands may use dry-run.
bool dry_run
