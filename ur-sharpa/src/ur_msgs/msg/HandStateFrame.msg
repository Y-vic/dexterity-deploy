# SharpA dual-hand measured state.
#
# Layout is the FROZEN 44-DOF concatenation: LEFT hand [0:22] then RIGHT hand
# [22:44], thumb -> index -> middle -> ring -> pinky within each hand
# (interface/sharpa_interface/sharpa.py JOINT_NAMES_PER_HAND).

std_msgs/Header header

# Sampling time reported by the hand controller, nanoseconds.
uint64 timestamp_ns

# Measured joint positions, radians. Shape (44,), left hand first.
float32[44] q

# Measured joint velocities, radians per second. Shape (44,), same layout.
# All zeros when dq_valid is false.
float32[44] dq
bool dq_valid

# Measured joint torques, newton-meters. Shape (44,), same layout.
# All zeros when tau_valid is false. TauFrame carries the same quantity split
# per hand for the sensor buffer path; this field is for the state path.
float32[44] tau
bool tau_valid

# Per-hand freshness. A hand with valid=false has stale or missing telemetry
# and its 22-DOF slice must not be buffered.
bool left_valid
bool right_valid

# Last full 44D target successfully handed to the SharpA SDK. This is the
# executed-command cache used by recording; it is not inferred from measured q.
# q_cmd_valid is false after a stop/source transition, SDK command failure, or
# partial-hand command. q_cmd_source uses HandCommand.SOURCE_* values.
float32[44] q_cmd
bool q_cmd_valid
uint64 q_cmd_timestamp_ns
uint8 q_cmd_source
