# ACK / NACK for one hardware command step.
#
# Published by ur_node and sharpa_node for every ArmCommand / HandCommand they
# consume, so the executor can distinguish "delivered" from "accepted".

std_msgs/Header header

# Identity of the command being acknowledged, copied verbatim.
uint8 source
string action_id
uint64 revision
uint32 step_index

# Which device produced this result: "ur" or "sharpa".
string device

# True when the device accepted and applied the step.
bool success

# Machine-readable rejection reason, empty on success.
# e.g. "ur_command_failed", "gate_closed", "dimension_mismatch", "stale_ttl".
string failure_code

# Human-readable rejection detail, empty on success.
string failure_message

# Time the device finished handling the step, nanoseconds.
uint64 timestamp_ns
