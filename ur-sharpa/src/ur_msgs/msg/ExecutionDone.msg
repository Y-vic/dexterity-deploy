# Terminal report for one executed slice. action_execute publishes this EXACTLY
# ONCE per ExecutableAction, whether the slice completed, failed or was aborted.
#
# The policy client only accepts a done whose identity matches the action it is
# waiting on, then converts it into the execution_feedback of the next
# observation. A missing or mismatched done stalls the synchronous state
# machine, so publish it on every exit path including exceptions.

std_msgs/Header header

# --- Identity passthrough, copied verbatim from the ExecutableAction ---
string session_id
uint64 request_id
string action_id
uint64 revision

# Slice bounds this report covers, echoed from the ExecutableAction.
uint32 execute_start
uint32 execute_length

# CUMULATIVE step count. FROZEN semantics:
#
#     executed_steps = execute_start + executed_in_slice
#
# It is NOT the slice-local count. The server needs a monotonically increasing
# progress counter to compute lag across replans; a slice-local count resets on
# every replan and deadlocks the policy client, which compares this value
# against the slice bounds. On full success
# executed_steps == execute_start + execute_length.
# Build it with interface/sharpa_interface/server/executed_steps.py rather than
# adding the two numbers by hand.
uint32 executed_steps

# True when the whole slice ran without fault.
bool success

# Always true: this message is only published on termination. Present so the
# wire form matches the sharpa_execution_done.v1 dict key-for-key.
bool done true

# Failure detail, empty string on success (the dict's None).
string error

# Time the slice terminated, nanoseconds.
uint64 timestamp_ns
