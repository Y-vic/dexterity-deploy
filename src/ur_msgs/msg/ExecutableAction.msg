# One executable action slice: the policy chunk truncated to
# [execute_start : execute_start + execute_length], on its way to action_ik and
# then action_execute.
#
# IDENTITY PASSTHROUGH IS MANDATORY. action_ik may replace or add the joint
# plan; it must NOT touch session_id, request_id, action_id, revision,
# timestamp_ns or any execution field. action_execute matches the
# ExecutionDone it publishes against exactly these values, so any rewrite
# silently desynchronizes the policy client.

std_msgs/Header header

# --- Identity passthrough (copied verbatim from the server action) ---
string session_id
uint64 request_id
string action_id
uint64 revision
uint64 timestamp_ns     # server-side action creation time, nanoseconds

# --- Execution params ---
float64 frequency_hz    # step rate, hertz; step period is 1/frequency_hz
uint32 action_length    # length of the ORIGINAL chunk, before truncation
uint32 execute_start    # index of the first step of this slice in the chunk
uint32 execute_length   # number of steps in this slice == first dim of arrays
uint32 rtc              # 0: fetch at completion; N: fetch with N steps remaining

# --- Action slice ---
# Every array below is flattened ROW-MAJOR with execute_length rows:
#   index = step * dim + component,  step in [0, execute_length)
# An empty array means that channel is absent for this action; a present array
# must have exactly execute_length * dim elements.

# EEF trajectory, dim 9 per step: [x,y,z] meters then rot6d = first two
# COLUMNS of R. Length execute_length * 9.
# Poses are in canonical_root with left/right hand_C_MC wrist children.
# Always absolute — relative EEF is rejected at the interface boundary.
float32[] left_eef
float32[] right_eef

# Frame the EEF trajectories are expressed in; must be "canonical".
string eef_frame

# Arm joint trajectory, radians, dim 6 per step, UR base-to-wrist order.
# Length execute_length * 6. Empty on the action_ik INPUT topic and filled on
# its output; action_execute requires it to be present.
float32[] left_arm_joint
float32[] right_arm_joint

# Hand joint trajectory, radians, dim 44 per step: left hand [0:22] then right
# [22:44], FROZEN order from sharpa.py. Length execute_length * 44.
float32[] hand_joint
