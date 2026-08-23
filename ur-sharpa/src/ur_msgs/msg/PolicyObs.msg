# Aggregated observation REFERENCE, published once per observation build,
# after forward kinematics has resolved the EEF poses.
#
# This is a join key, not a data carrier. Bulk modalities (JPEG, deformation)
# stay on their own topics; this message records WHICH sample of each modality
# went into the observation, so the dashboard and the recorder can reconstruct
# the exact policy input without the observation path paying for a second copy
# of ~600 KB of tactile data.
#
# The small state vectors ARE inlined, because they are the part that FK just
# produced and there is no other topic that carries the post-FK result.

std_msgs/Header header

# Monotonic observation counter, restarts at 0 on session reset.
uint64 seq

# Identity of the inference request this observation was built for.
string session_id
uint64 request_id
string metadata_format_id

# Time the observation was assembled, nanoseconds.
uint64 timestamp_ns

# Task prompt sent alongside this observation.
string prompt

# --- Inlined post-FK state ---
# Arm joint angles, radians, UR base-to-wrist order. Shape (6,) per arm.
float32[6] left_joint
float32[6] right_joint

# Canonical EEF pose, shape (9,) per arm: absolute xyz in canonical_root followed
# by the first two COLUMNS of root-from-wrist R:
# [x,y,z,r00,r10,r20,r01,r11,r21].
float32[9] left_eef
float32[9] right_eef

# EEF frame for both arms; must be "canonical".
string eef_frame

# Hand joint positions, radians. Shape (44,), left hand [0:22] then right
# [22:44], FROZEN order from sharpa.py.
float32[44] hand_joint

# --- References into the modality topics ---
# Sampling timestamps (nanoseconds) of the samples that were selected. Zero
# means that modality had no sample and was reported absent to the policy.
uint64 ego_timestamp_ns
uint64 left_wrist_timestamp_ns
uint64 right_wrist_timestamp_ns
uint64 tau_timestamp_ns
uint64 wrench_timestamp_ns
uint64 deformation_timestamp_ns

# Per-modality history depth actually supplied, in the order
# [ego, left_wrist, right_wrist, state, tau, wrench, deformation].
# Shape (7,). Shorter than the metadata request means the buffer was still
# filling.
uint32[7] history_len

# True when every modality the metadata marked required was present.
bool complete

# Human-readable note when complete is false, e.g. "deformation buffer empty".
string incomplete_reason
