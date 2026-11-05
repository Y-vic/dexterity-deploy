# Dual-arm UR state: joint angles plus forward-kinematics EEF pose.
#
# Published by ur_node at a fixed rate. This is a "fact" topic: consumers
# (policy_client buffers, dashboard, recorder) treat it as read-only truth.

std_msgs/Header header

# Sampling time of the underlying RTDE packet, nanoseconds.
uint64 timestamp_ns

# Measured joint angles, radians, UR base-to-wrist order (shoulder_pan,
# shoulder_lift, elbow, wrist_1, wrist_2, wrist_3). Shape (6,) per arm.
float32[6] left_joint
float32[6] right_joint

# Canonical EEF pose, shape (9,) per arm. Elements 0..2 are absolute position
# in metres in canonical_root (+X left, +Y up, +Z forward). Elements 3..8 are the
# first two COLUMNS of root-from-wrist R:
# [x,y,z,r00,r10,r20,r01,r11,r21]. Wrist child frames are left_hand_C_MC and
# right_hand_C_MC, matching the PND contract.
float32[9] left_eef
float32[9] right_eef

# Raw UR RTDE TCP poses, [x,y,z,rx,ry,rz] where rotation is a UR rotation
# vector. These are retained alongside the derived canonical wrist pose.
float32[6] left_tcp_pose6
float32[6] right_tcp_pose6

# Frame for both EEF poses. Must be "canonical".
string eef_frame

# ACK / validity status.
# normal_mode: both UR controllers report normal (not protective-stopped, not
#              in reduced mode); false means commands will be refused.
bool normal_mode
# valid: this frame is fully populated and fresh. false => do not buffer it.
bool valid
# Set when the last arm command was accepted by both controllers.
bool last_command_acked
# Human-readable reason the frame is invalid or the command was refused.
# Empty when valid and acked.
string status_message
