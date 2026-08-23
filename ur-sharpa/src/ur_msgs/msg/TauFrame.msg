# Per-hand SharpA joint torque, sensor buffer path.
#
# Shape (22,) per hand, FROZEN. Order matches JOINT_NAMES_PER_HAND:
# thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP,
# index_MCP_FE, index_MCP_AA, index_PIP, index_DIP, middle_*, ring_*,
# pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP.
#
# Kept split per hand (not the 44-D concatenation) because the policy sensor
# buffer indexes tau by hand and one hand can drop out independently.

std_msgs/Header header

# Sampling time reported by the hand controller, nanoseconds.
uint64 timestamp_ns

# Joint torques, newton-meters. Shape (22,) per hand.
float32[22] left
float32[22] right

# Per-hand freshness. false => that hand's 22 values are stale, do not buffer.
bool left_valid
bool right_valid
