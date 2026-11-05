# One synchronized set of camera frames for the UR + SharpA embodiment.
#
# The three views are kept as NAMED, SEPARATE fields. There is deliberately no
# merged / mosaic image field: the policy metadata addresses ego_cam,
# left_wrist_cam and right_wrist_cam independently, so stitching them here
# would destroy per-view identity and per-view sampling time.
#
# Every view carries its OWN sampling timestamp because the three sensors are
# not hardware-triggered together. Do not substitute header.stamp for these.

# Publish time of this aggregated set (workstation clock).
std_msgs/Header header

# Image codec for all three payloads. Contract fixes this to "jpeg".
string encoding "jpeg"

# Ego (head / scene) view.
uint64 ego_timestamp_ns          # sensor sampling time, nanoseconds
uint8[] ego                      # JPEG bytes
bool ego_valid                   # false => ego is stale or unavailable

# Left wrist view.
uint64 left_wrist_timestamp_ns   # sensor sampling time, nanoseconds
uint8[] left_wrist               # JPEG bytes
bool left_wrist_valid            # false => left_wrist is stale or unavailable

# Right wrist view.
uint64 right_wrist_timestamp_ns  # sensor sampling time, nanoseconds
uint8[] right_wrist              # JPEG bytes
bool right_wrist_valid           # false => right_wrist is stale or unavailable
