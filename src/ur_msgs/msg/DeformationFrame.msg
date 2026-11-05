# Per-hand SharpA fingertip deformation images.
#
# FROZEN shape (5, 240, 240) uint8 per hand, flattened ROW-MAJOR (C order):
#
#   index = finger * 240 * 240 + row * 240 + col
#
# PLANE order (finger, 5 planes) — pinky to thumb, the tactile wire order from
# sharpa.py TACTILE_FINGERS, the REVERSE of the hand-joint finger order:
#   plane 0 = pinky, 1 = ring, 2 = middle, 3 = index, 4 = thumb
#
# Each plane is a single-channel 240x240 image, row-major, top row first.
# Values are raw uint8 deformation magnitude (dimensionless sensor counts, no
# physical unit); they are NOT calibrated to millimeters.
#
# Payload is 5*240*240 = 288000 bytes per hand. Sent as an unbounded uint8[]
# rather than a fixed array so DDS fragmentation and zero-copy behave, and so a
# dropped hand can publish an empty vector instead of 288 KB of zeros.

std_msgs/Header header

# Sampling time reported by the tactile controller, nanoseconds.
uint64 timestamp_ns

# Plane / image geometry, restated on the wire for defensive reshaping.
# Contract fixes these to 5, 240, 240.
uint32 fingertip_count 5
uint32 height 240
uint32 width 240

# Flattened (5, 240, 240) uint8 per hand. Length 288000, or 0 if that hand has
# no data at all this frame.
uint8[] left
uint8[] right

# Per-fingertip freshness, shape (5,), same pinky->thumb plane order as above.
# A false entry means that plane is a zero-filled placeholder.
bool[5] left_valid
bool[5] right_valid
