# Per-hand SharpA fingertip wrench (6-axis force/torque per fingertip).
#
# FROZEN shape (5, 6) per hand, flattened ROW-MAJOR (C order):
#
#   index = finger * 6 + axis
#
# ROW order (finger, 5 rows) — pinky to thumb, the tactile wire order from
# sharpa.py TACTILE_FINGERS. This is intentionally the REVERSE of the
# hand-joint finger order, so never reuse a joint index here:
#   row 0 = pinky, 1 = ring, 2 = middle, 3 = index, 4 = thumb
#
# COLUMN order (axis, 6 columns):
#   col 0..2 = Fx, Fy, Fz   in newtons
#   col 3..5 = Tx, Ty, Tz   in newton-meters
#
# Frame: fingertip-local. +Z along the fingertip normal (out of the pad),
# +X along the finger long axis (proximal -> distal), +Y = Z x X.

std_msgs/Header header

# Sampling time reported by the tactile controller, nanoseconds.
uint64 timestamp_ns

# Flattened (5, 6) wrench per hand. Shape (30,) on the wire.
float32[30] left
float32[30] right

# Per-fingertip freshness, shape (5,), same pinky->thumb row order as above.
# A false entry means that fingertip's 6 values are zero-filled placeholders.
bool[5] left_valid
bool[5] right_valid
