# One trajectory after Actor IK has resolved the executable action.
# Dashboard and preview logging consume this message instead of subscribing to
# policy/replay input topics, so the displayed joints are the Actor plan.

std_msgs/Header header

string provider
bool dry_run
string trace_dir

string session_id
uint64 request_id
string action_id
uint64 revision
uint64 timestamp_ns

float64 frequency_hz
uint32 action_length
uint32 execute_start
uint32 execute_length
uint32 rtc

string eef_frame
float32[] left_eef
float32[] right_eef
float32[] left_arm_joint
float32[] right_arm_joint
float32[] hand_joint

bool ik_applied
string diagnostics_json
