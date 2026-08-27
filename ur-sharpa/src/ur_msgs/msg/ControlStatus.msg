# Aggregated control-layer status: three independent state machines plus the
# reason the next transition is currently blocked.
#
# Published by the control node on every state change and at a low heartbeat
# rate. Dashboard and recorder observe it; nobody schedules from it.

# --- UR arm state machine (3 states) ---
uint8 UR_STOPPED = 0   # controllers idle or protective-stopped, no motion
uint8 UR_HOMED = 1     # at the bias/home pose, servo loop ready, not streaming
uint8 UR_RUNNING = 2   # accepting and executing ArmCommand steps

# --- SharpA hand state machine (2 states) ---
uint8 SHARPA_STOPPED = 0   # hand controller not accepting commands
uint8 SHARPA_RUNNING = 1   # accepting and executing HandCommand steps

# --- Recording state machine (2 states) ---
uint8 RECORD_IDLE = 0        # not writing an episode
uint8 RECORD_RECORDING = 1   # writing an episode to disk

std_msgs/Header header

# Publish time of this status, nanoseconds.
uint64 timestamp_ns

# Current state of each machine; one of the corresponding constants above.
uint8 ur_state
uint8 sharpa_state
uint8 record_state

# Which ArmCommand.SOURCE_* is currently allowed to drive the arms.
# SOURCE_UNKNOWN means no producer is gated in and all arm commands are refused.
uint8 active_source

# Deploy-only soft pause. The actor keeps the current chunk but must not
# publish another hardware command or ExecutionDone until this becomes false.
bool execution_paused

# Actor execution is enabled in validation-only preview mode. UR and SharpA
# remain stopped; Actor commands carry dry_run=true and drivers must not write
# to hardware.
bool execution_dry_run

# Human-readable explanation of why the next transition is blocked, e.g.
# "ur not homed: run bias first" or "sharpa telemetry stale 0.8s".
# Empty string means nothing is blocked right now.
string gate_reason

# Identifier of the episode being recorded. Empty when record_state is IDLE.
string session_id
