# replay_node

`replay_node` reads one finalized `ur_sharpa_sample.v1` episode and publishes
direct-joint `ur_msgs/ExecutableAction` chunks on `/replay/executable_action`.
The actor accepts this topic only when `/control/execution_provider` is
`replay`; policy and replay therefore cannot drive the actor concurrently.

The loader requires `COMPLETE`, validates the manifest/schema and SHA-256
checksums, requires the arm and selected hand channel to pass the joint-only
validation report, and preserves every timeline row. Invalid SharpA rows are
repaired by holding the last valid hand pose rather than deleting a timestep.
Before a run, the node interpolates from measured UR/SharpA joints to the next
sample row. Replay is a finite buffer and does not loop. Each execution has a
unique identity, and every chunk waits for a full identity-matched
`/policy/execution_done` before the next chunk is published.

Dashboard commands are JSON on `/replay/command`: `load`, `start`, `pause`,
`resume`, `stop`, and `unload`. Status is JSON on `/replay/status`. Dry run uses
the same Actor IK chunk/ACK path as live execution, with hardware writes
disabled by the execution gate. Pause retains the current cursor. Stop closes
the gate and resets the cursor while retaining the validated sample in memory;
unload additionally releases the sample and sequencer. A load that only changes
execution settings reuses the same in-memory sample.
