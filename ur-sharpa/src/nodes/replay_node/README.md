# replay_node

`replay_node` reads one finalized `ur_sharpa_sample.v1` episode and publishes
direct-joint `ur_msgs/ExecutableAction` chunks on `/replay/executable_action`.
The actor accepts this topic only when `/control/execution_provider` is
`replay`; policy and replay therefore cannot drive the actor concurrently.

The loader requires `COMPLETE`, validates the manifest/schema and SHA-256
checksums, requires the arm and selected hand channel to pass the joint-only
validation report, and preserves every timeline row. Invalid SharpA rows are
repaired by holding the last valid hand pose rather than deleting a timestep.
Before a live run, the node interpolates from measured UR/SharpA joints to the
next sample row. Loop mode performs the same alignment before returning to row
zero. Each execution has a unique identity, and every chunk waits for a full
identity-matched `/policy/execution_done` before the next chunk is published.

Dashboard commands are JSON on `/replay/command`: `load`, `start`, `pause`,
`resume`, and `stop`. Status is JSON on `/replay/status`; dry-run trajectories
are published only on `/replay/preview_action`. Dry run automatically previews
all chunks at the configured rate with an independent preview cursor; it never
advances the live execution cursor. Pause and stop request `stop_deploy` on
`/control/ui_command`, so the execution gate is closed even when the command
does not originate from Dashboard.
