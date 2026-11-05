# ROS node packages

All executable ROS 2 packages live here. Shared code and messages remain one
level above in `src/ur_common/` and `src/ur_msgs/`.

The two supported graphs are composed by `ur_launch`:

- Teleop: shared hardware + `vive_node`, `manus_node`, `recording_node`, and `monitor_node`.
- Deploy: shared hardware + `policy_client_node`, `replay_node`,
  `actor_ik_node`, and `dashboard_node`. `policy_client_node` owns the policy
  observation buffers and model transport. `status_node` selects
  exactly one execution provider (`policy` or `replay`) while the machine is
  stopped; both providers use the same actor and hardware safety gates.

Do not create a second hardware launch: `ur_node` and `sharpa_node` are the only
owners of their physical devices in both modes. Build and start instructions
are in the workspace [README](../../README.md).
