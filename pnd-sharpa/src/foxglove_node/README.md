# Foxglove Visualization

This package starts the Foxglove visualization path for Adam + Sharpa command
targets. It does not control hardware.

## Data Flow

```text
/adam_command_joint_states + /sharpa_command_joint_states
  -> foxglove_joint_state_merge
  -> /foxglove/joint_states
  -> robot_state_publisher
  -> /robot_description, /tf, /tf_static
  -> foxglove_bridge on ws://<robot-ip>:8765
```

The merge node reads the Adam + Sharpa URDF and publishes every movable joint.
Joints that have not received a command yet stay at `0.0`, so the model can be
shown before teleoperation enters an active state.

## Scene JSON

The Foxglove 3D panel scene config is stored at:

```text
src/foxglove_node/config/noitom_scene_panel.json
```

Paste this JSON into the Foxglove 3D panel settings. It configures the grid,
camera, `world` follow frame, mesh up-axis, and makes `/robot_description`
visible. The JSON is only a display layout; the ROS side still needs to provide:

```text
/robot_description
/tf
/tf_static
/foxglove/joint_states
```

## Run

From the workspace:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch foxglove_node foxglove_node.launch.py
```

Connect Foxglove Studio to:

```text
ws://<robot-ip>:8765
```

For the current robot network, common candidates are the host IPs on
`10.10.20.x` or `192.168.5.x`.
