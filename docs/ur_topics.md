# UR ROS topic 契约

**所有节点必须用这里的 topic 名和消息类型。** 这是各节点并行开发的唯一对接依据；节点内部可以
有自己的参数，但 topic 名的默认值必须与本表一致，否则图接不起来。

命名规则：`/ur/*` 硬件、`/sharpa/*` 手、`/camera/*` 相机、`/teleop/*` 遥操、`/control/*` 状态、
`/policy/*` 部署。

## 硬件层

| Topic | 类型 | Pub | Sub |
|---|---|---|---|
| `/ur/state` | `UrStateFrame` | `ur_node` | `robot_states_node` `bias_node` `vive_node` `dashboard_node` `recording_node` `foxglove_node` |
| `/ur/command` | `ArmCommand` | `bias_node` `vive_node` `actor_ik_node` | `ur_node` |
| `/ur/command_result` | `HardwareCommandResult` | `ur_node` | `actor_ik_node` `bias_node` `status_node` |
| `/sharpa/state` | `HandStateFrame` | `sharpa_node` | `robot_states_node` `bias_node` `dashboard_node` `recording_node` `foxglove_node` |
| `/sharpa/command` | `HandCommand` | `bias_node` `manus_node` `actor_ik_node` | `sharpa_node` |
| `/sharpa/command_result` | `HardwareCommandResult` | `sharpa_node` | `actor_ik_node` `bias_node` `status_node` |
| `/sharpa/tau` | `TauFrame` | `sharpa_node` | `robot_states_node` `recording_node` |
| `/sharpa/wrench` | `WrenchFrame` | `sharpa_node` | `robot_states_node` `recording_node` |
| `/sharpa/deformation` | `DeformationFrame` | `sharpa_node` | `robot_states_node` `recording_node` |
| `/camera/set` | `CameraSet` | `camera_node` | `robot_states_node` `recording_node` `dashboard_node` `status_node` |
| `/hardware/stop` | `std_msgs/Bool` | `status_node` | `ur_node` `sharpa_node` |

## 控制层

| Topic | 类型 | Pub | Sub |
|---|---|---|---|
| `/control/status` | `ControlStatus` | `status_node` | 所有节点 |
| `/control/record_command` | `std_msgs/String` | `status_node` | `recording_node` |
| `/control/bias_request` | `std_msgs/String` | `status_node` | `bias_node` |
| `/control/bias_status` | `std_msgs/String` | `bias_node` | `status_node` |
| `/control/record_status` | `std_msgs/String` | `recording_node` | `status_node` `dashboard_node` |

`/control/record_command` 取值 `start` / `stop`。`/control/bias_request` 取值
`home_ur` / `home_sharpa` / `home_both`。两个 status 是 JSON 字符串。

## Teleop 层

| Topic | 类型 | Pub | Sub |
|---|---|---|---|
| `/teleop/vive_status` | `std_msgs/String` | `vive_node` | `status_node` `dashboard_node` |
| `/teleop/manus_status` | `std_msgs/String` | `manus_node` | `status_node` `dashboard_node` |

`vive_node` 和 `manus_node` 的实际命令分别发到 `/ur/command`（`source=SOURCE_VIVE`）和
`/sharpa/command`。

## Deploy 层

| Topic | 类型 | Pub | Sub |
|---|---|---|---|
| `/policy/obs` | `PolicyObs` | `robot_states_node` | `policy_client_node` `dashboard_node` |
| `/policy/executable_action` | `ExecutableAction` | `policy_client_node` `replay_node` | `actor_ik_node` `dashboard_node` |
| `/policy/execution_done` | `ExecutionDone` | `actor_ik_node` | `policy_client_node` `dashboard_node` |
| `/policy/status` | `std_msgs/String` | `policy_client_node` | `status_node` `dashboard_node` |
| `/policy/joint_plan` | `ArmCommand` | `actor_ik_node` | `dashboard_node` |

服务：`/policy/build_observation`（`BuildObservation`，`robot_states_node` 提供）、
`/policy/reset`（`ResetPolicy`，`policy_client_node` 提供）。

`/policy/joint_plan` 只给 dashboard 面板 5 观测用，不参与执行——执行走 `/ur/command`。

## 网页端口

| 服务 | 端口 |
|---|---|
| `status_node` | 8080 |
| `dashboard_node` | 8088 |
| `foxglove_node` bridge | 8765 |

## 仲裁

`/ur/command` 有三个 publisher，`ur_node` 按 `/control/status` 的 `ur_state` +
`active_source` 仲裁，规则见 `ur_node/arbitration.py`：

| UR 状态 | 接受 |
|---|---|
| `UR_STOPPED` | 无 |
| `UR_HOMED` | 仅 `SOURCE_BIAS` |
| `UR_RUNNING` | `SOURCE_BIAS` + 当前 `active_source` |

`/sharpa/command` 同理，`sharpa_node` 按 `sharpa_state` 仲裁（2 态，`SHARPA_STOPPED` 下
只接受 bias）。

## QoS

除相机和触觉外一律默认 `depth=10` reliable。`/camera/set` 和 `/sharpa/deformation` 数据量大，
用 `depth=1` + `best_effort`（丢帧优于堆积；帧完整性由 `recording_node` 自己按时间戳判断）。
