# Architecture

`interface` 只定义跨 server 和本体的稳定 dict 边界；坐标变换、IK、硬件协议和
安全控制全部属于具体 embodiment ROS。

```text
ego / wrist cameras ─┐
hand state / tau ────┼─> policy_client buffers ─> policy server
deformation / wrench ┘           │                      │
                                 │ executable slice     │ action + metadata
                                 ▼                      │
                            action_ik ─> action_execute ┘
                                           │
                                           └─ execution_done ─> policy_client
```

## Interface 负责

- server metadata/reset/infer transport 和 dict validator。
- 七个独立固定容量 buffer，以及按 metadata 选择 history/current。
- server action 校验、`execute_start/execute_length` 截断和身份透传。
- synchronous 状态机：同一时间最多一个 inference 和一个 execution。
- `execution_done` 匹配和累计 `executed_steps`。
- SharpA 关节、deformation、wrench 的公共顺序与 shape。

Interface 不定义任何固定的 robot base。EEF 只有 `eef_def`，默认
`absolute`；server model 的 relative 表示必须在 server policy 内部转换为 absolute
后再返回。

## Embodied ROS 负责

- 相机、机器人状态、SharpA 状态与触觉的 ROS producer。
- 本体坐标语义、FK/IK、EEF 到 robot joint 的转换。
- `action_ik`：把 interface 的 absolute EEF slice 转成本体 joint plan。
- `action_execute`：按 frequency 下发 plan、处理 ACK/TTL/safe-stop，并发布一次
  匹配的 `execution_done`。
- PND 的 TCP/RTP/DDS、Adam/SharpA controller；UR 的 RTDE/Wave SDK。
- dashboard、recording、replay 和真机安全开关。

## PND deploy nodes

| Node | 唯一职责 |
| --- | --- |
| `obs_node` | PND robot state/tactile 经 TCP 发到 workstation。 |
| `actor_node` | 接收 workstation command，检查 TTL，写 Adam/SharpA controller。 |
| `robot_states` / `tactile` / `vision` | TCP/RTP 解包为 workstation ROS facts。 |
| `obs_sync` | 对齐 facts 并持续向 policy client 七路 buffer 写入。 |
| `policy_client` | 唯一 server client；metadata、buffers、sync fetch 和 done gate。 |
| `action_ik` | absolute EEF slice 到 Adam joint plan；透传 action 身份。 |
| `action_execute` | 唯一 action 出口；执行 joint plan 并发布 `execution_done`。 |
| `dashboard` | 只观测状态，不参与调度。 |
| `replay` | 代替 policy client 提供离线 plan，不参与 server 通信。 |

Async 当前明确 unsupported。所有 model 使用同一个 workstation launch；模型差异由
metadata、action 和 server policy adapter 表达。
