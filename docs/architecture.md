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

PND 的异步执行模式保持原有限制。所有 model 使用同一个 workstation launch；模型差异由
metadata、action 和 server policy adapter 表达。


## UR 工作台

`ur_launch/workbench.launch.py` 启动共享硬件、状态、录制节点及 UR 工作台，顶部支持
数采、Policy、Replay。遥操手部来自 AE01，双臂来自 Vive；硬件由状态节点的模式和
执行门仲裁。HIL 核心默认关闭，页面不显示 HIL 顶部入口。

Policy 使用现有 `interface` 连接远端服务。Dashboard 的 `RemoteModelManager`
读取 server profile，管理该实例拥有的远端 tmux 会话；`policy_client_node` 根据
`/model/status` 切换 SSH 隧道、获取 metadata 并维护观测 buffer。三路推理图像来自
原始共享内存，经 JPEG 编码进入既有协议，不经过低频网页预览。

远端请求在后台单工作线程中执行，ROS 控制回调保持可响应。当前请求的 generation
失效时丢弃返回结果。协议中的 `execution.rtc` 只决定提前 fetch 时机，不提供模型
前缀条件或本地 openpi runtime。最多一个请求在途，可以与上一 chunk 的执行重叠。

UR 的 `actor_ik_node` 合并了上图中的规划和调度职责，Policy 与 Replay 共用执行链。
驱动消费 `/arm/command` 和 `/hand/command`，Actor 接收命令结果并输出 trace；成功
ACK 不作为每步串行等待门。保护、HIL 交还和录制握手也在 Actor 侧接入。
录制器保存原始消息、相机视频、触觉及结构化时间轴，执行与采集质量分别报告。

详细启动方式、默认目录、日志和配置见 [UR README](../ur-sharpa/README.md)。
