# ur_msgs

UR5 + SharpA 本体的 ROS 2 接口包。分三层：hardware facts / control 状态机 /
deploy 执行链。dict 边界定义见 [interface_contract.md](../../../docs/interface_contract.md)，
关节顺序与 shape 的唯一来源是 `interface/sharpa_interface/sharpa.py`。

## 冻结 shape

| 量 | shape | 单位 | 顺序 |
| --- | --- | --- | --- |
| hand joint / tau | 每手 22，双手 44 | radian / N·m | 左手在前；手内 thumb→index→middle→ring→pinky |
| UR arm joint | 每臂 6 | radian | shoulder_pan→shoulder_lift→elbow→wrist_1→wrist_2→wrist_3 |
| EEF | 9 = `xyz` + `rot6d` | meter | `[x,y,z,r00,r10,r20,r01,r11,r21]`，旋转矩阵前两列 |

### rot6d 约定

`UrStateFrame` 与 `ExecutableAction` 全部使用 interface 已冻结的
canonical column-Rot6D：`[x,y,z,r00,r10,r20,r01,r11,r21]`。Legacy row-Rot6D
只允许留在显式兼容 adapter 内部，不能出现在 ROS topic 或数据集边界。
| wrench | 每手 `(5,6)`，展平 30 | N / N·m | 行 pinky→thumb；列 Fx,Fy,Fz,Tx,Ty,Tz |
| deformation | 每手 `uint8 (5,240,240)`，展平 288000 | 无量纲 counts | 面 pinky→thumb；面内行主序 |
| camera | JPEG bytes | — | ego / left_wrist / right_wrist 三路各自命名、各自时间戳 |

触觉的手指顺序 `pinky→thumb` 是 `sharpa.py` 的 `TACTILE_FINGERS`，与 hand joint 的
`thumb→pinky` **相反**，不要用关节下标索引触觉数组。

## Message

| 类型 | 一句话职责 |
| --- | --- |
| `CameraSet` | 三路命名相机 JPEG，每路独立采样时间戳；不提供拼接图字段。 |
| `ArmCommand` | 双臂 6D joint 目标，`source` 区分 `bias`/`vive`/`actor`；Actor 可携带 `dry_run`。 |
| `UrStateFrame` | 双臂 6D joint + 9D EEF（FK 结果）+ ACK/有效性状态。 |
| `HandCommand` | 44D 手部 joint 指令，含左右手 enable 与 Actor `dry_run`。 |
| `HandStateFrame` | 44D 手部 `q`，附 `dq`/`tau`，以及最后收到且仍新鲜的 Manus Teleop `q_cmd/valid/timestamp/source`。 |
| `TauFrame` | 每手 22D 力矩，sensor buffer 路径，按手拆分。 |
| `WrenchFrame` | 每手 `(5,6)` 指尖力/力矩，行主序展平。 |
| `DeformationFrame` | 每手 `uint8 (5,240,240)` 指尖形变图，行主序展平。 |
| `ControlStatus` | UR(5 态)/SharpA(5 态)/Record(2 态) 三个状态机 + `gate_reason`。 |
| `HardwareCommandResult` | 单步硬件指令的 ACK / 拒绝原因。 |
| `ExecutableAction` | 截断后的 action slice，含完整 identity 与执行参数透传。 |
| `ActorTrajectory` | Actor IK 后的统一 chunk 轨迹；Dashboard 与预览日志的唯一预测来源。 |
| `ExecutionDone` | slice 的 fetch/done 双边沿；只有 `execute_done` 是终态，`executed_steps` 为**累计**语义。 |

## 三个状态机

| 机 | 取值 |
| --- | --- |
| UR | `UR_STOPPED=0` / `UR_HOMED=1` / `UR_RUNNING=2` / `UR_HOMING=3` / `UR_STARTING=4` |
| SharpA | `SHARPA_STOPPED=0` / `SHARPA_RUNNING=1` / `SHARPA_HOMING=2` / `SHARPA_HOMED=3` / `SHARPA_STARTING=4` |
| Record | `RECORD_IDLE=0` / `RECORD_RECORDING=1` |

`gate_reason` 是人类可读字符串，说明当前为什么不能进入下一个状态；空串表示无阻塞。
`active_source` 指出当前允许驱动手臂的 `ArmCommand.SOURCE_*`，为
`SOURCE_UNKNOWN` 时所有手臂指令都会被拒绝。

## executed_steps 累计语义

```text
executed_steps = execute_start + executed_in_slice
```

这是 `interface/sharpa_interface/server/executed_steps.py` 冻结的定义，**不是**
slice 内计数。slice-local 计数在每次 replan 时归零，会让 policy client 的同步状态机
死锁。全部执行成功时 `executed_steps == execute_start + execute_length`。
构造该值请调用 `make_progress()`，不要手工相加。

`action_ik` 只允许替换或补充 joint plan，`session_id`/`request_id`/`action_id`/
`revision`/`timestamp_ns` 和全部 execution 字段必须原样透传。

## 构建

```bash
cd ur-sharpa
./scripts/build.sh --packages-select ur_msgs
```

`scripts/setup_env.sh` 会清除 Conda/ROS 1 路径并固定系统 Python 3.10，避免
`ament_package`/`rosidl_adapter` 被错误解释器覆盖。
