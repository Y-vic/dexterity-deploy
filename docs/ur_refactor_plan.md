# UR-SharpA ROS 重构计划

把 `/home/fqx/ur_sharpa` 的单体脚本（teleop / 数采 / replay）重构成 ROS 图，与 PND 保持同一套
架构逻辑，同时让 UR 侧复用 `interface/` 而不是自己维护平行副本。

## 约束

1. `/home/fqx/ur_sharpa` **只读**，只拷贝不修改。
2. `/home/yns/deploy/sharpa_policy_v3_ws` 只读参考。它是已部署跑过的 v3 代码，是行为基准的
   权威来源；与 `Deploy/ur-sharpa/src/nodes/sharpa_policy_v3_client` 是同一份代码的两个分叉副本。
3. `interface/` 是冻结 ABI（`sharpa.py` 明确要求改动需 schema bump + 迁移说明 + 各本体签字）。
   UR 侧 import，不再抄。
4. v1 **关闭全部安全保护**，但做成 launch 参数（默认 `false`），单开关可完整恢复。
5. 硬件是 **UR5**（非 UR5e）。DH 不同，IK 和 URDF 都要对上。
6. 落地在 `Deploy/ur-sharpa/src`，运行节点统一放在 `src/nodes/`。

## 目标包布局

```
ur-sharpa/src/
├── ur_msgs/              # 消息与服务定义
├── ur_common/            # 共享 FK/IK、几何、关节序、安全开关
└── nodes/                # 所有 ROS 运行节点与 launch 包
    ├── ur_node/          # 双 UR5 RTDE 唯一出口
    ├── sharpa_node/      # SharpA Wave 4.3.12
    ├── camera_node/      # ZED ego + 双腕 RealSense
    ├── vive_node/        # Vive tracker → 臂 retarget（teleop）
    ├── manus_node/       # Manus 手套 → 手 44D retarget（teleop）
    ├── bias_node/        # UR + SharpA 初始位 / 复位
    ├── status_node/      # 独立网页 + 7 状态闸门
    ├── robot_states_node/ # 状态聚合 + FK → wrist EEF（deploy）
    ├── policy_client_node/ # 唯一 server client（deploy）
    ├── actor_ik_node/    # IK + 执行 + execution_done（deploy）
    └── ur_launch/        # 统一 teleop/deploy/replay 启动入口
├── recording_node/       # 录制
├── replay_node/          # data sample → action chunk
├── dashboard_node/       # 5 面板网页
├── foxglove_node/        # URDF 可视化
└── ur_launch/            # mode:=teleop|deploy|replay
assets/ur_sharpa_description/   # UR5 + 双 HA4 复合 URDF
```

## 两条链

**Teleop**

```
vive_node ──arm joint──┐
manus_node ──hand 44D──┼─> [status 闸门] ─> ur_node / sharpa_node ─> 硬件
bias_node ──init/reset─┘
camera_node ──CameraSet──> recording_node / dashboard / foxglove
```

**Deploy**

```
camera_node + ur_node + sharpa_node
        │
        ▼
robot_states_node ──(FK → wrist 9D)──> policy_client_node ──> server
                                            │           ▲
                                     slice ─┤           └── action + metadata
                                            ▼
                          actor_ik_node ──(IK → joint plan)──> ur_node / sharpa_node
                                            └── execution_done ──> policy_client_node
```

`camera_node` / `ur_node` / `sharpa_node` / `status_node` / `recording_node` 两条链共用。

## 节点职责

| Node | 职责 | 拷贝来源（只读） |
|---|---|---|
| `camera_node` | 独占 ZED + 双腕 RealSense；一条 `CameraSet` 带三个具名 JPEG + 各自采样时间戳，不合图 | `image_server/local_camera.py` |
| `vive_node` | Vive tracker（ZMQ 传输）→ 腕位姿 → 调 `ur_common` IK → 发臂关节目标 | `ks3util_output/mocap/mocap_wrapper.py` |
| `ur_node` | 双 UR5 RTDE 唯一出口；三路命令按 status 仲裁；发 6D joint + 9D EEF + ACK | v3 `ur_node.py` + `robot_control/ur5.py` |
| `sharpa_node` | 44D 命令；发 44D 状态 + tau + 5×6 wrench + 5×240×240 触觉 | v3 `sharpa_node.py` |
| `manus_node` | Manus 采集 + 手 retarget 子进程 | `hand_retargeting_optimizer`（外部路径） |
| `bias_node` | UR 初始位 + SharpA 归零；启动和复位都走它 | teleop `_calibrate_initial_position` + home mats |
| `status_node` | 独立 HTTP 网页 + 7 状态闸门 + `ur_safe_jog` 并入 | `collect_ui.py` 状态条 |
| `robot_states_node` | 聚合 UR/SharpA/相机；自己做 FK 产出 wrist EEF | 新写 + `ur_common` |
| `policy_client_node` | 唯一 server client；七路 buffer、sync 状态机、slice 截断、done gate | v3 `policy_client_node.py` 去重后 |
| `actor_ik_node` | EEF slice → IK → joint plan → 下发 → 发 `execution_done` | v3 `action_node.py` + `hardware_execution.py` |
| `recording_node` | 录制 `episode_XXXX/data.json + colors/ + tactiles*` | `utils/episode_writer.py` |
| `replay_node` | 读 data sample → 发 action chunk | `replay_ur5_sharpa_hand.py` |
| `dashboard_node` | 5 面板网页（独立于 status 网页） | `collect_ui.py` + `pybullet_visualizer.py` |
| `foxglove_node` | URDF + joint_state 合并 | PND `foxglove_node` |
| `ur_common` | **共享 FK/IK**（teleop 和 deploy 同一后端）、几何、安全开关 | `ik_solver.py` + `ikfastpy` + `coord_transform.py` + `wire_geometry.py` + `sharpa_fk.py` |

## status_node 状态机（7 状态）

- **UR 3 态**：`stopped` → `homed` → `running`
  - 「启动 UR」= `stopped→homed`（触发 bias）
  - 「复位 UR」= 任意态 → bias → `homed`
- **SharpA 2 态**：`stopped` → `running`（启动含 bias 归零）
- **Record 2 态**：`idle` → `recording`

闸门规则（`ur_node/arbitration.py`，纯函数、无硬件可测）：

| UR 状态 | 接受的命令源 |
|---|---|
| `stopped` | 无 |
| `homed` | 仅 `bias` |
| `running` | `bias` + 当前 `active_source`（vive 或 actor） |

`bias` 在 `running` 下保留权限，保证复位在 episode 中途也能落地。拒绝必须带原因字符串——静默
丢弃会让「卡住的臂」和「断开的 producer」无法区分。

`recording` 要求 UR + SharpA 均 `running` 且三路相机都有新帧，否则网页按钮置灰。

`ur_safe_jog` 的三段 confirmation 并入本节点：保留「先 dry-run 打印目标、再确认执行」的流程，
确认动作从命令行 token 变成网页按钮，避免两套确认互相绕过。

## 硬约束

| # | 约束 | 违反后果 |
|---|---|---|
| 1 | `actor_ik_node` 原样透传 identity，发一次匹配 `execution_done`，`executed_steps` 用累计语义 | policy_client done gate 卡死 |
| 2 | FK（`robot_states_node`）和 IK（`actor_ik_node`、`vive_node`）同源 `ur_common` | 观测/执行标定漂移 |
| 3 | `ur_node` 是 UR 唯一出口，`sharpa_node` 是 SharpA 唯一出口（Wave SDK 独占串口） | 串口争用 / 双写 |
| 4 | server 边界 EEF 必须 `absolute`，`xyz+rot6d` shape `(9,)`，**列主序** | interface 校验拒绝 / 姿态静默错 |
| 5 | `recording_node` 输出逐字段对齐 P0 基线 `data.json` | 历史训练数据失效 |
| 6 | 任一模态缺失丢弃整条 episode | 残缺帧污染数据集 |

## 阶段与依赖

```
P0 ──────────────> P5 门槛
P1 ─> P2 ─> P3 ─> P4 ─> P5
      P1 ─> P6 ─> P9
      P2 ─> P7 ─> P8 ─> P9 ─> P10
```

P1 / P6 / P9 不需真机，可与真机阶段并行。

| 阶段 | 内容 | 门槛 | 真机 |
|---|---|---|---|
| P0 | 基线套件 `ur-sharpa/baseline/` | 一条完整 episode + deploy log | 是 |
| P1 | `ur_msgs` + `ur_common` + interface 去重 | FK→IK→FK 闭环；`contract_tests` 过 | 否 |
| P2 | `camera_node` + `ur_node` + `sharpa_node` + `ur_bringup` | dry-run → 画圆空载 → 30Hz 三路不掉帧 | 是 |
| P3 | `vive_node` + `bias_node` + `status_node` | 安全全关，**空载 + 手按急停** | 是 |
| P4 | `manus_node` | 44D 延迟与基线同量级 | 是 |
| P5 | `recording_node` | 与 `baseline/episode_ref/` 逐字段 diff 过 | 是 |
| P6 | `ur_sharpa_description` + `foxglove_node` | URDF FK 与 `ur_common` 一致 | 否 |
| P7 | `robot_states_node` + `policy_client_node` | mock 四例过 → 真机 shadow | 是 |
| P8 | `actor_ik_node` | 契约测试过 → 空载执行 → 断线恢复 | 是 |
| P9 | `dashboard_node` 5 面板 | — | 否 |
| P10 | `ur_launch` + `replay_node`，删旧副本 | — | 是 |

### dashboard 5 面板

| 面板 | 内容 |
|---|---|
| 1 | 输入视频（ego + 双腕） |
| 2 | 真机当前状态 → 仿真渲染 |
| 3 | retarget 后观测（送进模型的 wrist 9D + 44D） |
| 4 | 模型输出 action chunk + `execution` 参数 |
| 5 | 模型动作 → IK → joint → 仿真渲染 |

2 和 5 并排即「现在在哪 vs 模型要它去哪」。离屏 PyBullet 渲染，
`robot_control/pybullet_visualizer.py` 已支持双 UR5 + 双 SharpA。

## 已完成

| 产出 | 位置 | 状态 |
|---|---|---|
| P0 基线套件 | `ur-sharpa/baseline/` | 等真机跑 |
| `ur_msgs` | `ur-sharpa/src/ur_msgs/` | 13 msg + 2 srv，colcon 过 |
| `ur_common` | `ur-sharpa/src/ur_common/` | 89 测试全过（py3.10 + pinocchio） |
| drift 报告 | `docs/ur_interface_drift.md` | 41 项，12 项契约违反 |
| P7/P8 验收清单 | `docs/deploy_acceptance.md` | 9 组根因，23 条测试 |
| buffer 竞争复现 | `contract_tests/test_buffer_concurrency.py` | 3 条预期失败 |
| buffer 加锁方案 | `docs/buffer_locking_proposal.md` | 待 PND 签字 |
| rot6d 结论 | `docs/rot6d_convention.md` | 列主序，两侧一致，非阻塞 |
| rotvec 缺陷 | `docs/rotvec_near_pi_defect.md` | 已记录，P8 修 |

## 未决

| # | 项目 | 需要 |
|---|---|---|
| 1 | interface buffer 加锁 | PND owner 签字 |
| 2 | rotvec 修正时机 | 旧链路是否继续做实验 |
| 3 | `executed_steps` + slice 归属权 | 必须与截断一起原子改（P8） |

第 3 条是 drift 报告 C1 + C2：UR 现在是「policy_client 发完整 chunk，硬件侧自己切片自己数
步数」，契约要求「policy_client 切片后再发，执行侧只管执行区间并回累计步数」。只改一半会得到
既不符合旧模型也不符合契约的中间态。

`execution.py:199` 的 `executed_steps == execute_stop` 比较是支点——正因为有它，slice 归属权
未修时 deploy 链是**启动即失败**而非静默执行错区间。放宽它等于把唯一防线交给测试套件。
