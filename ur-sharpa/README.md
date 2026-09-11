# UR SharpA

双 UR5 + SharpA 双手的 ROS 2 Humble 工作区，支持 Teleop、数据录制和 Deploy。

## 构建

系统依赖见 [`docs/system_deps.md`](../docs/system_deps.md)，外部 SDK 放置方式见
[`external/README.md`](external/README.md)。

```bash
cd /home/yns/deploy/Deploy-v2/ur-sharpa
./scripts/build.sh
```

## 启动 Teleop

### 1. Windows 启动 Vive

先启动 SteamVR 并确认 Tracker 在线，再运行：

```powershell
cd D:\project\DexCap\STEP1_collect_data_202408updates
python .\vive_process.py
```

### 2. Linux 启动 Teleop

```bash
cd /home/yns/deploy/Deploy-v2/ur-sharpa
source scripts/setup_env.sh
ros2 launch ur_launch teleop.launch.py \
  recording_root:=/media/yns/T9/0825_0826_wyq_ppp
```

Monitor 地址：<http://127.0.0.1:8080/>。
录制根目录由 ROS launch 参数 `recording_root` 指定，默认为
`/media/yns/T9/0826_wyq_ppp`。未定义记录直接保存在该根目录，
已分类记录分别保存在其 `success/` 和 `fail/` 子目录。

Manus 左右手独立控制：一只手无数据时，另一只手仍可使用。Client 和
retarget 只在 Teleop 启动时各启动一次；掉线、数据超时或进程退出时仅在 Monitor
报错，不自动重启。需要重连时请停止后重新启动 Teleop。

Vive 左右 Tracker 和 UR 双臂也独立处理；某侧 Tracker 或 IK 失败时，另一侧
继续工作。胸部 Tracker、全局硬停和控制停止属于双臂共享条件。

## 数据录制

在 Monitor 中开始、停止、丢弃或上传录制。默认远端目录：

```text
BAAI:/share/project/yenaisheng/sharpa-deploy/datasets/postrain/ur-sharpa/raw_data/
```

录制启动不依赖 UR、SharpA 或 Vive 的运行状态。录制包含 UR、SharpA、触觉和
三路相机数据；各数据源保留自身时间戳。SharpA 的 `q_state` 与最近一次成功交给
SDK 的完整双手 `q_cmd` 位于同一反馈帧，并写入同一个 timeline row；停止、来源
切换、单手命令或 SDK 失败时 action 无效。Vive 双臂命令不保存为 action。完整
数据结构见
[`src/nodes/recording_node/README.md`](src/nodes/recording_node/README.md)。

停止保存后，Monitor 会显示 Ego、Left/Right Wrist Camera、Arm State、Hand
State、Hand Action、Wrench、Deform 共 8 项；`25 Hz` 及以上达标，触觉按手指
显示缺失比例。Sample 先保存在根目录，可选择 Success/Fail 后移动到对应子目录；
上传时保持根目录、`success/`、`fail/` 的相对位置。开始下一条记录时摘要清零。
Monitor 的 Drop 会永久删除当前展示的整条 sample，并清空校验摘要。

## 启动 Deploy

```bash
cd /home/yns/deploy/Deploy-v2/ur-sharpa
source scripts/setup_env.sh
ros2 launch ur_launch deploy.launch.py
```

Monitor 地址：<http://127.0.0.1:8088/>。

Dashboard 的执行来源可选 `Policy` 或 `Replay`。Replay 默认加载：

```text
/media/yns/T9/0824_yns_unscrew_cap/success/2026-08-24_16-01-44.683076
```

默认 `execution steps=40`、`frequency=30 Hz`。机器控制区的“预览轨迹”与“启动”
共用 Actor IK 执行链：Actor 按 action 要求的绝对频率持续发布 UR/SharpA command，ACK
仅用于诊断，不会阻塞下一帧或终止 action。若某个设备在一个命令周期内未返回 ACK
（例如 30 Hz 时为 33.33 ms），Actor 记录 warning 后继续执行；迟到 ACK、NACK 和实际延迟
都会写入 trace。预览 command 携带 `dry_run=true`，两个硬件节点只校验、不写控制器。
Dashboard 的输出可视化只订阅
`/actor_ik/trajectory`，显示 IK 后的最终规划，不再直接显示 Policy/Replay 原始 action。
Dashboard 的 Actor latency 仍表示 chunk 边界的额外空档；正常命令周期不计入 latency，
所以连续 30 Hz chunk 不会固定显示为 33.33 ms。

每次实机执行或预览统一写入仓库下的 `ur-sharpa/log/deploy/<timestamp>/`，包含 manifest、
逐 chunk 轨迹、Policy 推理、Actor 排程、UR/SharpA SDK、command/ACK 事件、停止时的
未完成现场和 summary；当前目录会显示在机器控制区。Policy 与 Replay 共用这一目录。
Actor 控制链上的 command、ACK、周期 warning 和 late-ACK 写入 `events.jsonl`；UR、SharpA 和
Policy 的跨节点诊断写入独立的 `node_events.jsonl`。跨节点日志使用 BEST_EFFORT 小队列
和后台写盘，存储变慢时只丢诊断样本，不会阻塞硬件 ACK；丢弃数和写入错误记录在
`summary.json` 的 `external_trace_diagnostics` 中。
UR 与 SharpA 的 `command_result` 保持 RELIABLE QoS，但使用按 topic 精确匹配的 Fast DDS
writer profile，将丢包恢复 heartbeat 从默认 3 秒缩短为 50 ms。配置位于
`config/fastdds_ack_profiles.xml`；launch 明确保持 `RMW_FASTRTPS_USE_QOS_FROM_XML=0`，
不会改变其他 topic 的 publication mode、history memory 或 data sharing。
日志默认保留 7 天、最多 100 次且总量最多 5 GiB，任一限制超出时按时间删除最旧的
已完成日志，当前执行日志不会被删除；三个限制都可通过 launch 参数修改。Replay
使用同一预览链路，整个已加载样本是一次性有限 buffer：到末尾保持 `COMPLETE` 并
关闭执行门，不能循环或自动重启，必须点击“停止”重置进度后才能再次运行。暂停/继续
保留当前 mode、chunk 和 step；统一“停止”只结束当前执行：Policy 保留已加载模型，
Replay 保留已加载样本并把 cursor 重置为 0。卡片上的“卸载模型”和“卸载样本”才会
释放对应资源。

旧版预览日志已经迁移到 `ur-sharpa/log/deploy/legacy_previews/`，也受上述保留策略
约束。自动清理不会处理 `ur-sharpa/log/` 下的 colcon 构建日志；需要时可由操作者
单独清除。
