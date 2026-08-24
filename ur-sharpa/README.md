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
  recording_root:=/media/yns/T9/0824_yns_unscrew_cap
```

Monitor 地址：<http://127.0.0.1:8080/>。
录制根目录由 ROS launch 参数 `recording_root` 指定，默认为
`/media/yns/T9/0824_yns_unscrew_cap`。未定义记录直接保存在该根目录，
已分类记录分别保存在其 `success/` 和 `fail/` 子目录。

Manus 左右手独立控制：一只手无数据时，另一只手仍可使用。Client 和
retarget 只在 Teleop 启动时各启动一次；掉线、数据超时或进程退出时仅在 Monitor
报错，不自动重启。需要重连时请停止后重新启动 Teleop。

Vive 左右 Tracker 和 UR 双臂也独立处理；某侧 Tracker 或 IK 失败时，另一侧
继续工作。胸部 Tracker、全局硬停和控制停止属于双臂共享条件。

## 数据录制

在 Monitor 中开始、停止、丢弃或上传录制。默认远端目录：

```text
BAAI:/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/raw_data/
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

默认 `execution steps=40`、`frequency=30 Hz`，且启用 `Dry run`。先在 Dry
run 下点击“加载”和“开始 / 继续”检查输出预览；需要真实执行时，保持机器停止，
选择 Replay，取消 Dry run 后重新加载，再依次执行“机器复位”→“机器启动”→
“开始 / 继续”。Replay 会先从当前关节状态平滑对齐到样本起点，再逐段发送，且每段
都等待 actor 的 UR 与 SharpA ACK。暂停或停止 Replay 会同时关闭机器执行门；继续前
需要再次启动机器。
