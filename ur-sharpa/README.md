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
python .\vive_process_send.py
```

### 2. Linux 启动 Teleop

```bash
cd /home/yns/deploy/Deploy-v2/ur-sharpa
source scripts/setup_env.sh
ros2 launch ur_launch teleop.launch.py
```

Monitor 地址：<http://127.0.0.1:8080/>。

Manus 左右手独立控制：一只手无数据时，另一只手仍可使用。单手持续无数据
1 秒后自动重启 Manus 链路；连续 3 次未恢复后改为每 30 秒重启一次，双手恢复
后重启次数清零。Monitor 会显示“正在重启”、“恢复检查”或“恢复失败 3/3”。

Vive 左右 Tracker 和 UR 双臂也独立处理；某侧 Tracker 或 IK 失败时，另一侧
继续工作。胸部 Tracker、全局硬停和控制停止属于双臂共享条件。

## 数据录制

在 Monitor 中开始、停止、丢弃或上传录制。默认远端目录：

```text
BAAI:/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/raw_data/
```

录制启动不依赖 UR、SharpA 或 Vive 的运行状态。录制包含 UR、SharpA、触觉和
三路相机数据；各数据源使用自身时间戳异步对齐。`actions` 保存新鲜的 Manus
双手 Teleop 命令（左右手各 22 维），即使 SharpA 暂停也会保存；Vive 双臂命令
不保存为 action。完整数据结构见
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
