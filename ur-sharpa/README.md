# UR SHARPA Deploy

双 UR5 + SharpA Wave 的 ROS 2 Humble 单机部署。UR 直接连接 policy server，不需要 workstation 中转。当前只支持 deploy 模式，不做遥操。

| 本体 | 默认路径 | 职责 |
| --- | --- | --- |
| UR 主机 | `/home/yns/deploy/sharpa_policy_v3_ws` | 双 UR5、SharpA 双手、ZED、单机 ROS graph、v3 直连 server |
| Server | `/share/project/yenaisheng/dreamzero` | 模型运行环境，同 PND 复用 |

## 1. Setup

```bash
cd /home/yns/deploy
git clone <ur-sharpa repo> sharpa_policy_v3_ws
cd sharpa_policy_v3_ws
colcon build --symlink-install
source install/setup.bash
```

### 外部 SDK 资产

| 资产 | 版本 | 下载源 | 放置目录 |
| --- | --- | --- | --- |
| ur_rtde | 1.6.0 | `pip install ur-rtde==1.6.0`（[PyPI](https://pypi.org/project/ur-rtde/1.6.0/)） | 系统 site-packages |
| SharpA Wave SDK | 4.3.12 | [sharpa-wave-sdk releases](https://github.com/sharpa-robotics/sharpa-wave-sdk/releases)（下载 `SharpaWaveSDK_4_3_12.zip`） | 解压到固定路径，写入 `sdk_root` |
| ZED SDK | 4.2 | [stereolabs.com/developers/release](https://www.stereolabs.com/developers/release) 选 4.2 / CUDA 12 / Ubuntu 22.04 | 系统安装 `/usr/local/zed/` |
| pyzed | 4.2 | `pip install pyzed==4.2`（安装 ZED SDK 后） | 系统 site-packages |

安装完成后修改 `src/sharpa_policy_v3_client/config/hardware_v3.yaml`：

```yaml
sdk_root: /home/fqx/Sharpa/SDK/SharpaWaveSDK_4_3_12  # 改为你的实际解压路径
```

> UR 用 WaveSDK **4.3.12**（zip），PND 用 sharpa-wave-sdk **5.0.2**（apt deb），版本和安装方式都不同，不能互换。

## 2. Deploy

### 启动 Server（BAAI2）

同 PND 部署一致，参考 [pnd-sharpa/README.md](../pnd-sharpa/README.md) 的 Server 章节：

```bash
ssh BAAI2
cd /share/project/yenaisheng/dreamzero
DZ_DEPLOY_RUN_ID="YOUR_RUN_ID" bash scripts/deploy/<model>/launch_checkpoint.sh
```

模型统一监听 `127.0.0.1:5500`。UR 直连该 server，不需要 workstation。

### 启动 UR ROS

```bash
cd /home/yns/deploy/sharpa_policy_v3_ws
source install/setup.bash

ros2 launch sharpa_policy_v3_client hardware_v3.launch.py \
  server_base_url:=http://<baai2-ip>:5500 \
  enable_execution:=false
```

首次执行前设 `enable_execution:=false`，观察一段时间 obs 和 action 数值正常后，再改为 `true` 让 UR 实际执行。

启动参数：

| 参数 | 说明 |
|---|---|
| `server_base_url` | Server HTTP 入口，`http://host:port`，client 会自动派生 `/healthz` `/metadata` `/reset` 和 WebSocket `/infer` |
| `enable_execution` | `false` 时 UR 和 SharpA 只发状态、不执行 action；上真机前一定要先 shadow 一段时间 |
| `ur_confirmation` / `sharpa_confirmation` / `action_confirmation` | 三段执行确认串，用于防止误开启执行 |

### Nodes

UR 单机 graph，一个进程组：

| Node | 职责 |
|---|---|
| `ur_node` | RTDE 双 UR5 驱动；接收关节命令；发布 6D joint、9D EEF、ACK 状态 |
| `sharpa_node` | SharpA 44D 位置命令；发布 44D 状态、5×6 wrench、5×240×240 触觉 |
| `zed_node` | ZED ego 相机；发布 JPEG（腕部相机 producer 暂缺） |
| `state_node` | 聚合 UR/SharpA/ZED/触觉，维护 history buffer，暴露 `BuildObservation` 服务 |
| `policy_node` | 直连 server；调 `/healthz` `/metadata` `/reset` `/infer`；发布策略 action |
| `action_node` | 按 action slice 逐步下发 UR/SharpA；等待 ACK；构造 `executed_steps` 累计 feedback；失败时 safe-stop |

### 查看运行状态

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic list | grep sharpa/v3
ros2 topic echo --once /sharpa/v3/policy/status
ros2 topic echo --once /sharpa/v3/action/status
```

### 停止

在 launch 终端 Ctrl-C 即可。安全起见，执行前先给 UR 上急停开关，遇到异常直接压下比 Ctrl-C 快。

## 3. 已知限制

- 只有 ego 相机，缺左右腕部相机 producer。想跑需要 wrist camera 的模型（如 GCC/PACE 用腕部视角）之前需要接入。
- `sdk_root` 目前是配置文件里的绝对路径，不同机器需手动改。
- 没有 workstation 侧 dashboard，直接看 ROS topic。
- 没有 replay 功能。要复现旧数据需要用 PND workstation 的 replay。

## 4. 常见问题

| 现象 | 处理 |
|---|---|
| `sdk_root` 找不到 | 确认 SharpaWaveSDK 已解压到 `hardware_v3.yaml` 里的路径 |
| UR RTDE 连不上 | 检查 UR 控制柜 IP、Remote Control 模式、防火墙 |
| SharpA 没状态 | 查看 `sharpa_node` 日志；确认 USB 权限和 tactile.json 路径 |
| ZED 无画面 | 确认 ZED SDK 4.2 已装、pyzed 4.2 版本匹配、CUDA 12 可用 |
| Server 连不上 | 确认 `server_base_url` 可达；先 `curl <server_base_url>/healthz` 测通 |
| action 不执行 | 检查 `enable_execution` 是否为 `true`；确认三段 confirmation 已传入 |
| `executed_steps` 不对 | UR 已迁移到累计语义（`execute_start + executed_in_slice`）；不再用 slice-local 计数 |
