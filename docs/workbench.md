# UR Deploy 工作台

双 UR5 + SharpA 双手的 ROS 2 Humble 工作区。本项目仅维护 UR 工作台；Policy 使用 `interface/` 的远端模型协议，通过
选择 BAAI、UR本机 (4090D) 或额外的 PS (4090，SSH alias `ps`)。UR本机模式直接连接模型，其余机器使用SSH。

## 构建与启动

系统依赖见 [system_deps.md](system_deps.md)，厂商 SDK 放置方式见
[external/README.md](../external/README.md)。手部遥操采用 AE01；安装厂商 Avatar SDK 后，
可运行 `scripts/install_ae01_python.sh` 安装其 Python 依赖。Vive 仍由 Windows
SteamVR/Tracker 与原有 `vive_process.py` 提供输入。

```bash
cd /home/yns/deploy
bash scripts/build.sh --executor sequential
source scripts/setup_env.sh
ros2 launch ur_launch workbench.launch.py \
  recording_root:=/home/yns/deploy/data/episodes
```

打开 <http://127.0.0.1:8088/>。顶部提供「数采」「Policy」「Replay」，共用机器人、
相机、状态和录制节点；切换模式前先停止执行。该 launch 会启动硬件驱动，不能将它
当作离线页面预览。只查看启动参数可运行：

```bash
ros2 launch ur_launch workbench.launch.py --show-args
```

也保留 `teleop.launch.py`（数采页面端口 8080）和 `deploy.launch.py`
（部署页面端口 8088）。各入口只能选择一个运行，避免争用设备与端口。
保护配置统一放在 [config/protection.yaml](../config/protection.yaml)，通过
`protection_config:=/absolute/path.yaml` 可指定替代配置，启动时校验。
标定文件默认 `runtime/config/teleop_bias.json`，可用 `bias_file` 指定。

## 数采

在「数采」页面完成设备连接、标定和复位，设置数据保存目录，点击「开始采集并录制」。
该操作在启动条件通过后同时启用遥操与录制；也支持单独开始／结束录制。
顶部「停止并保存」停止执行并保存当前记录。保存后可以标注 Success/Fail、上传或丢弃。
目录通过录制节点确认后保存到 `runtime/config/ur_sharpa/recording.json`，下次启动恢复；
显式 `recording_root` 参数优先。

三路相机和触觉独立采集，网页低频预览不作为录制或推理图像源。默认视频编码为
`hevc_nvenc`，需要可用的 FFmpeg/NVIDIA NVENC 环境；可用后端及磁盘格式见
[数采文档](../src/nodes/recording_node/README.md)。数据中保留原始时间戳、实测状态、
实际成功下发的双手命令、质量报告与原始消息。`COMPLETE` 表示写盘收尾完成，采集质量
仍须查看 `validation.json` 和 `manifest.quality`。

上传由页面操作触发，目的地通过 `upload_host` / `upload_path` 设置；工作台默认
`BAAI:/share/project/yenaisheng/datasets/postrain/ur-sharpa/raw_data`。

## 远端 Policy

本轮八组黄鸭放盘 DFM 模型已在本机覆盖配置中接入 BAAI / PS (4090)；UR (4090D)
保留软件与机器入口，缺少本轮 checkpoint 的选项禁用。checkpoint、标定、推理 bundle
和逐个测试流程见 [DFM Policy 部署](dfm_policy.md)。

仓库内 [config/model_servers.json](../config/model_servers.json) 预置 BAAI、UR本机和PS三个机器入口，**模型目录为空**。此时页面展示“尚未配置模型目录”，模型列表为空，
不会扫描远端或启动模型。先按 [模型服务配置](../config/README.md) 填写真实路径和远端
task/run catalog，再按「机器 → 任务 → checkpoint」加载模型。任务和checkpoint均可配置自己的任务文本，页面逐项显示最终生效文本；切换机器后重新选择任务。工作台优先读取 `config/model_servers.local.json`，
也可通过 `model_config:=/absolute/path.json` 显式指定。

加载模型时，Dashboard 在所选机器创建独立 tmux 会话，等待健康检查。
`policy_client_node` 随 `/model/status` 切换本机直连或 SSH 隧道目标。DFM 完成两次预热后
开放 `/health`，ROS Python 3.10 客户端通过隔离 Python 3.12 worker 使用原生双向接口。
确认 model、task、run、有效文本和 checkpoint 路径后发送观测并接收动作；Actor 执行
完整动作中的有效窗口和频率。接口 revision 与提案生命周期见 [DFM 部署文档](dfm_policy.md)。
模型启动使用固定配置端口，端口已占用会报错；卸载只清理本实例创建的 tmux 会话。

Policy 来源面板显示请求耗时、累计平均和当前 Chunk 的已下发步数。请求等待中显示
当前 Request 编号和客户端已等待时间，返回后显示最近完成的 Request 编号及客户端
完整请求耗时；DFM 计时包含 IPC、worker 处理、网络传输和推理，原生调用耗时另存于
`native_rpc_latency_s` 诊断字段。累计平均只统计当前加载模型成功且被当前会话接受的
请求，跨停止／继续和 rollout reset 保留；切换、卸载模型时清零。失败、取消和过期
结果不计入平均，模型加载预热也不计入请求次数。

Chunk 进度直接读取 Actor 的已发布命令计数，显示 `已下发 / 实际执行切片长度`，
不按时间估算。DFM 默认为 30 步，其他长度按实际结果展示；若完整 30 步中只执行
索引 `[5,20)`，进度分母为 15，并注明从原始第 6 步开始。设备 ACK 和物理到位不是
这个计数的含义。暂停保留当前真实步数，停止、失败、断线或换模型后清除活动进度。
网页在请求或 Chunk 活动期间每 200 ms 更新状态，空闲及暂停时每 500 ms 更新。

界面提供UR本机模型选择；不提供RTC前缀、execution horizon或sync skip控件。
已有 wire 字段 `execution.rtc` 保留兼容：它表示“剩余 N 步时预取下一次推理”，
不向模型注入上一 chunk 前缀，也不声明 checkpoint 支持某种 RTC 训练方式。
输入/输出维度、EEF 表示、history 窗口由现有
[interface contract](interface_contract.md) 约束。

推理 RPC 在后台单请求线程运行；停止、暂停、换模型或人工接管后，过期结果不会进入
执行队列。Actor 按时间发布命令，成功 ACK 不阻塞每帧调度；当前命令失败、状态过期等
故障会关闭执行。页面预览走同一规划链，但 command 带 `dry_run=true`，驱动不下发。

## Replay

在参考数据区选择 episode，再在 Replay 页加载。默认不指定样本；可通过
`replay_sample_dir` 指定初始路径。频率默认 30 Hz、chunk 默认 40 步，可在页面调整；
支持双臂＋双手或仅双手执行范围。预览与真机执行使用同一 Actor 链路。

Replay 是有限数据流，到末尾完成并关闭执行门，不循环。暂停保留当前进度，停止后保留
样本并复位 cursor；卸载才释放样本。完成后先点击停止清空进度，再从头启动。
加载和启动等待本次命令的回执；样本校验失败直接显示具体原因，旧失败不会覆盖新请求。
数据保存开关只用于 Policy rollout，Replay 不自动录制。

Replay 的文件校验与源时间对齐位于 `replay_node/sample.py`，参数与分段游标位于
`sequencer.py`，ROS 订阅、执行状态机和回执位于 `replay_node.py`。
更新后重新构建并一起重启 Dashboard、Replay 节点，再刷新页面。

## HIL 与执行日志

保留人工接管、交还平滑、失败结束和 rollout 录制核心，默认 `enable_hil:=false`，
顶部 HIL 入口始终隐藏。高级集成可显式启用后通过现有控制 API 使用 HIL；本版没有
本地模型的影子推理或在线 RLT 训练流程。交还策略后从新观测请求远端推理，丢弃接管前
未完成请求的结果。HIL rollout 以及选择保存的 Policy rollout 在正式执行前等待录制
节点确认就绪；预览不录制。Policy 的「保存推理数据」旁显示数采节点确认的 UR 保存目录，
点击「保存目录」可直接打开共用目录选择器。推理机器选择 PS 或 BAAI 时，录制数据仍保存在 UR。
录制或保存期间选择的新目录仅为待应用项，页面继续显示当前生效目录；结束后在「数据保存」
点击「应用目录」，收到数采节点确认后才更新并记忆。
每条 episode 保存在 `<recording_root>/<开始录制的本地时间>/`，推理事件位于
`<episode>/raw/policy_inference.jsonl`，三路相机、实测状态与命令保存在同一 episode 的对应文件中。
标记成功或失败后，整个 episode 移至根目录下的 `success/` 或 `fail/`。
详细字段见 [数采文档](../src/nodes/recording_node/README.md)。

Policy/Replay 的 Actor 执行日志保存到 `runtime/log/execution/<timestamp>/`，包括轨迹、调度、
命令/ACK 和跨节点诊断。默认保留 7 天、100 次、总量 5 GiB，超过任一限制时删除最旧
的已完成记录；可通过 `execution_log_*` launch 参数调整。录制 sample 与执行 trace
用途不同，质量检查和故障诊断均应保留。

## 完整退出 ROS

顶部红色 `killed` 按钮用于退出本次工作台 ROS launch。点击后先关闭执行门、停止
Replay，等待当前 episode 保存完成并卸载本工作台加载的模型，再向 launch 发送
SIGINT。独立后台进程在网页节点退出后继续检查本次 launch 的子进程，必要时依次
使用 SIGTERM、SIGKILL 清理残留，释放相机和机器人连接。此操作只处理本次 launch
的进程树。普通「停止」仍只停止当前执行。

退出期间禁止继续提交操作。录制保存或模型卸载失败、超过 60 秒未完成时，页面
显示错误并保留 ROS，检查后可以再次点击 `killed`。网页服务关闭后停止轮询和相机
预览，提示重新启动 ROS 后刷新页面。退出日志位于
`runtime/log/ros/workbench_shutdown.log`，最后一行确认 launch 及其子进程均已退出。
仅由 `ur_launch` 的 workbench/deploy/teleop launch 启动的 Dashboard 支持此操作；
离线页面或单独启动的节点不会清理其他 ROS。
