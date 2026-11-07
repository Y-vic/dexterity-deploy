# DFM 黄鸭放盘部署

页面按「推理机器 → 任务 → checkpoint」选择模型。当前任务为 `ur_duck_placement`
（黄鸭放盘），包含 2026-11-06 八组 2 GPU / 40000 步实验的最终训练产物：

| 页面名称 | 训练 run | checkpoint |
| --- | --- | --- |
| 101 real | `flow-duck-real101-2gpu-40k-20261106` | `checkpoint-40000` |
| 101 real + 100 sim | `flow-duck-sim100-real101-2gpu-40k-20261106` | `checkpoint-40000` |
| 101 real + 200 sim | `flow-duck-sim200-real101-2gpu-40k-20261106` | `checkpoint-40000` |
| 101 real + 1k sim | `flow-duck-sim1k-real101-2gpu-40k-20261106` | `checkpoint-40000` |
| 101 real + 2k sim | `flow-duck-sim2k-real101-2gpu-40k-20261106` | `checkpoint-40000` |
| 101 real + 4k sim | `flow-duck-sim4k-real101-2gpu-40k-20261106` | `checkpoint-40000` |
| 101 real + 8k sim | `flow-duck-sim8k-real101-2gpu-40k-20261106` | `checkpoint-40000` |
| 8k sim · 纯仿真 | `flow-duck-sim8k-2gpu-40k-20261106` | `checkpoint-40000` |

八组配置、最终 checkpoint 和训练记录均为 40000 步。清单位于
[task.json](../config/model_tasks/ur_duck_placement/task.json)。PS 和 BAAI 上上一轮四个 real101
`checkpoint-45000` 的权重保留，当前任务清单显示本轮八组。

含真实数据的七组部署显式使用真实101条数据的任务文本：

> Pick up the yellow ducks and place them on the pink plate.

纯仿真 8k checkpoint 显式使用其训练数据的文本：

> Pick up all the yellow ducks on the table and place them in the plate.

混合训练数据包含上述两种文本，含真实数据的 checkpoint 在真机部署时选择真实文本。
纯仿真组没有真实训练样本，不能继承任务默认的真实文本。任务提供默认 `prompt`，各
checkpoint 通过自身 `prompt` 指定实际生效文本；页面逐项显示，加载身份也核对这一文本。
切换任务或机器后不会继承别的任务文本。对比实验时保持相同物理起始条件、执行设置和
复位过程，并保留每个 checkpoint 上述训练文本。

## 推理机器

`config/model_servers.local.json` 将三台机器映射到各自模型目录：

| 页面机器 | 连接 | DFM 根目录 |
| --- | --- | --- |
| BAAI | SSH alias `BAAI` | `/share/project/yenaisheng/DFM` |
| UR (4090D) | 工作台本机直连 | `/home/yns/deploy/runtime/dfm-source` |
| PS (4090) | SSH alias `ps` | `/home/ps/DFM` |

本轮 checkpoint 下载到 PS 的
`/home/ps/DFM/logs/<run>/checkpoints/checkpoint-40000`；BAAI 保留完整训练产物。
PS 只有下载完成并出现正式 checkpoint 目录的选项可加载；下载暂存目录不参与部署。
测试时选择 PS (4090) 已完成的 checkpoint，或选择 BAAI。UR 本机的 checkpoint 已移除，本轮不提供 UR (4090D)
权重；它仍保留模型软件和机器配置，清单会将缺少 checkpoint 的选项禁用。部署 bundle
位于各 DFM 根的 `logs/deploy-runtime`。UR 的隔离客户端使用
`/home/yns/deploy/runtime/venvs/dfm-serving/bin/python`。环境和 task catalog 由 bundle
提供，模型在所选机器上运行；UR 本机选择不创建 SSH 隧道。

只列出 checkpoint 目录和部署依赖完整的模型。下载与源代码校验记录用于离线验收，
不作为模型加载时的 hash 门禁。准备 bundle 只复制模型软件、清单和标定，不启动硬件。
已经运行的服务需由操作者按当前工作台流程卸载和重新加载后使用更新。

在 UR 准备本机 bundle：

```bash
python3 scripts/prepare_dfm_runtime.py \
  --dfm-root /home/yns/deploy/runtime/dfm-source \
  --python /home/yns/deploy/runtime/venvs/dfm-serving/bin/python \
  --calibration /home/yns/deploy/runtime/config/dfm_calibration.json \
  --data-contract /home/yns/deploy/runtime/config/real101_data_contract.json
```

准备 BAAI 或 PS 时增加 `--host <SSH alias>`，并将 DFM 根目录、Python 路径改为该机器
上的实际路径。`--calibration` 和 `--data-contract` 是 UR 端当前测量结果与真实101条
训练数据的物理契约，随 bundle 复制；八个 run 的 `required_files` 同时检查两份 JSON、
原生环境转换、单模型 worker 和服务入口。

## 接口与执行

`dfm_flow` launcher 使用 DFM 原生 `InferenceServer`。UR 的 ROS Python 3.10 进程使用
轻量 `NativePolicyClient`，通过有长度上限的 MessagePack IPC 与隔离 Python 3.12 worker
交换数据；worker 调用原生 DFM `InferenceClient`，以 WebSocket `/infer` 连接所选机器的
`InferenceServer`。本机直连和 SSH 隧道共用这一接口。模型加载、预热、reset、预测和关闭
均在推理机器的同一个模型 worker 线程执行，保留 CUDA 与编译器上下文。

模型加载后先完成两次预热，覆盖首次与缓存推理，再恢复 seed 和历史，随后绑定 `/health`。
客户端核对 model、task、run、有效文本和完整 checkpoint 路径，确认协商后的观测/动作
能力；页面只有在确认后显示就绪。状态 metadata 保留 `native_interface`、
`interface_revision` 和 `pending_interface_update`。

卸载时，在 owned tmux 会话退出后继续等待旧模型释放服务端口，期间保持「停止中」并
拒绝重载。等待超过 12 秒会明确报告端口未释放，避免首次重载与正在退出的服务冲突。

`/policy/interface_update` 接收完整提案，消息字段精确为
`{model_generation, session_id, interface_revision, interface}`。模型代次、rollout 会话和
当前接口 revision 必须匹配，提案也必须满足当前模型视图、生成频率和 horizon。收到
提案回执时当前接口不变；下一次返回的完整动作仍使用采集时接口与标定，验证后才提交
新声明、递增 revision，并供下一份观测使用。完整动作保留 30×62；`execute_start` 和
`execute_length` 指定 Actor 的执行窗口。

未提交的连接提案在断线或显式 rollout reset 时取消。控制变化使提案回执过期时，客户端
先取消其未提交提案并重置 rollout，再恢复推理。已提交声明跨 reset 保留；由模型侧提出
的声明保持模型侧归属。推理失败或 IPC 断开后通过显式 reset 恢复，并重新核对原有模型
身份；超时或失败的推理不会自动重试。

checkpoint 保存了 Ego、左右腕三路 RGB、62D 状态与动作、30 Hz / 30 步。
物理值使用米、column Rot6D 和弧度。DFM processor 负责图像预算和归一化，Policy 返回
物理动作；部署边界使用与真实101条训练数据对应的标定处理手根坐标和手关节次序。
相机或机械安装变化后需要重新核对标定。

Policy 的有效执行切片在 Actor 入队前统一处理手部目标：按当前固件 44 个关节的逐项
上下限投影，单关节修正超过 `hand.max_projection_rad`、非有限值或错误维度会拒绝整段，
不会发布轨迹或双臂、手部命令。完整模型预测保留原始物理值；Actor 轨迹保存投影后的
执行目标，`actor_hand_plan_projected` 事件另存原始有效切片、修正数量和最大修正量。

投影只修正允许范围，不能代替启动复位。Actor 在入队前按手部配置校验首步与实测
姿态、预取切片与上一计划末步及相邻规划目标差。每步发布前，Actor 保留原有
`actor.hand_step_rad` 实测状态限幅，再对最终实际命令检查固件范围、启动、步差和跟踪
保护；0.4 rad 是当前每步限幅值，不是新增的原始规划目标拒绝阈值。轨迹保存投影后的
期望目标，`command_published.hand_joint` 保存最终限幅命令，二者沿用同一动作身份。

Policy 手部命令携带真实 rollout session ID 和 `strict_targets=true`，出口在手部 SDK
下发前核对设备序列号/型号/固件版本与严格目标，不再修改该最终命令。Actor 的手部
检查失败会在发布当步的双臂或手部命令前拒绝。页面显示启动姿态偏差时应复位并重新
观测，不能依靠预览或范围投影绕过。Replay 和 HIL 保留原有执行行为。

运行时按协商后的有效执行切片执行。页面请求耗时为客户端完整请求耗时，包含 IPC、
worker 处理、网络传输和服务端推理；`client_request_latency_s` 保存同一计时，
`native_rpc_latency_s` 单独保存 worker 内原生调用及边界转换耗时。累计平均仅统计当前
加载模型成功且被当前会话接受的请求，失败、取消或过期结果不计入平均，预热不计入
用户请求。Chunk 进度来自 Actor 实际已发布命令计数。切换机器或模型会使旧连接与
未完成请求失效，旧动作不能进入新会话队列。

原相机 JPEG 现在通过现有 v1 图像 envelope 原样传给 DFM，服务端才解码 RGB，避免
每次先解码再压成 PNG；没有改变相机分辨率、JPEG 质量或模型像素。RPC 完成后使用
ROS guard condition 唤醒 executor，仍在原回调内验证会话与控制状态，再发布动作，
减少原 100 ms 定时轮询造成的完成等待。

`model_inference_completed` 执行事件和 `/policy/inference` 的完整结果都保存分阶段
`diagnostics`：`native_ipc_*` 表示本地 IPC，`native_observation_prepare_s` 表示观测
边界转换，`client_message_*` / `client_response_*` 表示 WebSocket 编码、发送、等待与
解码，`model_seconds` 为模型已有诊断，`native_action_convert_s` /
`native_action_parse_s` 为动作转换与解析。计时是嵌套区间，不能把模型时间再加到总
request 时间中。服务端等待还包含图像解码和模型前处理，不能全部当成网络延迟。

Policy「保存推理数据」旁可打开「保存目录」，页面显示 recorder 确认后的 UR 实际目录。
更改目录沿用共享目录选择器；录制中的草稿不改变当前 rollout 的保存位置。正式录制
并保存后，推理数据位于 `<episode>/raw/policy_inference.jsonl`，执行日志另存于
`runtime/log/execution/<timestamp>/`。只推理或预览不会生成正式 rollout 数据文件。

## 网页测试流程

1. 按[工作台文档](workbench.md)启动，打开 `http://127.0.0.1:8088/`。
2. 切换到 Policy，选择 PS (4090) 使用本轮下载的 checkpoint，或选择 BAAI 使用训练源产物。
   本轮 UR (4090D) 没有本机权重，缺少 checkpoint 的模型不可加载。
3. 打开「选择任务 / checkpoint」，选择「黄鸭放盘」，核对 checkpoint 和显示的任务文本。
4. 点击「加载 checkpoint」，等待模型加载、接口与身份确认。加载模型本身不开始机器人执行。
5. 完成相机、状态、标定和复位条件后先预览；预览使用原规划链并设置 `dry_run=true`。
6. 确认后由操作者开始正式执行。比较模型时使用相同起始条件、文本和执行设置。
7. 停止执行、卸载模型，再选择另一 checkpoint。执行 trace 位于 `runtime/log/execution`。

离线 mock、构建和模型推理检查用于确认软件、传输和数值接口，不能代替真机任务效果评估。
