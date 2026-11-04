# 数据采集、格式与诊断

启动与页面操作见 [UR 工作台 README](../../../README.md)。HIL 核心默认关闭，顶部入口隐藏。
本文维护采集格式、NVENC、原始消息完整性和性能诊断；旧本地模型的 RTC/RLT、影子推理
不属于当前远端部署流程。

工作台「数采 → 数据保存」提供「开始采集并录制」：复位完成后，一次请求启用遥操并开始录制。控制层先检查机器启动条件、三路相机及数采节点状态；检查未通过时两项均不启动。原有「启用遥操」「开始录制」「结束录制」仍可单独使用；顶部「停止并保存」同时停止机器并保存当前记录。

部署网页的 D435/D435i Demo 录制是独立服务，保存原始 ROS bag、RGB 和十指信息叠加视频，新相机仅启用 RGB。启动参数、文件格式和指标定义见 [DEMO.md](DEMO.md)。

HIL rollout 使用本数采节点，与 D435 Demo 录制独立。部署时开启 `auto_record_rollout` 等待页面选择：HIL 自动采集，Policy 默认不采集、可选择保存，Replay 不触发采集。订阅 `/hil/rollout`，正式执行前创建 sample 并确认就绪，暂停／停止／执行故障后自动保存，预览不录制。默认目录为 `/home/yns/data/test`，可用 `recording_root` 启动参数指定，或在工作台“数据保存”输入／粘贴路径后应用，也可点击“选择目录”浏览主机文件系统并新建文件夹；采集启动、录制、保存期间可先选择，结束后再应用；页面修改由数采节点确认后原子保存到 `~/.config/ur_sharpa/recording.json`（设置 `XDG_CONFIG_HOME` 时使用该目录）；重启默认恢复上次成功应用的目录，显式 `recording_root` 启动参数优先。未应用的输入及被拒绝的请求不会覆盖记忆，记忆写入失败会拒绝目录修改。人工接管和策略交还不会拆分 sample。该模式忽略手动 start/stop/discard 命令，由执行生命周期统一管理；HIL 的失败按钮在保存完成后按 rollout ID 将对应数据移到 `fail/`，保存后仍支持原有分类操作。

rollout 复用以下所有数采字段和文件，在 `data[].metadata` 增加 `is_hil`，并在 `timeline/clock.npz` 增加 `is_hil[T] bool`。HIL 标记来自同一成功 SDK 命令的 `HandStateFrame.is_hil`，与 `q_cmd` 同步；无效行标 false，需要结合原有 `q_cmd_valid` 判断。rollout 接受 Actor（source=3）的完整双手动作，普通示教仍只接受 AE01（source=2）。`hand_action_source/q_cmd_source` 字段继续记录真实来源，其余字段、维度、视频编码和触觉选项沿用普通数采。

### 执行关联与终止信息

Actor 在录制握手中提供本机 trace 目录。保存时把可用的 `manifest.json`、
`events.jsonl`、`node_events.jsonl`、`trajectories.jsonl`、summary/final_state
复制到 episode 的 `actor/`，纳入 checksums。`actor/archive.json` 记录源 trace 状态、
复制字节数和缺失回执；执行 trace 仍 active 时只复制保存时快照。

录制器保留执行审计派生器，但远端 Actor 只提供自身实际记录的命令、轨迹及回执。
不要假设旧本地 runtime 的每步决策观测、冻结前缀、RLT reference 或分组替换上下文
一定存在。缺失字段和回执应按缺失处理，不得推断为成功，也不能将“推理完成”视为
“动作已执行”。跨节点诊断丢弃和写盘错误由日志及质量检查报告。

`termination.json` 保存 `success / failure / truncated / discarded`、终止时刻、
可用的动作 ID、标注来源和标注时刻。普通停止默认 truncated，不自动推断失败。
事后分类保留原终止信息。HIL 的 `is_hil` 标记与成功下发的手部命令关联，需同时查看
`q_cmd_valid`；暂停、停止和故障会结束当前 rollout，人工接管／交还本身不拆分 sample。

### 远端模型推理日志

HIL 自动采集和 Policy 的“保存推理数据”保存 `raw/policy_inference.jsonl`。
`policy_client_node` 发布 `/policy/inference` String，schema 为
`ur_sharpa_policy_inference.v1`。每次请求有一条 `requested`，以及一条 `completed`、
`failed` 或 `cancelled` 终态；`request_id` 是该日志请求的 UUID，
`policy_request_id` 是 interface 的整数请求 ID，两者不可混用。

事件包含提交时捕获的 `rollout_id`、`session_id`、`model_generation`、
`submitted_unix_ns`、`metadata_format` 和 `execution_feedback`。终态包含
`completed_unix_ns`；成功时另有 `latency_s` 与 `result`。`result` 是通过 interface
校验后的 `ParsedPolicyAction` 字段，包括 execution、left_wrist、right_wrist、
hand_joint、auxiliary、diagnostics 和 next_metadata_format 等，并非本地模型的
固定形状 `actions` 张量。数组使用 `{dtype, shape, data}` 编码。

动作长度、频率、joint/EEF 表示由远端 metadata/action 决定：完整双臂双手 joint
是 56D，EEF 表示是 62D，但不得假设 chunk 固定 25 步或推理固定 30 Hz。
`execution.rtc` 沿用现有提前 fetch 语义，日志中没有额外冻结前缀或 RTC token。

模型或控制状态切换使请求失效时立即记录 `cancelled`，晚到结果被丢弃，不会进入新的
rollout。人工接管期间暂停远端策略请求；交还时请求新观测，没有额外影子推理流。
日志不重复存储 JPEG 或完整输入，原始状态及相机数据由录制器单独保存。JSONL 记录
请求生命周期，不是 30 Hz 执行表，其条数不等于执行步数。

事件经过录制器异步原始消息队列，CDR 同时保留在 `raw/messages.sqlite3`。停止采样后
等待已登记请求的终态，最长 12 秒；缺失、队列拒收、超时或写盘失败会使原始日志完整性
检查失败并保留错误 sample。异常进程退出仍可能丢失尚未送达的事件。

## 话题

订阅 `/control/record_command`（`start`/`stop`/`discard`）、`/ur/state`、`/sharpa/state`、`/sharpa/tau`、`/sharpa/wrench` 和 `/camera/set`；发布 `/control/record_status` JSON。普通话题 reliable depth=10，相机预览 best-effort depth=1。本机 deformation 默认从独立的 16 槽共享内存环逐帧读取，每槽保留原始时间戳、左右手及十指有效位；环覆盖明确计错，不把旧图当新图。SharpA 仍在有订阅者时发布 `/sharpa/deformation`，供远端、Demo 和旧消费者使用。录制节点设置 `deformation_shared_memory:=false` 可改用 reliable depth=16 的 ROS 订阅；`deformation_shm_prefix` 必须与 SharpA 节点一致，默认 `ur_sharpa_deformation_v1`。

`/sharpa/state` 在同一反馈帧内携带实测 `q` 与最近一次成功交给 SharpA SDK 的完整 44 维 `q_cmd`；停止、来源切换、SDK 失败或单手命令都会使该命令失效。有效的 `q_cmd` 可以保持上一次成功命令，动作时间戳重复不等同于反馈丢帧；质量检查保留动作重复统计，但只对真实反馈、相机和形变要求源帧持续推进。无效、失败或单手动作仍导致对应训练行无效。

`SharpA hand action has no complete successful SDK command` 表示该采集帧的
`q_cmd_valid=false`，缺少可确认的完整双手下发动作，不等于双手实测状态无效或
整条 episode 未保存。驱动在同一反馈帧内通过 `q_cmd_invalid_reason` 区分原因：
首条有效动作前的 `not_commanded` / `source_changed`，以及停止后的 `stopped`
作为启停边界提示，不再产生这条采集错误。`command_failed`（SDK 下发失败）、
`partial_command`（单手命令）、`invalid_command`（命令数据无效）、未知原因，
以及持续采集中出现的启动／来源切换原因仍报错，停止提示也不会清除已有错误。
这些行仍保留在原始时间轴并标记无效，不会用实测 `q` 伪造 `q_cmd`；查看
`sharpa/joints_44.npz` 的 `q_cmd_valid/q_cmd_invalid_reason` 和 `validation.json`
判断影响范围。边界帧仍计入 `dropped_frames`，并按原因单独计入
`manifest.json.quality.boundary_frames`；网页通过 `capture_notice` 显示提示。
更新消息定义后，需要重新构建并整体重启相关 ROS 节点。
HIL 交还的平滑过渡记录实际成功下发的动作；`is_hil` 的具体取值以驱动反馈帧为准。

## 写盘与完整性

30 Hz timeline 独立采样三路相机、UR 状态、SharpA 状态、tau、wrench 和 deformation；缺少必需模态的 tick 会计入 `dropped_frames`。三路相机按各自 source timestamp 独立进入非阻塞视频队列，网页预览不会参与录制。停止时 writer 等待队列并完成 MP4 封装。所有保存目录都使用开始录制时间命名；采集频率和手指缺失由 `validation.json` 单独报告。

内部逻辑分成两层：`RecorderCore` 只负责最新消息、freshness、逐 tick 校验和
样本 schema；`EpisodeWriterCompat` 负责派生指尖 FK、结构化帧、相机和 deformation 三组异步
队列以及落盘生命周期。停止录制固定经过 `drain → close media → quality report`，
随后只进入以下一个终态：

- `discard`：删除整个 `.partial` 目录；
- `error`：仍按开始录制时间命名，保留可诊断数据并写入
  `RECORDING_ERRORS.json`，错误状态不再编码进目录名；
- `complete`：写入 manifest、validation 和 checksums 后原子改名为最终 sample。

`COMPLETE` 只表示文件收尾成功。`manifest.quality.files_ok` 记录写盘完整性，
`quality.ok/capture_ok` 跟随 `validation.ok`；重复帧、低频或缺失传感器会使其为
false，并写入 `CAPTURE_QUALITY_FAILED`，保留数据供诊断，不再误报采集质量合格。

ROS callbacks 不直接做 FFmpeg、压缩或 NPZ 写盘。新增 modality 时必须先加入
`MODALITY_NAMES` 并明确 raw contract；未知 key 会立即报错，避免静默产生残缺数据。

## 磁盘格式

`YYYY-MM-DD_HH-MM-SS.ffffff/data.json` 顶层是 `info`、`data`，保留历史 per-frame key 结构；`color_0=ego`、`color_1=left_wrist`、`color_2=right_wrist`，但字段指向连续 MP4，不再指向逐帧 JPEG。三路视频分别位于 `cameras/<name>/rgb.mp4`，`index.npz` 保存 frame index、source timestamp、wall-clock 和 timeline 映射，`rgb_meta.json` 保存 codec、帧数和实测源频率。

```text
YYYY-MM-DD_HH-MM-SS.ffffff/
├── data.json
├── manifest.json
├── schema.json
├── checksums.sha256
├── COMPLETE
├── timeline/
│   ├── clock.npz
│   └── source_quality.npz
├── ur/
│   └── physical_12.npz
├── sharpa/
│   ├── joints_44.npz
│   └── tactile/
│       ├── force6d.npz
│       ├── deform_u8.zst
│       └── deform_index.npz
├── raw/
│   ├── messages.sqlite3
│   ├── messages_meta.json
│   ├── policy_inference.jsonl  # 有模型推理事件的 rollout
│   └── diagnostic_preroll.json
└── cameras/
    ├── ego/{rgb.mp4,index.npz,rgb_meta.json}
    ├── left_wrist/{rgb.mp4,index.npz,rgb_meta.json}
    └── right_wrist/{rgb.mp4,index.npz,rgb_meta.json}
```

每个 `data` frame 包含 `idx`、`colors`、`depths`、`states`、`actions`、
`tactiles`、`audios`、`tactile_images*` 和 `metadata`。当前 RGB 数据位于 MP4，
`depths` 和 `audios` 为空。`states` 的主要字段为：

- `left_arm/right_arm.qpos`：左右 UR 实测关节，各 6 维；并包含 RTDE 原始
  `tcp_pose6`。
- `left_arm_pose/right_arm_pose.pose`：canonical wrist Pose9，各 9 维。
- `left_hand/right_hand.qpos`：SharpA 实测关节，各 22 维。
- `left_hand_positions/right_hand_positions.positions`：每只手 5×3 指尖位置。
- `left_hand_torques/right_hand_torques.torques`：每只手 22 维关节力矩。
- `tactile.left_tactile/right_tactile.<finger>`：左右手各五指的 6D wrench，
  `[Fx,Fy,Fz,Tx,Ty,Tz]`；力为 N，力矩为 N·m，坐标系为指尖局部。
  有效性见同行 `metadata.tactile_valid.wrench`，无效指填零。数采、Policy 保存
  推理数据及 HIL 均使用此格式，与旧程序的 `force6d.npz` 和逐指 6D 字段兼容。

`actions` 只包含 `left_hand.qpos[22]` 和 `right_hand.qpos[22]`。它们来自
`sharpa_node` 最近一次成功交给 SDK 的完整 44D command（普通示教要求 AE01 来源，rollout 使用 Actor）；
不保存 Vive 双臂 command。每个 action 与同一 `HandStateFrame` 中的实测手部状态
写入同一 timeline row；停止、来源切换、单手命令或 SDK 失败会使 action 失效。

同时写入原始分目录：`timeline/clock.npz`、`timeline/source_quality.npz`（含各源原始 timestamp）、`ur/physical_12.npz`（12D joints、raw TCP Pose6、canonical wrist Pose9）、`sharpa/joints_44.npz`、`sharpa/tactile/force6d.npz` 和 `sharpa/tactile/deform_u8.zst + deform_index.npz`。`joints_44.npz` 将同一 `/sharpa/state` 帧中的实测 `q` 与成功下发的 `q_cmd/q_cmd_valid/q_cmd_timestamp_ns/q_cmd_source` 写入同一个 timeline row；`data.json.actions` 保存所选采集模式的左右手 `q_cmd`，不保存 Vive 臂命令。Pose9 固定为 `[x,y,z,r00,r10,r20,r01,r11,r21]`，不是 9DoF；TCP Pose6 保持 RTDE 原始 `[x,y,z,rx,ry,rz]`。

主要 NPZ shape（`T` 是 30 Hz timeline 行数，各相机帧数独立）如下：

| 文件 | 字段与 shape |
| --- | --- |
| `timeline/clock.npz` | `time_unix_ns[T]`、`elapsed_ns[T]`、`sample_rate_hz`；rollout 另含 `is_hil[T]` |
| `timeline/source_quality.npz` | `source_names[6]`、`valid[T,6]`、`age_ms[T,6]`、`receive_age_ms[T,6]`、`source_age_ms[T,6]`、`source_timestamp_ns[T,6]` |
| `ur/physical_12.npz` | `q[T,12]`、`raw_tcp_pose6[T,2,6]`、`canonical_wrist_pose9[T,2,9]`、`timestamp_ns[T]`、`valid[T]` |
| `sharpa/joints_44.npz` | 同行保存的 `q[T,44]`、成功下发 `q_cmd[T,44]`、`q_cmd_valid[T]`、`q_cmd_timestamp_ns[T]`、`q_cmd_source[T]`、`q_cmd_invalid_reason[T]`（字符串，有效命令为空）、`tau[T,44]`，以及 q/tau timestamp 和 valid |
| `sharpa/tactile/force6d.npz` | `wrench[T,2,5,6]`、`force[T,2,5,3]`（N）、`torque[T,2,5,3]`（N·m）、`timestamp_ns[T]`、`valid[T,2,5]`；左右手顺序，小指到拇指 |
| `sharpa/tactile/deform_u8.zst` | 解压后的逻辑 shape 为 `[N,10,240,240] uint8`，plane 顺序记录在 `schema.json` |
| `cameras/<name>/index.npz` | 相机独立的时间、timeline row、frame index、source timestamp、byte count 和 valid |

`force` 和 `torque` 是 `wrench` 前／后三轴的显式字段，沿用同一个时间轴和有效标记，
不在 ROS 回调里另写一份逐帧文件；停止后统一生成 NPZ。`schema.json.tactile_wrench`
记录单位、坐标系、左右手和轴顺序。`metadata.stamps.tactile` 使用 wrench 的时间戳，
`joint_torque` 单独记录关节力矩时间戳；两者不会互相代用。
`age_ms` 保留历史含义，与 `receive_age_ms` 相同，表示距离本机回调接收的时间；
`source_age_ms` 是 timeline 时间减去传感器源时间，缺失源时间为 NaN。
消息延迟后成批到达时，接收新鲜不能证明物理测量新鲜，需同时检查两者。

保存结束后读取落盘文件生成 `validation.json`。Monitor 显示三路相机、Arm State、
Hand State、Hand Action、Joint Torque、Wrench、Deform 共 9 项；禁用触觉时为 7 项。
每项显示有效帧数、实际保存 Hz、缺失情况；目标 30 Hz，允许 2% 频率容差
（最低 29.4 Hz）。`timing` 另统计推进的源帧 Hz、重复源帧、倒退和超过 1.5 个周期
的间隔（30 Hz 时为 50 ms），重复缓存或长时间停顿不能凭名义帧率达标。
`sampling` 检查整个 timeline 的节拍。Wrench/Deform 按具体手指显示缺失比例。
启动前缓存帧仍保留；源帧间隔只检查与录制窗口相交的部分，避免把录制前的空隙
算作录制中的停顿，裁剪次数记录为 `timing.boundary_gaps_clipped`。
开始下一条记录时，上条校验摘要自动清零；极短 episode 的帧率不能用作持续性能验收。

为了减少对 33.3 ms 采样预算的占用，触觉向量每手每 tick 只转换一次，非有限值
只使对应手指无效；指尖 FK 在异步 writer 线程执行，并串行保护共享 Pinocchio
缓存。形变按源时间戳去重，不重复排队写入大块图像，也不延长重复帧的有效期。
相机保留独立共享内存环和三路编码线程，录制回调不等待编码或文件 I/O。
实时 `/control/record_status.sampling_work` 报告最近 300 帧的采样耗时 P95、最大值
与累计超预算次数，`writer_queues` 报告结构化和形变队列的待处理数量；保存后的
`manifest.json.quality.sampling_work` 保留这些采样指标。队列溢出和缺帧仍明确报错。

无硬件压测使用合成三路图像（1280×720、两路 480×270）、十指 240×240 形变、
真实 FK、CDR 序列化和落盘；分别覆盖数采、Policy、含接管切换的 HIL 录制模式，
后两种额外写入模拟的推理返回。它不运行模型、不连接 SDK 或发布机器人命令：

```bash
bash ur-sharpa/scripts/benchmark_recording.sh --output-root logs/recording_benchmark --seconds 30
# 无 NVIDIA 驱动时可检查 MJPEG 回退路径，不能据此验收 NVENC。
bash ur-sharpa/scripts/benchmark_recording.sh --output-root logs/recording_benchmark_mjpeg --seconds 30 --video-backend mjpeg
```

`benchmark.json` 保存各模式的实际 Hz、源帧 Hz、采样耗时、错过的节拍和排空后的
队列数。失败返回非零状态。现场仍须用真实相机、传感器、目标数据盘及推理负载
检查同一组指标，不能把合成输入压测表述为实机 30 Hz 验证。

真实样本还需对比源时间、回调接收时间与采样时刻。2026-10-20 的 82.3 s 样本中，
SharpA state 源端约 30 Hz，但接收存在 200–350 ms 的停顿与突发；tau/wrench
原始日志约 29.6 Hz，最新值采样后推进的有效源帧却只有约 7.2/6.2 Hz。
主相机实际只捕获约 26.3 Hz；NVENC 接收与写入数一致，不能通过提高 MP4
名义帧率修复。该样本证明低负载的同进程合成测试不足以覆盖实采的 DDS 和 CPU 竞争。

只读审计与手部反馈恢复工具如下，输出目录必须在原 episode 之外：

```bash
source ur-sharpa/scripts/setup_env.sh
/usr/bin/python3 ur-sharpa/infer/audit_recording.py /path/to/episode \
  --output-dir logs/recording_audit --recover-feedback
```

`audit.json` 对比采样唯一源帧、源数据年龄、跨模态偏差、原始接收频率与延迟。
可选 `recovered_feedback.npz` 从原始 CDR 按源时间匹配现有 timeline，默认允许
500 ms 到达延迟、20 ms 源时间偏差；每个源帧最多使用一次，不插值、不补造缺失值。
实测 q、成功下发 q_cmd、来源、HIL 标记、tau、wrench、force/torque 及各自
时间戳和 valid 均保留，缺失值为 NaN/false。它只恢复已有手部日志，不能补回
相机和未保存的形变，不会把原样本改成完整合格 episode。

当前缓解措施包括：小型反馈 topic 保留 RELIABLE，使用与硬件 ACK 一致的
50 ms Fast DDS heartbeat；输入、硬件和录制进程默认限制 BLAS/OMP/MKL 为
单线程，显式环境设置优先，模型服务不受此默认值影响；原始日志后台攒批提交。
这些改动需要重启对应进程才生效，仍需实采确认。

当前录制节点默认使用有界时间戳队列与固定 30 Hz 目标网格，独立工作线程约每
1 ms 排空媒体并尝试成帧。Teleop、Policy 和 HIL 共用该路径。每个反馈／相机源帧
最多匹配一次，SharpA 的 q、成功 q_cmd、HIL 标记、tau、wrench 和 deformation
绑定同一个手部源时间戳。数据到齐即成帧，`alignment_wait_frames` 默认为 2
（30 Hz 下等待预算约 66.7 ms，可设置 0 回到旧采样以作对照）；不会固定等待
500 ms，也不改变机器人控制、状态门和命令来源。等待只影响录制行的组装，
原始消息、视频和触觉仍在到达时独立入队。
统一 launch 对应参数为 `recording_alignment_wait_frames:=2.0`，触觉环名称可用
`deformation_shm_prefix` 同时配置生产与录制两端。

反馈匹配窗口为目标时刻 ±20 ms，相机独立时钟使用 ±一个周期（30 Hz 下
33.3 ms），优先消费窗口内最早的未使用源帧，避免相位抖动导致跳过帧。
超过等待预算、设备源端缺帧、队列覆盖均保留无效标记，不插值或复制补帧。
停止时允许在途数据在同一预算内排空，排除停止之后采集的媒体。
`timeline/clock.npz.time_unix_ns` 是目标网格，
`sampled_unix_ns` 是实际组装时刻；各模态源时间戳保留原值。
`source_quality.source_age_ms` 是目标时刻减源时间，可为负；
对齐模式的 `receive_age_ms` 是组装时刻减接收时刻。
`sampling_alignment` 和保存后的 `sampling_work.alignment` 记录匹配缺失、覆盖、
倒序及实际最大成帧延迟；操作系统调度抖动可能使实际值略超预算。

2026-10-20 18:33:41 的 146.3 s 样本全量解码未发现视频解码错误或完全重复图像，
ego 有一次 66.631 ms 源间隔，对应 SDK 抓帧耗时 65.534 ms；三路后续传输和
NVENC 写入未丢帧。原始 UR／SharpA 小型反馈均为 4388 条唯一源帧，旧最新值
采样产生 UR 重复 1 行、手部／tau／wrench 各重复 3 行；独立形变流缺少两个
已在手部状态中出现的源时刻。137.1–137.4 s 的 10 条无效训练行来自单手命令，
需要与吞吐问题区分。验收同时检查唯一源帧、跨模态偏差、实际成帧延迟、
输入/写入计数和至少 60 s 实采，不能仅检查目标网格是否为 30 Hz。

上述样本中，硬件节点收到命令到发出结果事件的中位／P95 耗时为：
SharpA 0.704／1.442 ms，UR 0.846／1.674 ms；这不包含上游输入、IK、传输
和机械响应，不能作为端到端控制延迟。优化后的离线 720p 图像传输对照测试中，
跨进程交付中位耗时从约 2.5–2.8 ms 降至约 0.5 ms。
合成输入结合真实 FK、CDR、磁盘和 NVENC，Teleop／Policy／HIL 各 20 s
均得到 600 个唯一有效帧；隔离 ROS 域另测 40 ms 延迟到达的图像，
180 帧均完整，成帧延迟 P95 为 45.1 ms、最大 55.5 ms。
这些是软件路径验证，尚不能证明实机相机、输入设备及控制的端到端延迟或
长时间稳定 30 Hz。

完整 sample 最初保存在记录根目录。保存后可在 Monitor 选择 Success 或 Fail，
sample 会移动到根目录的 `success/` 或 `fail/`；目录名使用开始录制时的本地时间。
工作台的“Episode 结果”同时提供“丢弃数据”：点击后直接永久删除当前显示的
episode 整个目录，包括已标记成功／失败的数据。录制、保存期间不可丢弃；
删除请求绑定页面显示的样本路径，样本已变化时拒绝执行，删除后清空保存结果。
上传保持当前相对目录：根目录 sample 上传到远端根目录，分类 sample 上传到远端
`success/` 或 `fail/`。写盘或结构异常的目录也只使用开始录制时的时间名，保留
`RECORDING_ERRORS.json`、`ERROR` 和可诊断 timeline，不能上传。
Monitor 的 Drop 只删除当前展示的最近一条 sample（无论它在根目录、success
还是 fail），删除后不可恢复。

## FK

44D SharpA 状态在 writer 线程调用 `ur_common.kinematics.UrCommonKinematics.hand_fingertip_positions`，内部复用缓存的 Sharpa FK 模型，输出每只手 5×3 指尖位置。计算始终使用采样时冻结的同行关节值。


## 视频与触觉配置

统一 launch 默认 `video_backend:=hevc_nvenc`：相机完整 BGR 通过独立共享内存环交给
录制器，每路一个持续运行的 FFmpeg NVENC 进程，录制主链不经过 JPEG 编解码。
JPEG 只用于限频预览，预览积压时取最新帧，主动跳帧与录制丢帧分别计数。
ZED SDK 仍在独立进程中运行，图像跨进程改用两个有确认的共享内存槽，
管道仅传小型描述符；父进程完成像素复制后才归还槽，防止读到覆盖中的图像。
原始帧使用独立约 1 ms 发布线程，推理与录制不再等待预览的 ROS 定时回调。
结束保存时逐帧解码计数，并把解码器错误视为文件校验失败；成功退出码本身
不足以证明没有坏帧。

| launch 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `video_backend` | `hevc_nvenc` | NVENC HEVC；`mjpeg` 为旧 CPU JPEG 路线 |
| `ego_video_bitrate_mbps` | `30.0` | Ego VBR 目标码率 |
| `wrist_video_bitrate_mbps` | `12.0` | 每台腕部 VBR 目标码率 |
| `camera_preview_fps` | `10.0` | 预览 JPEG 编码上限 |
| `camera_raw_queue_size` | `64` | 每路待传输原始帧队列 |
| `camera_encoded_queue_size` | `256` | JPEG 结果队列，不是录制 NVENC 队列 |
| `camera_shm_slots` | `64` | 每路原始帧环槽数 |
| `camera_shm_bytes` | `4194304` | 每槽需容纳 `width * height * 3 + 20` 字节 |
| `sharpa_skip_tactile` | `false` | `true` 时跳过触觉力／形变的 SDK 采集、发布、保存与校验，同时不再向 AE01 提供指尖力 |

统一 launch 默认启用触觉；不需要时设置 `sharpa_skip_tactile:=true`。仅关闭手套
反馈可设置 `ae01_feedback_enabled:=false`，触觉仍可采集。关节 state、command 和 torque 在
两种设置下都保留；元数据记录 `tactile_enabled`，不能把主动禁用项算作故障。
指尖 F6 与形变按各自最近一次 SDK 回调独立检查 `tactile.max_age_s`（默认
0.25 秒）；过期、无效数据或失联手的对应项补零并标为无效，重复发布不会刷新
SDK 缓存的接收时间。

编码使用 p4、低延迟、8-bit YUV420、VBR CQ=18、GOP=2 **帧**、B=0、lookahead=0，
每两帧强制 IDR。峰值码率与 VBV 缓冲按目标两倍设置，实际 VBR 随画面变化。
录制器每路原始帧队列也为 64；默认三路最大容量共享内存约 768 MiB，预览另用
每路 2 槽，进程内队列另占 RAM。改变分辨率需核对槽容量与内存，扩大队列只吸收
短时抖动，不能补偿持续吞吐不足。

从仓库根目录启动，按需追加编码参数：

```bash
source ur-sharpa/scripts/setup_env.sh
ros2 launch ur_launch teleop.launch.py recording_root:=/absolute/dataset \
  video_backend:=hevc_nvenc ego_video_bitrate_mbps:=40.0 wrist_video_bitrate_mbps:=16.0
```

录制节点用合成画面做 NVENC 启动检查，不可用明确报错，不静默回退 CPU。
真实编码上下文在每段第一帧到达后按实际尺寸创建。保存停止时排空已接收帧，
关闭编码器并核对视频／索引帧数；未排空应重试停止，不能直接开始下一段。

输出保持 `cameras/<name>/rgb.mp4`，HEVC 消费端需支持解码。
`rgb_meta.json` 记录实际 codec、编码器、GOP、码率与探测结果；`encoder_command.json`
保存实际 FFmpeg 参数，`ffmpeg.log` 保存错误。`index.npz.byte_count` 表示输入帧包
大小（`input_payload_bytes`），不是压缩后的帧大小。真实采样率以源时间戳判断。

`/control/record_status.video_encoding` 提供 accepted/written/rejected、队列深度／峰值、
排队与写入耗时。相机 `hardware_encode=false` 描述 JPEG 预览；录制端
`video_encoding.hardware_encode=true` 才描述 NVENC，二者不矛盾。
评估录制看 capture、recording_raw_drop_count、accepted/written 和最终视频帧数，
不把约 10 Hz 的 encoded_count 当作录制帧率。共享内存覆盖、拒收、编码失败会导致
保存质量失败；驱动层缺帧还需结合源端计数检查。

## 原始消息日志与完整性

`raw/messages.sqlite3` 按接收保存 UR/手状态、tau、启用的 wrench、机械臂和手部命令与结果、
SDK 执行事件及相机／SharpA 采集诊断。图像与形变沿用独立存储，不重复写入 SQLite。
无效位、NaN、单侧有效性与命令原值保留；训练 timeline 的重采样和 `data.json`
筛选不影响原始消息保存。

`messages` 表字段为 `id`、`topic`、`type_name`、`receive_sequence`、
`received_unix_ns`、`received_monotonic_ns`、`source_timestamp_ns`、`cdr`。
接收序号是录制期内各 topic 的独立计数，不是设备帧号；CDR 可用 ROS 反序列化恢复。
String 没有统一的源时间戳字段，内部时间需从 JSON 载荷读取。

日志独立线程批量提交，默认 `raw_message_budget_mb=64`，预算含待写与正在写的载荷
及元数据估算，不是进程总内存上限。预算耗尽时明确 rejected、序号继续递增；拒收、
序列化或写入错误导致原始日志不完整，并使 episode 质量失败。正常停止先停止接收、
排空并关闭数据库，再完成其他文件；超时保留目录、禁止新录制，可重试停止。
进程异常退出／断电仍可能丢未提交队列。
后台每批最多等待 5 ms 或收集 128 条，SQLite 使用 WAL 并保留 `synchronous=FULL`；
关闭时 checkpoint 并恢复 DELETE 模式，归档为单个数据库，后续只读审计不会
在原 episode 中生成 WAL/SHM 文件。
回调不等待提交，正常停止立即排空。`commits/max_batch_size/commit_ms_max`
用于检查高频 execution event 是否导致过多小事务。

`messages_meta.json` 和 manifest 的 `quality.raw_messages` 记录 received/written/
rejected、积压峰值、错误及 complete。complete 只证明本节点已接收到的消息落盘，
不证明设备持续出帧或 DDS 无丢包。被动命令／诊断订阅使用 BEST_EFFORT，
完整性验收同时看源端计数、有效行频率、时间戳和相机输入／输出。

离线检查不需要 ROS 或硬件：

```bash
python3 ur-sharpa/scripts/inspect_raw_messages.py /path/to/episode
```

输出接收频率、最长间隔、超过 50 ms 间隔、源时间戳重复／回退、接收序号内部缺口。
50 ms 是诊断阈值；边界丢失需看 rejected，不能只看内部序号。

## 连续采集与 USB 诊断

相机节点每秒写 `acquisition_*.jsonl`，覆盖空闲、录制和保存阶段。默认目录为
`$ROS_LOG_DIR` 或 `~/.ros/log/`，可用 `acquisition_log_dir:=/path/to/logs` 指定。
每段约 32 MiB，保留当前段和 3 个历史段；sample 的 `raw/diagnostic_preroll.json`
保留开始前最近最多 30 条诊断。

日志包含 USB 拓扑／协商速率／控制器、可读取的内核事件、CPU/iowait/压力与磁盘网络
计数、相机抓取／队列／丢帧／编码耗时和录制状态。相机逐帧时序含
`capture_ready` / `raw_published`，按源时间戳关联；字段解释见
[UR 执行日志说明](../../../README.md#hil-与执行日志)。
SharpA 诊断包含状态回调尝试数、执行耗时和回调间隔，结合 SDK 事件判断源端与接收延迟。
尝试数不是有效反馈帧数，诊断定时器自身也可能延迟。

`speed=480/5000` 是 USB 协商速率（Mbps），不是实时带宽；URB 数量也不是流量。
有 usbmon 读取权限时，另开终端从仓库根目录运行可选辅助程序：

```bash
sudo modprobe usbmon
sudo /usr/bin/python3 ur-sharpa/scripts/usb_traffic.py
```

它只读 URB 头、累计完成字节与错误，不复制图像载荷，每秒写
`/tmp/ur_sharpa_usbmon.json`。相机日志自动读取，5 秒未更新标过期；Ctrl+C 停止。
完成字节数不含全部协议开销，辅助程序自身丢事件另计；未启用时不能把缺失解释为零负载。

复测固定场景与相机配置：启动后空闲 20 秒，录制 60 秒，保存后保持节点运行 20 秒；
保留 sample 和覆盖前／中／后的连续日志，再做持续采集检查。ZED 抓取变慢看 SDK／
调度和 USB，采集正常但排队增加看编码／CPU／磁盘。腕部按 RealSense SDK 序列号
绑定，两台需稳定枚举并核对 USB 3 连接；不同插口仍可能共用控制器。

历史 NVENC 实机采集改善和未达 30 Hz 的证据统一见
[实验分析汇总](../../../../analysis/README.md)，不能将早期「尚未切换 HEVC」报告
作为当前配置。保存成功、checksum 一致、可解码和视觉无坏帧也应分别验收。
