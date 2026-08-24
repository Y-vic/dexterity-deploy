# recording_node（中文说明）

## 话题

订阅 `/control/record_command`（`start`/`stop`）、`/ur/state`、`/sharpa/state`、`/sharpa/tau`、`/sharpa/wrench`、`/sharpa/deformation` 和 `/camera/set`；发布 `/control/record_status` JSON。普通话题 reliable depth=10；相机和 deformation 使用 best-effort depth=1。`/sharpa/state` 在同一反馈帧内携带实测 `q` 与最近一次成功交给 SharpA SDK 的完整 44 维 `q_cmd`；停止、来源切换、SDK 失败或单手命令都会使该命令失效。

## 写盘与完整性

30 Hz timeline 独立采样三路相机、UR 状态、SharpA 状态、tau、wrench 和 deformation；缺任何模态的 tick 会计入 `dropped_frames`。三路相机按各自 source timestamp 独立进入非阻塞视频队列，网页预览不会参与录制。停止时 writer 等待队列并完成 MP4 封装。所有保存目录都使用开始录制时间命名；采集频率和手指缺失由 `validation.json` 单独报告。

内部逻辑分成两层：`RecorderCore` 只负责最新消息、freshness、逐 tick 校验和
样本 schema；`EpisodeWriterCompat` 只负责结构化帧、相机和 deformation 三组异步
队列以及落盘生命周期。停止录制固定经过 `drain → close media → quality report`，
随后只进入以下一个终态：

- `discard`：删除整个 `.partial` 目录；
- `error`：仍按开始录制时间命名，保留可诊断数据并写入
  `RECORDING_ERRORS.json`，错误状态不再编码进目录名；
- `complete`：写入 manifest、validation 和 checksums 后原子改名为最终 sample。

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
- `tactile`：左右手各五指的 6D wrench。

`actions` 只包含 `left_hand.qpos[22]` 和 `right_hand.qpos[22]`。它们来自
`sharpa_node` 最近一次成功交给 SDK 且来源为 Manus Teleop 的完整 44D command；
不保存 Vive 双臂 command。每个 action 与同一 `HandStateFrame` 中的实测手部状态
写入同一 timeline row；停止、来源切换、单手命令或 SDK 失败会使 action 失效。

同时写入原始分目录：`timeline/clock.npz`、`timeline/source_quality.npz`（含各源原始 timestamp）、`ur/physical_12.npz`（12D joints、raw TCP Pose6、canonical wrist Pose9）、`sharpa/joints_44.npz`、`sharpa/tactile/force6d.npz` 和 `sharpa/tactile/deform_u8.zst + deform_index.npz`。`joints_44.npz` 将同一 `/sharpa/state` 帧中的实测 `q` 与成功下发的 Manus `q_cmd/q_cmd_valid/q_cmd_timestamp_ns/q_cmd_source` 写入同一个 timeline row；`data.json.actions` 只保存左右手的 Manus Teleop `q_cmd`，不保存 Vive 臂命令。Pose9 固定为 `[x,y,z,r00,r10,r20,r01,r11,r21]`，不是 9DoF；TCP Pose6 保持 RTDE 原始 `[x,y,z,rx,ry,rz]`。

主要 NPZ shape（`T` 是 30 Hz timeline 行数，各相机帧数独立）如下：

| 文件 | 字段与 shape |
| --- | --- |
| `timeline/clock.npz` | `time_unix_ns[T]`、`elapsed_ns[T]`、`sample_rate_hz` |
| `timeline/source_quality.npz` | `source_names[6]`、`valid[T,6]`、`age_ms[T,6]`、`source_timestamp_ns[T,6]` |
| `ur/physical_12.npz` | `q[T,12]`、`raw_tcp_pose6[T,2,6]`、`canonical_wrist_pose9[T,2,9]`、`timestamp_ns[T]`、`valid[T]` |
| `sharpa/joints_44.npz` | 同行保存的 `q[T,44]`、成功下发 Manus `q_cmd[T,44]`、`q_cmd_valid[T]`、`q_cmd_timestamp_ns[T]`、`q_cmd_source[T]`、`tau[T,44]`，以及 q/tau timestamp 和 valid |
| `sharpa/tactile/force6d.npz` | `wrench[T,2,5,6]`、`timestamp_ns[T]`、`valid[T,2,5]` |
| `sharpa/tactile/deform_u8.zst` | 解压后的逻辑 shape 为 `[N,10,240,240] uint8`，plane 顺序记录在 `schema.json` |
| `cameras/<name>/index.npz` | 相机独立的时间、timeline row、frame index、source timestamp、byte count 和 valid |

保存结束后会读取落盘文件生成 `validation.json`。Monitor 显示 Ego、Left Wrist、
Right Wrist、Hand State、Hand Action、Wrench、Deform、Arm State 共 8 项的有效
帧数、记录时长、实际 Hz 和缺失情况，`25 Hz` 及以上视为频率达标；
Wrench/Deform 会按左右手具体手指显示缺失比例。开始下一条记录时，Monitor 中
上一条校验摘要自动清零。

完整 sample 最初保存在记录根目录。保存后可在 Monitor 选择 Success 或 Fail，
sample 会移动到根目录的 `success/` 或 `fail/`；目录名使用开始录制时的本地时间。
上传保持当前相对目录：根目录 sample 上传到远端根目录，分类 sample 上传到远端
`success/` 或 `fail/`。写盘或结构异常的目录也只使用开始录制时的时间名，保留
`RECORDING_ERRORS.json`、`ERROR` 和可诊断 timeline，不能上传。
Monitor 的 Drop 只删除当前展示的最近一条 sample（无论它在根目录、success
还是 fail），删除后不可恢复。

## FK

44D SharpA 状态在线调用 `ur_common.kinematics.UrCommonKinematics.hand_fingertip_positions`，内部复用缓存的 Sharpa FK 模型，输出每只手 5×3 指尖位置。
