# recording_node（中文说明）

## 话题

订阅 `/control/record_command`（`start`/`stop`）、`/ur/state`、`/sharpa/state`、`/sharpa/tau`、`/sharpa/wrench`、`/sharpa/deformation` 和 `/camera/set`；发布 `/control/record_status` JSON。普通话题 reliable depth=10；相机和 deformation 使用 best-effort depth=1。`/sharpa/state` 同时携带仍新鲜的 Manus Teleop `q_cmd`；它表示收到的操作者目标，不依赖 SharpA 是否运行或硬件仲裁是否放行。

## 写盘与完整性

30 Hz timeline 独立采样三路相机、UR 状态、SharpA 状态、tau、wrench 和 deformation；缺任何模态的 tick 会计入 `dropped_frames`。三路相机按各自 source timestamp 独立进入非阻塞视频队列，网页预览不会参与录制。停止时 writer 等待队列并完成 MP4 封装。写盘或文件结构损坏才进入 error 目录；采集频率和手指缺失由 `validation.json` 单独报告。

## 磁盘格式

`sample_XXXX/data.json` 顶层是 `info`、`data`，保留历史 per-frame key 结构；`color_0=ego`、`color_1=left_wrist`、`color_2=right_wrist`，但字段指向连续 MP4，不再指向逐帧 JPEG。三路视频分别位于 `cameras/<name>/rgb.mp4`，`index.npz` 保存 frame index、source timestamp、wall-clock 和 timeline 映射，`rgb_meta.json` 保存 codec、帧数和实测源频率。

```text
sample_XXXX/
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
`sharpa_node` 最近一次收到、仍在 freshness 窗口内且来源为 Manus Teleop
的完整 44D command；不保存 Vive 双臂 command。这里记录的是操作者输入，
因此 SharpA 暂停、仲裁拒绝或硬件发送失败不会删除仍然新鲜的 Manus action。

同时写入原始分目录：`timeline/clock.npz`、`timeline/source_quality.npz`（含各源原始 timestamp）、`ur/physical_12.npz`（12D joints、raw TCP Pose6、canonical wrist Pose9）、`sharpa/joints_44.npz`、`sharpa/tactile/force6d.npz` 和 `sharpa/tactile/deform_u8.zst + deform_index.npz`。`joints_44.npz` 同时保存实测 `q` 与收到的 Manus `q_cmd/q_cmd_valid/q_cmd_timestamp_ns/q_cmd_source`；`data.json.actions` 只保存左右手的 Manus Teleop `q_cmd`，不保存 Vive 臂命令。Pose9 固定为 `[x,y,z,r00,r10,r20,r01,r11,r21]`，不是 9DoF；TCP Pose6 保持 RTDE 原始 `[x,y,z,rx,ry,rz]`。

主要 NPZ shape（`T` 是 30 Hz timeline 行数，各相机帧数独立）如下：

| 文件 | 字段与 shape |
| --- | --- |
| `timeline/clock.npz` | `time_unix_ns[T]`、`elapsed_ns[T]`、`sample_rate_hz` |
| `timeline/source_quality.npz` | `source_names[6]`、`valid[T,6]`、`age_ms[T,6]`、`source_timestamp_ns[T,6]` |
| `ur/physical_12.npz` | `q[T,12]`、`raw_tcp_pose6[T,2,6]`、`canonical_wrist_pose9[T,2,9]`、`timestamp_ns[T]`、`valid[T]` |
| `sharpa/joints_44.npz` | `q[T,44]`、收到的 Manus `q_cmd[T,44]`、`q_cmd_valid[T]`、`q_cmd_timestamp_ns[T]`、`q_cmd_source[T]`、`tau[T,44]`，以及 q/tau timestamp 和 valid |
| `sharpa/tactile/force6d.npz` | `wrench[T,2,5,6]`、`timestamp_ns[T]`、`valid[T,2,5]` |
| `sharpa/tactile/deform_u8.zst` | 解压后的逻辑 shape 为 `[N,10,240,240] uint8`，plane 顺序记录在 `schema.json` |
| `cameras/<name>/index.npz` | 相机独立的时间、timeline row、frame index、source timestamp、byte count 和 valid |

保存结束后会读取落盘文件生成 `validation.json`。Monitor 显示 Ego、Left Wrist、
Right Wrist、Hand State、Hand Action、Wrench、Deform、Arm State 共 8 项的有效
帧数、记录时长、实际 Hz 和缺失情况，`25 Hz` 及以上视为频率达标；
Wrench/Deform 会按左右手具体手指显示缺失比例。开始下一条记录时，Monitor 中
上一条校验摘要自动清零。

完整 sample 最初保存在记录根目录。保存后可在 Monitor 选择 Success 或 Fail，
sample 会移动到根目录的 `success/` 或 `fail/`；编号在三处目录中保持全局递增。
上传保持当前相对目录：根目录 sample 上传到远端根目录，分类 sample 上传到远端
`success/` 或 `fail/`。真正写盘损坏的目录使用 `sample_XXXX.error_YYY`，保留
`RECORDING_ERRORS.json`、`ERROR` 和可诊断 timeline，不能上传。
Monitor 的 Drop 只删除当前展示的最近一条 sample（无论它在根目录、success
还是 fail），删除后不可恢复。

## FK

44D SharpA 状态在线调用 `ur_common.kinematics.UrCommonKinematics.hand_fingertip_positions`，内部复用缓存的 Sharpa FK 模型，输出每只手 5×3 指尖位置。
