# status_node（中文说明）

## 作用

本节点发布 `/control/status`，管理 UR（stopped→homed→running）、SharpA（stopped→running）和录制（idle→recording）三个独立状态机。Teleop 中它是隐藏的 ROS 控制层，不提供网页；网页由 `monitor_node` 提供。Deploy 模式可选保留 `127.0.0.1:8080` 兼容 API。

## 话题

发布：`/control/status`（`ControlStatus`）、`/control/bias_request`（`String`）、`/control/record_command`（`String`）、`/hardware/stop`（`Bool`）。订阅：`/control/ui_command`、`/camera/set`、`/control/bias_status`、`/control/record_status`、`/teleop/vive_status`、`/teleop/manus_status`、`/ur/command_result`、`/sharpa/command_result`。普通话题 reliable depth=10；CameraSet 为 best-effort depth=1。

## 参数与闸门

`mode:=teleop|deploy` 决定 `active_source` 为 `SOURCE_VIVE|SOURCE_ACTOR`；`port` 默认 8080；`camera_max_age_s` 默认 0.25。录制按钮要求 UR running、SharpA running、ego/left_wrist/right_wrist 三路 CameraSet 自身采样时间戳均新鲜且 valid。所有拒绝都写入 `gate_reason`。

## 操作流程

启动 UR → 等待 bias status=done → 允许 UR 运行；启动 SharpA → 等待归零完成；三者满足后开始录制。复位 UR 会先停止录制并重新发 `home_ur`。Jog 必须先 Dry-run 预览目标位姿，再点击确认；距离≤5 mm、速度≤0.02 m/s、加速度≤0.1 m/s²。
