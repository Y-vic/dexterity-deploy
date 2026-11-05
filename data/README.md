# 本机数据

- `episodes/`：数采与 Policy rollout 的默认保存目录，也是参考数据默认入口。
- `demo_recordings/`：独立 Demo 相机录制。

这些目录保留现有录制数据，不纳入 Git。切换保存目录请通过工作台完成，
设置保存到 `runtime/config/ur_sharpa/recording.json`。
不要将源码、SDK、模型 checkpoint 或构建产物放入数据目录。
