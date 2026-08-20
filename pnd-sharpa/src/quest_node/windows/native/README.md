# PND Quest Windows Client

该客户端通过 Meta Quest Link 的 OpenXR 会话读取头显和 Touch 手柄位姿，将 tracking
发送给 PND ROS，并在头显内显示 ZED 视频。完整安装、编译和正式启动流程见
`src/quest_node/README.md`。

## Build

```cmd
dotnet restore
dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=false -o publish
```

运行 `Start-PNDQuestTeleop.cmd --check` 检查 Meta OpenXR runtime、程序路径和网络地址，
然后运行 `Start-PNDQuestTeleop.cmd`。可在启动前设置：

```cmd
set PND_QUEST_ROS_HOST=10.10.20.127
set PND_QUEST_ZED_HOST=10.10.20.126
set PND_QUEST_ZED_PORT=5602
set PND_QUEST_DISABLE_HW_DECODE=1
set PND_QUEST_NETWORK_CACHE_MS=100
```

启动脚本默认使用已验证稳定的软件解码路径；当前 `1280x720@30` 视频的软件解码和
纹理上传均能稳定达到 30 fps。设置 `PND_QUEST_DISABLE_HW_DECODE=0` 可重新启用
LibVLC 硬件解码。实际帧率修复来自默认的 `100 ms` 网络缓存：它避免 30 fps
视频在 `20 ms` 缓存下被 LibVLC 持续判定为迟到帧。

客户端必须从交互式 Windows 桌面启动，不能通过 SSH 或 Windows service 启动。运行日志
写入程序目录的 `quest-teleop.log`，目录不可写时回退到 `%TEMP%\PNDQuestTeleop.log`。
