# Quest 遥操

Quest 通过 Windows 上的 Meta Quest Link 原生客户端提供头部和双手柄位姿。PND 端完成
标定、Adam Pro 上半身 retarget、指令门控和 ZED 视频转发；不需要浏览器或 SteamVR。

## 数据链路

```text
Quest 3 --USB--> Meta Quest Link --OpenXR--> PNDQuestTeleop.exe
  tracking --WSS--> quest_webvr (/_quest/tf, /_quest/joy)
                    -> quest_retarget (19D JointState)
                    -> quest command gate
                    -> /adam_command_joint_states
  video    <--TCP-- ZED MPEG-TS stream (default 10.10.20.126:5602)
```

Windows 主动连接 PND 和 ZED，因此不需要固定 Windows IP。默认地址可通过
`PND_QUEST_ROS_HOST`、`PND_QUEST_ZED_HOST` 和 `PND_QUEST_ZED_PORT` 修改。

## 标定与 retarget

- 机器人完成 `bias_init -> bias` 后，操作者保持头部水平、双手水平向前且掌心向下，
  按右手柄 `A`。
- 当前 Quest 头部和双手柄位姿成为人体零位；同一时刻最终 Bias 的 FK 位姿成为机器人
  零位。之后手腕平移按米制 `1:1` 映射，旋转使用相对四元数。
- 左右手柄独立维护 `Normal`/`Error` 状态。`Known` 和 `Inferred` 都执行；`Lost` 或
  单帧位置跳变超过 `0.1m` 的手柄冻结在最后一帧 `Normal` 位姿，另一只手继续执行。
  手柄恢复为 `Known`/`Inferred` 后，若距离最后一帧 `Normal` 不超过 `0.1m` 会自动恢复；
  跳变较大时必须按右手柄 `A` 接受新位置。A 标定命令按手柄独立生效，不要求另一只手
  同时为 `Normal`。
- Waist 保持 Bias 值，头显朝向控制 `neckYaw` 和 `neckPitch`，双臂各 7 个关节由腕部
  `XYZ + orientation` 目标求解。
- `quest_enable_neck:=false` 可关闭头显到颈部的映射；此时 `neckYaw` 和
  `neckPitch` 保持按下 A 标定时的 Bias 数值，双臂不受影响。默认值为 `true`。
- 若 OpenXR 没有上报 A 键，可在 Quest tracking 有效时运行
  `ros2 service call /quest/calibrate std_srvs/srv/Trigger {}`，它与按 A 使用同一套标定逻辑。
- IK 使用关节限位、上一帧连续性和 Bias 姿态代价。误差自适应 LM 阻尼与非线性回溯
  用于稳定近奇异位形；输出没有速度限幅或碰撞冻结。
- Quest 不提供肘部位姿，因此冗余肘部构型由上一帧和 Bias 姿态共同确定。
  `nonlinear_ik` 额外将左肘约束到躯干 `+Y` 外侧、右肘约束到 `-Y` 外侧；默认软约束
  权重固定为 `50.0`，肩肘外侧余量固定为 `0.02` 米，启动时无需单独传递参数。若求解
  结果仍让某侧肘部继续向内，该侧手臂保持上一帧，另一侧不受影响。

仅真实 OpenXR tracking frame 会刷新 watchdog。断连或超过 `0.2 s` 没有新帧都会停止
Quest 指令；单手 tracking 失效时只冻结该手的最后一帧，另一只手仍可继续执行。

## Windows 客户端

### 环境

1. 安装 Meta Quest Link，并在应用设置中将 Meta Quest Link 设为当前 OpenXR runtime。
2. 安装 .NET 8 SDK x64。
3. 使用 USB 3 线连接 Quest，确认 Meta Quest Link 能识别头显和两个 Touch 手柄，并在
   头显内进入 Quest Link。

### 放置与编译

将完整的 `src/quest_node/windows/native` 目录放到 Windows，例如：

```bash
scp -r src/quest_node/windows/native \
  baai@10.10.20.124:'C:/Users/baai/Desktop/PNDQuestTeleop'
```

在 Windows `cmd` 中编译：

```cmd
cd C:\Users\baai\Desktop\PNDQuestTeleop
dotnet restore
dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=false -o publish
```

如果 SDK 安装在用户目录，将 `dotnet` 替换为实际路径，例如
`C:\Users\baai\.dotnet-sdk\dotnet.exe`。必须复制整个目录，不能只复制 `.exe`；
`publish` 中还包含 StereoKit、LibVLC 和 OpenXR 运行依赖。

### 地址与启动

默认不需要修改地址。网络变化时，在启动前设置：

```cmd
set PND_QUEST_ROS_HOST=10.10.20.127
set PND_QUEST_ZED_HOST=10.10.20.126
set PND_QUEST_ZED_PORT=5602
Start-PNDQuestTeleop.cmd --check
Start-PNDQuestTeleop.cmd
```

客户端必须从登录后的 Windows 桌面启动。SSH 和 Windows service 属于 Session 0，无法
创建 Meta OpenXR 会话。启动脚本会检查当前 OpenXR runtime、关闭旧客户端，并只对当前
进程禁用可能冲突的 MANUS OpenXR hand-tracking layer。

如需在 Windows 桌面查看 Quest Link 头显画面，运行：

```cmd
"%ProgramFiles%\Oculus\Support\oculus-diagnostics\OculusMirror.exe"
```

`OculusMirror` 只用于桌面镜像查看，不参与 tracking、ZED 视频或 ROS 数据传输。

头显内只显示跟随视线的 ZED 图像和左右手柄绿色坐标系。Windows 客户端无条件上传 Joy、
标定命令和左右手柄 tracking 状态；按钮输入和 6DoF 位姿是两条独立的 OpenXR 状态链。
Quest 无法在手柄完全离开头显摄像头视野时凭软件强制恢复绝对位置；要长期保持稳定位置，
必须让手柄上的红外标记处于头显摄像头可见范围，并保证光照、无遮挡和 Quest Link 会话稳定。

## ROS 启动

正式遥操统一使用根目录 README 中的命令：

```bash
# `quest_retarget_method` 三种模式：
#   nonlinear_ik：非线性全局 IK，默认使用
#   shoulder_prior：双腕目标 + 肩部姿态软先验
#   local_qp：局部速度级 QP IK
ros2 launch teleoperation teleoperation.launch.py \
  mode:=teleop teleop_source:=quest \
  quest_retarget_method:=nonlinear_ik \
  start_manus:=true start_sharpa:=true
```

切换后必须重启 launch 并重新按右手柄 A 标定。`nonlinear_ik` 需要 `quest_python` 指向的
Python 环境安装 `casadi>=3.7.2`。

Quest 只替代 Adam 上半身输入；正式命令仍启动 Manus 手套与 SharpA 驱动。Xbox 单击
`LB` 打开 Manus-SharpA 手部通道，单击 `LT` 打开 Quest-Adam 上肢通道。

安全 dry-run 验证使用：

```bash
ros2 launch teleoperation quest_test.launch.py
```

Quest 模式会自动启用 ZED 的 Quest TCP 视频流。`quest_test.launch.py` 默认关闭 Adam
硬件输出、Noitom、SharpA 和 actor/obs，仅保留 Quest、Bias、Status、Adam、ZED、记录与
Foxglove；只有有人看护真机时才传入 `adam_dry_run:=false`。

## 网络与 nginx

PND nginx 需要包含 `config/nginx_quest_locations.conf` 中的 `/webvr/`、
`/vrwebsocket` 和 runtime-config 路由。Windows 需要能够访问 PND HTTPS/WSS 和 ZED
TCP `5602`。网页 WebXR 与 TURN 仍可作为调试方式显式启用，但不是正式 Meta Link 链路。

Foxglove 可观察 `/adam_command_joint_states`、
`/_quest/retargeted_joint_states_raw`、Quest TF、Joy、tracking status 和 command status。
