# 推理接口与部署配置设计

本设计统一模型启动配置、推理输入输出声明和机器人连接配置。模型声明需要哪些观测、输出哪些动作，机器人端通过 metadata 获取声明并完成数据组装和执行。EEF 支持 root 或 camera 参考系，输入和输出可分别选择。

状态：设计已整理，尚未接入配置加载器和运行时协议。当前生效的 wire 协议仍见 [interface_contract.md](interface_contract.md)；本文新增字段不能直接交给现有客户端使用。以下路径、模型名和数值均为示例。

## 配置职责

| 配置 | 使用方 | 内容 |
| --- | --- | --- |
| 推理服务配置 | 模型服务 | 模型加载、任务身份、监听地址、输入输出要求、执行默认值 |
| 连接配置 | 机器人 policy client | 服务地址、超时、SSH 连接信息 |
| 远端启动配置 | 部署启动器 | SSH 主机、代码与资产目录、环境脚本、launcher |
| 本体标定配置 | 机器人 adapter | 相机外参、root 映射、TCP 偏移、硬件参数 |

沿用已有 task/run 选择流程。启动器从 task.json 解析任务、run、checkpoint 和 prompt，生成最终推理服务配置；同一信息不在多个文件中独立维护。命令行覆盖应在启动前完成解析，metadata 发布实际生效值。

## 推理服务配置

```yaml
schema: sharpa_inference_config.v1

model:
  family: your_policy
  checkpoint_path: /path/to/checkpoint
  device: cuda:0
  precision: float32

task:
  task_id: example_task
  run_id: example_run
  prompt: "执行指定任务"

server:
  host: 127.0.0.1
  port: 5500
  max_message_size: 67108864

observation:
  format_id: example_obs_v1
  image:
    ego_cam:         {history_len: 0, current: true}
    left_wrist_cam:  {history_len: 0, current: true}
    right_wrist_cam: {history_len: 0, current: true}
  state:
    history_len: 0
    current: true
    eef_frame: root
    left_wrist:  {joint: false, eef: true}
    right_wrist: {joint: false, eef: true}
    hand_joint: {left: true, right: true}
  sensor:
    tau:         {history_len: 0, current: false}
    wrench:      {history_len: 0, current: false}
    deformation: {history_len: 0, current: false}

action:
  eef_frame: camera
  eef_camera: ego_cam
  left_wrist:  {joint: false, eef: true}
  right_wrist: {joint: false, eef: true}
  hand_joint: {left: true, right: true}

execution:
  frequency_hz: 20
  action_length: 32
  execute_start: 0
  execute_length: 16
  prefetch_steps: 0
```

`model.family` 选择模型适配器；checkpoint 必须可加载，device 和 precision 必须受该适配器支持。模型专属预处理、归一化和内部动作表示由模型适配器管理。

`history_len` 不包含 current，必须为非负整数。`history_len: 0` 且 `current: false` 表示禁用该模态。相机和传感器的固定名称沿用现有协议。state 的历史长度由所选状态量共享；关闭 state 时，其各项选择也应关闭。

action 中的布尔值声明服务返回哪些控制量。左右手腕可独立启用 joint 或 eef；若两者同时启用，必须在接入前明确 adapter 的消费规则，不能隐式选择或重复下发。第一版可只允许每侧二选一。至少启用一个动作输出。

## EEF 参考系与可选规则

`eef_frame` 的可选值为 `root`、`camera`。optional 指是否启用 EEF；启用后参考系必须明确，不允许猜测。

| 条件 | 规则 |
| --- | --- |
| 任一侧启用 EEF | 对应 observation 或 action 的 eef_frame 必填 |
| 两侧均未启用 EEF | 省略 eef_frame 和 eef_camera |
| eef_frame 为 root | 省略 eef_camera |
| eef_frame 为 camera | eef_camera 必填，指向明确的相机 frame |

输入配置放在 `observation.state`，输出配置放在 `action`。各自配置对左右手腕共同生效；第一版不支持同一方向内左右手使用不同参考系。输入为 root、输出为 camera 等组合允许使用。

为保持现有坐标定义，设计中的 root 指现有 canonical_root，而非未经转换的 UR base 或任意机器人底座。root 轴沿用 +X=left、+Y=up、+Z=forward。若实际模型的 root 定义不同，需由 adapter 显式转换。

camera 指指定相机的 optical frame，采用 +X=right、+Y=down、+Z=forward。`eef_camera` 使用 ego_cam、left_wrist_cam 或 right_wrist_cam 等已注册名称；名称到实际 frame 和标定的映射由本体配置提供。选择相机参考系不强制启用该相机图像输入，但必须能取得对应变换。

EEF 都表示所选参考系中的绝对 wrist 位姿，位置单位为米，旋转为 column-Rot6D，pose9 顺序沿用现有协议。wrist child 仍是 left_hand_C_MC / right_hand_C_MC；TCP 偏移保留在 adapter 内。root/camera 与 absolute/relative 是两个独立概念，本设计不新增 relative wire 动作。

转换使用列向量约定：`T_root_wrist = T_root_camera @ T_camera_wrist`；反向使用 `inverse(T_root_camera)`。机器人 adapter 负责跨参考系转换，模型服务负责将模型内部输出转换为声明的绝对位姿表示。

相机外参必须与观测时间对应。对移动相机，动作 chunk 的 camera 参考系固定为对应请求的参考时刻，不能在每一步执行时重新绑定相机位置。后续协议需携带可关联的 frame、参考时间和 calibration_id，机器人保留该请求的变换快照直到动作消费完毕；历史状态逐帧使用各自时间的变换。缺少标定、时间对应关系或变换快照时拒绝执行，不静默退回 root。具体 wire 字段和版本须随协议升级一并确定。

## 动作执行规则

execution 是服务生成每次 action.execution 的默认配置。每次返回仍需包含完整执行参数，并与实际数组长度一致；客户端按本次返回值执行。

```text
frequency_hz > 0
action_length > 0
execute_start >= 0
execute_length > 0
execute_start + execute_length <= action_length
0 <= prefetch_steps < execute_length
```

长度、索引和预取步数均为整数。输出动作数组第一维为 action_length，实际执行区间为 `[execute_start, execute_start + execute_length)`。机器人还需检查频率和动作类型是否受本体支持。

配置字段 prefetch_steps 映射到现有 wire 字段 execution.rtc：0 表示本块完成后请求，N 表示剩余 N 步时请求。沿用现有协议，预取步数必须小于 execute_length，因此只执行一步时只能设为 0。它不代表模型内部的 RTC 冻结前缀推理。同一时间最多一个推理请求在途，下一块须等待当前块执行完成才能切换。

## 机器人连接配置

```yaml
schema: sharpa_policy_connection.v1
url: ws://127.0.0.1:5500/infer
timeout_s: 60
ssh:
  host: BAAI
  remote_host: 127.0.0.1
  remote_port: 5500
```

直连时省略 ssh。SSH 模式下 remote_host 和 remote_port 是远端访问的模型服务地址；启动时应检查其与 URL 指向一致。该结构映射到现有 ServerConfig 的扁平 SSH 字段。连接配置不重复声明观测或动作格式，以服务 metadata 为准。

远端 repo_root、asset_root、task_config_root、env_file 继续沿用 [模型服务配置](../config/README.md)，用于发现任务和启动 launcher；标定矩阵也不混入连接配置。

## 服务生命周期与配置发布

1. 部署端选择 task/run，解析路径并启动模型服务。
2. 服务校验配置、加载模型，准备完成后通过健康检查。
3. 客户端读取 metadata，检查协议版本、输入输出能力及坐标变换条件。
4. 客户端 reset 建立新 session，再按输入声明组装 observation。
5. 服务返回 action，客户端校验身份、格式和执行切片后交给 adapter。
6. 停止或切换模型时废弃过期结果；重新读取 metadata 并 reset。

沿用 `/healthz` 健康检查、`GET /metadata`、`POST /reset` 和 WebSocket `/infer`。metadata/reset 和推理载荷继续使用现有 MsgPack 编解码约定。

metadata 应发布实际生效的模型任务身份、协议版本、输入声明、输出声明和执行默认值。输入声明映射到现有 metadata_format；输出声明和执行默认值为待新增部分。连接端不能仅凭 schema 名称猜测 camera 支持。

运行中修改输入格式必须更新 format_id，并同步完整声明；参考系或标定变更第一版要求停止并 reset，不在已有动作 chunk 中途切换。

## 与现有代码的衔接

| 位置 | 当前能力 | 后续实现 |
| --- | --- | --- |
| [server/client.py](../interface/sharpa_interface/server/client.py) | ServerConfig、connect、reset、infer_action | 加载连接配置并检查服务能力 |
| [server/metadata.py](../interface/sharpa_interface/server/metadata.py) | metadata_format 和服务 metadata 校验 | 增加输出声明、执行默认值、EEF frame 校验 |
| [server/observation.py](../interface/sharpa_interface/server/observation.py) 与 [buffers.py](../interface/sharpa_interface/server/buffers.py) | 观测校验和历史组装 | 输入 frame 与时间标定关联 |
| [server/action.py](../interface/sharpa_interface/server/action.py) | action 解析，默认要求 absolute | 输出 frame 校验及请求变换快照关联 |
| 本体 adapter | 现有 canonical 与硬件坐标转换 | root/camera 转换和标定有效性检查 |
| 模型服务 | 本仓库接口层提供客户端契约 | 实现配置加载、模型适配和四个服务入口 |

当前接口文档仍固定 canonical，增加可选 YAML 字段本身不会改变运行时行为。落地时需升级 metadata/observation/action 中受影响的 schema，定义完整 wire 字段，并让旧客户端拒绝不支持的版本。旧协议继续保留原有 canonical 语义；不能将 camera 数值标记成旧 schema 发送。

实现验证应覆盖 root/camera 位姿往返、移动相机请求快照、缺失标定拒绝、无 EEF 时字段省略、metadata 版本兼容，以及执行切片与预取边界。
