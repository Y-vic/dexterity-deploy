# SharpA interface and embodiment deploy

本 repo 定义 SharpA policy server 与机器人之间的共享 dict 接口，并为 PND、UR
保留各自的 ROS、IK 和硬件实现。

```text
Deploy/
├── assets/          # 共用 URDF、mesh 和 MuJoCo 资源
├── interface/       # schema、validator、transport、buffers、execution gate
├── pnd-sharpa/      # PND ROS、Adam IK、SharpA、TCP/RTP
├── ur-sharpa/       # UR 工作台、采集、Replay、SSH 远端推理
└── docs/
```

UR 的当前入口是 [UR Deploy 工作台](ur-sharpa/README.md)：数采、Policy、Replay 共用
一套硬件节点。Policy 选择 SSH BAAI 或 4090（`ps`），模型位置尚未配置，参见
[模型服务配置](ur-sharpa/config/README.md)。HIL 核心保留、默认关闭，顶部入口隐藏。
PND 的独立部署实现保持原有结构。

## 共享协议

`policy_client` 负责 server metadata、观测 buffers、session/request 身份及有效执行
slice；本体适配器负责坐标变换、IK、命令发布和硬件保护。模型内部的 relative EEF
必须由服务端转换为 `eef_def="absolute"` 后返回。

`execution.rtc` 是现有协议的提前 fetch 步数：0 表示完成后请求，正数表示还剩指定步数
时预取下一次推理。它不表示向模型拼接冻结前缀；本次 UR 更新没有增加本地 RTC 推理或
修改模型训练假设。同一时间最多一个推理请求在途，停止/切换后的过期结果不执行。

完整 dict 定义见 [interface_contract.md](docs/interface_contract.md)，节点分工见
[architecture.md](docs/architecture.md)。列出的必需 key 不能缺失，允许增加扩展 key；
只有声明为 nullable 的值允许为 `None`。

模型启动、输入输出声明及 root/camera EEF 的后续配置设计见
[推理接口与部署配置设计](docs/inference_config.md)（尚未接入运行时）。

## 验证 interface

```bash
PYTHONPATH=interface python3 -m pytest interface/test -q
```

Python 入口：

```python
from sharpa_interface.server import (
    PolicyInputBuffers,
    ServerClient,
    SyncExecutionGate,
    parse_policy_action,
)
```

URDF、mesh 和 MuJoCo 资源放在 `assets/`；本体目录通过相对软链接引入公共
描述 package。机器人 SDK 安装方式见各本体的说明。
