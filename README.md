# SharpA interface and embodiment deploy

本 repo 定义 SharpA policy server 与机器人之间唯一的 dict 接口，并为 PND、UR
保留各自的 ROS、IK 和硬件实现。

```text
Deploy-v2/
├── assets/          # 跨 embodiment 共用的 URDF、mesh 和 MuJoCo 资源
├── interface/       # dict schema、validator、transport、buffers、sync execution
├── pnd-sharpa/      # PND ROS、Adam IK、SharpA、TCP/RTP
├── ur-sharpa/       # UR ROS、UR IK、RTDE、SharpA
├── contract_tests/  # server/client 共用契约测试
└── docs/
```

URDF、mesh 和 MuJoCo 等模型资源只在 `assets/` 保存一份。embodiment
目录只保留 ROS/hardware adapter，通过相对软链接引入公共 description
package。

## 唯一运行闭环

```text
ROS facts ──> policy_client buffers ──> policy server
                                          │
                                          ▼
action_execute done <── action_execute <── action_ik <── executable slice
          │
          └──────── policy_client fetch next action
```

- `policy_client` 是唯一连接 server 的 node，拥有 metadata、session、request、
  七路 buffer 和 synchronous 状态机。
- server action 越过 interface 时，EEF 必须是 `eef_def="absolute"`；模型内部
  relative EEF 由 server policy adapter 转换。
- `policy_client` 只执行 `execute_start:execute_start+execute_length`。
- `action_execute` 只执行 joint plan 并发布匹配 action 身份的 `execution_done`。
- async 当前明确不支持。
- workstation 不按模型切 launch；输入窗口和执行参数来自 server metadata/action。

完整 dict 定义见 [interface_contract.md](docs/interface_contract.md)，节点分工见
[architecture.md](docs/architecture.md)。

## 使用 interface

```bash
PYTHONPATH=interface python3 -m pytest contract_tests -q
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

所有 interface dict 遵守同一条扩展规则：列出的 key 不能缺失，允许增加 key；
声明为 nullable 的 value 可以为 `None`。
