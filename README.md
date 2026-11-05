# UR + SharpA Deploy

双 UR5＋SharpA 工作台。数采、远端 Policy、Replay 共用同一套 ROS 2 Humble 节点。
本仓库只维护这套设备；PND、Astribot 和旧实验代码已归档移出。

## 目录

```text
Deploy/
├── src/          # ROS 节点、共享运动学、消息；网页资源随 ur_common 打包
├── interface/    # 远端模型协议、校验、观测缓冲与执行门
├── config/       # 硬件保护、远端模型配置
├── scripts/      # 构建、启动及维护工具
├── external/     # 本机厂商 SDK 与 Python 绑定（不提交二进制）
├── docs/         # 架构、操作、协议与安装说明
├── data/         # episodes 与 demo_recordings（不提交采集数据）
└── runtime/      # build、install、log、配置记忆、缓存与临时文件（不提交）
```

## 构建与启动

```bash
cd /home/yns/deploy
bash scripts/build.sh --executor sequential
bash scripts/start.sh
```

打开 <http://127.0.0.1:8088/>。启动会连接硬件，不会自动开始采集、Policy 或 Replay。
启动前应停止已有实例，避免重复占用硬件和端口。仅检查参数：

```bash
bash scripts/start.sh --show-args
```

默认数据位于 `data/episodes`，Demo 位于 `data/demo_recordings`，执行日志位于
`runtime/log/execution`。`source scripts/setup_env.sh` 将 ROS 日志、临时文件、缓存和
操作设置统一放到 `runtime`，并加载本仓库的 `runtime/install`。

- [工作台操作](docs/workbench.md)
- [架构](docs/architecture.md) · [文件组织与迁移](docs/layout.md)
- [系统依赖](docs/system_deps.md) · [厂商 SDK](external/README.md)
- [远端模型配置](config/README.md) · [协议](docs/interface_contract.md)
- [四个 sim-real DFM checkpoint 的部署与测试](docs/dfm_policy.md)
- [推理配置设计（未接入运行时）](docs/inference_config.md)

## 验证

```bash
source scripts/setup_env.sh
python3 -m pytest interface/test src -q -o cache_dir=runtime/cache/pytest --basetemp=runtime/tmp/pytest
node src/nodes/dashboard_node/test/workbench_ui_test.cjs
node src/nodes/dashboard_node/test/workbench_remote_ui_test.cjs
```

模型在 BAAI 或 4090（SSH 别名 `ps`）运行，本机不存放模型 checkpoint。
HIL 核心保留、默认关闭，网页隐藏 HIL 顶部入口。
