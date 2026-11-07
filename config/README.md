# 推理机器与任务配置

`model_servers.json` 定义 BAAI、UR (4090D) 和额外的 PS (4090)。仓库默认不填模型目录；
未配置时页面列出机器，但不扫描模型或允许加载。工作台优先读取不入库的
`model_servers.local.json`，也可通过 launch 的 `model_config` 指定。

页面按「机器 → 任务 → checkpoint」选择。切换机器要求当前模型已停止，且没有本实例
遗留的启动任务或模型会话。切换会清空任务选择，更新 Policy 连接目标；目录请求和加载
请求绑定所选机器和页面显示的checkpoint路径/有效文本；清单变化时要求重新选择，旧异步
结果不能恢复另一台机器的模型项。

一个机器 profile 示例：

```json
{
  "label": "UR (4090D)",
  "local": true,
  "ssh_host": "",
  "repo_root": "/home/yns/deploy/runtime/dfm-source/logs/deploy-runtime",
  "asset_root": "/home/yns/deploy/runtime/dfm-source",
  "task_config_root": "/home/yns/deploy/runtime/dfm-source/logs/deploy-runtime/config/model_tasks",
  "env_file": "/home/yns/deploy/runtime/dfm-source/logs/deploy-runtime/environment.sh",
  "remote_host": "127.0.0.1",
  "port": 5500,
  "require_download_verification": false
}
```

UR 使用 `local=true`，模型进程在工作台本机运行。BAAI 与 PS 使用 `local=false` 和
已配置的 SSH alias；需预先完成认证和主机信任。机器需有 Python 3、tmux 和 curl。
`repo_root` 是含 `scripts/deploy` 的部署 bundle，`asset_root` 是模型资产根，
`task_config_root` 是 task catalog 根。路径均为机器上的绝对路径；`env_file` 在
launcher 前加载。DFM 源、权重和校验记录分别准备，默认不设源代码或模型 hash 加载门禁。
原生 DFM bundle 由 `scripts/prepare_dfm_runtime.py` 准备，必须传入 `--calibration` 和
`--data-contract`。当前八个 run 的部署依赖包括 `config/dfm_calibration.json`、
`config/real101_data_contract.json`、原生环境转换、单模型 worker 和服务入口。

## 任务与 checkpoint

枚举读取 `<repo_root>/scripts/deploy/*/launch.sh` 和
`<task_config_root>/*/task.json`。任务目录以 `ur_` 开头，结构为
`sharpa.deploy.task_config.v2`：

```json
{
  "schema": "sharpa.deploy.task_config.v2",
  "task": {"id": "ur_example", "label": "示例任务", "prompt": "任务默认文本", "dataset": ""},
  "runs": [{
    "id": "example",
    "run_id": "unique-immutable-run-id",
    "model": "dfm_flow",
    "checkpoint": "logs/example/checkpoints/checkpoint-40000",
    "prompt": "本 checkpoint 的任务文本",
    "status": "complete",
    "deployment": {"eligible": true, "required_files": ["environment.sh"]}
  }]
}
```

每个 run 的有效文本为自身 `prompt`，未提供时使用所属任务的 `prompt`。显式空文本会
禁用加载，不回退到别的任务。页面显示有效文本，model status 与模型客户端使用同一文本
核对加载身份。DFM 原生连接同时精确核对 model、task、run 和 checkpoint 路径；显式
恢复连接仍使用这一身份。不同任务可以使用不同文本；同一任务的各 checkpoint 也可分别指定。

`model` 精确对应 launcher 目录名；`run_id` 唯一且稳定，checkpoint 相对路径基于
`asset_root`。只有 UR任务、launcher/权重目录/部署依赖存在、任务文本有效、
`deployment.eligible=true` 且状态为 `complete` 或 `in_progress` 的 run 才允许启动。

启动仍调用：

```bash
bash scripts/deploy/<model>/launch.sh <task_name> <run_id>
```

环境变量为 `SHARPA_DEPLOY_ASSET_ROOT`、`SHARPA_DEPLOY_TASK_CONFIG_ROOT` 和
`SHARPA_DEPLOY_PORT`。`dfm_flow` 通过 DFM 原生服务运行，两次预热及 seed/历史恢复
完成后才绑定 `/health`；其余模型保留已有协议和 `/healthz`。UR 的 ROS Python 3.10
客户端通过 IPC 使用隔离 Python 3.12 worker，再连接原生 InferenceClient/InferenceServer。
模型软件启动不启动机器人执行。每个 Dashboard 使用
独立 tmux 会话；端口占用直接报错，卸载只清理本实例创建的会话。

当前八组 duck checkpoint 和机器路径见 [DFM部署文档](../docs/dfm_policy.md)。

当前黄鸭任务包含七个 real101／sim-real checkpoint 和一个纯仿真 8k checkpoint，均为
40000 步。纯仿真组显式覆盖为仿真训练文本，不能继承含真实数据组的 pink plate 文本。

本轮八个 checkpoint 仅下载到 PS；BAAI 保留训练产物，UR 本机权重已移除。测试选择
PS (4090) 或 BAAI。UR 的机器配置和软件仍保留，枚举时缺少 checkpoint 的八项会禁用加载。
