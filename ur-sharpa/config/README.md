# 远端模型服务配置

`model_servers.json` 将页面中的 BAAI / 4090 选择映射到 SSH 主机和远端代码目录。
4090 使用现有 SSH alias `ps`。当前两端 `repo_root` 都为空：未确定模型位置前不假设
任何 checkout/checkpoint，不发起 catalog SSH 请求，也不允许加载模型。

本机覆盖配置建议放到 `model_servers.local.json`，launch 会优先读取它；该文件不入库。
下例所有 `/absolute/...` 都是需要操作者替换的路径，不是现有模型位置：

```json
{
  "schema": "ur_sharpa.model_servers.v1",
  "default": "baai",
  "servers": {
    "baai": {
      "label": "BAAI",
      "ssh_host": "BAAI",
      "repo_root": "/absolute/remote/checkout",
      "asset_root": "/absolute/remote/assets",
      "task_config_root": "/absolute/remote/tasks",
      "env_file": "/absolute/remote/environment.sh",
      "remote_host": "127.0.0.1",
      "port": 5500
    },
    "4090": {
      "label": "4090",
      "ssh_host": "ps",
      "repo_root": "",
      "port": 5500
    }
  }
}
```

`repo_root` 是远端代码 checkout，必须是绝对路径。`asset_root` 默认等于它；
`task_config_root` 默认是 `<repo_root>/scripts/deploy/task_config`。
`env_file` 可省略，设置时会在启动 launcher 前 `source` 并导出变量。
SSH 使用 BatchMode，需预先配置 alias、认证和主机信任；远端需有 Python 3、tmux、curl。
页面只允许在模型停止且没有本实例遗留会话/启动任务时切换 server。

## 远端目录约定

沿用原有部署接口，模型枚举读取 `<repo_root>/scripts/deploy/*/launch.sh`，任务读取
`<task_config_root>/*/task.json`。没有本机模型别名表。任务目录以 `ur_` 开头；每个
`task.json` 使用已有 `sharpa.deploy.task_config.v2` 结构，包含 `task` 和扁平 `runs`：

```json
{
  "schema": "sharpa.deploy.task_config.v2",
  "task": {"id": "ur_example", "prompt": "example task", "dataset": "datasets/example"},
  "runs": [{
    "id": "example",
    "run_id": "unique-immutable-run-id",
    "model": "example_model",
    "checkpoint": "checkpoints/example",
    "status": "complete",
    "deployment": {"eligible": true}
  }]
}
```

`model` 必须精确对应 launcher 目录名，`run_id` 应保持唯一且稳定；checkpoint 必须是
存在的目录，相对路径基于 `asset_root`。只有 UR 任务、launcher 存在、checkpoint 存在、
`deployment.eligible=true` 且 status 为 `complete` 或 `in_progress` 的 run 才可启动。
远端执行：

```bash
bash scripts/deploy/<model>/launch.sh <task_name> <run_id>
```

传给 launcher 的环境变量为 `SHARPA_DEPLOY_ASSET_ROOT`、
`SHARPA_DEPLOY_TASK_CONFIG_ROOT` 和 `SHARPA_DEPLOY_PORT`。launcher 必须遵守
[interface contract](../../docs/interface_contract.md)，并在配置端口提供 `/healthz`
健康检查与原有模型服务协议；应在 tmux 前台运行，避免自行 daemonize 后脱离生命周期管理。

每个 Dashboard 实例使用独立 tmux 会话。端口占用时直接报错，不杀占用进程，也不自动换端口。
卸载和关闭只清理本实例的会话；切换 server 同时更新 `/model/status`，使 policy client
切换 SSH 隧道目标。当前尚无可用模型位置，因此仓库测试使用 mock，不代表真实模型已验证。
