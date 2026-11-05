# Sim-real DFM Policy 部署

工作台通过原有远端 Policy 链路测试四个最终训练 checkpoint：

| 页面名称 | 训练 run | checkpoint |
| --- | --- | --- |
| 50 real + 500 sim | `flow-duck-sim500-real50-4gpu-10k-20261104-eval500-sim1000` | `checkpoint-10000` |
| 50 real + 1k sim | `flow-duck-sim1k-real50-4gpu-10k-20261104-eval500-sim1000` | `checkpoint-10000` |
| 50 real + 2k sim | `flow-duck-sim2k-real50-4gpu-10k-20261104-eval500-sim1000` | `checkpoint-10000` |
| 50 real + 4k sim | `flow-duck-sim4k-real50-4gpu-10k-20261104-eval500-sim1000` | `checkpoint-10000` |

清单位于 [task.json](../config/model_tasks/ur_duck_sim_real/task.json)。不带
`eval500-sim1000` 后缀的同名实验是 step 10 的 smoke，不能作为这四个最终模型加载。
模型输入为 Ego、左右腕三路 RGB、62D 物理状态；生成频率和 horizon 固定为保存的
30 Hz / 30 步。默认完整执行 30 步，`rtc=0`。准确任务文本为
`Pick up the yellow ducks and place them on the pink plate.`。

## 已配置的推理主机

`config/model_servers.local.json` 是不入库的主机配置，已配置：

| 页面服务器 | SSH alias | DFM checkout | 部署 bundle |
| --- | --- | --- | --- |
| BAAI | `BAAI` | `/share/project/yenaisheng/DFM` | `<checkout>/logs/deploy-runtime` |
| 4090 | `ps` | `/home/ps/DFM` | `<checkout>/logs/deploy-runtime` |

两端使用同一相对 checkpoint 路径：`logs/<run>/checkpoints/checkpoint-10000`。
4090 清单要求 checkpoint 的 `.download_verified.json` 中 `verified=true`，避免下载
中的目录被加载。BAAI 直接使用完整训练产物。本机 UR 工作站不保存或加载模型权重。

部署 bundle 由此仓库的 adapter、launcher、task catalog 和标定组成。修改后在 UR 上重新
复制，操作不会启动模型或硬件：

```bash
cd /home/yns/deploy
python3 scripts/prepare_dfm_runtime.py \
  --host BAAI --dfm-root /share/project/yenaisheng/DFM \
  --python /share/project/yenaisheng/miniconda3/envs/dfm/bin/python \
  --calibration runtime/config/dfm_calibration.json
python3 scripts/prepare_dfm_runtime.py \
  --host ps --dfm-root /home/ps/DFM \
  --python /home/ps/.venvs/dfm-serving/bin/python \
  --calibration runtime/config/dfm_calibration.json --require-download-verification
```

推理环境需要最新 DFM 及 `interface` 的 `dfm_serving` extras（aiohttp、Pillow、MsgPack）。
bundle 的 `environment.sh` 指定 Python 和 DFM 模块目录。
检查某个 checkpoint 的配置与标定且不加载模型：

```bash
cd /share/project/yenaisheng/DFM/logs/deploy-runtime
source environment.sh
PYTHONPATH="$PWD/interface:$PYTHONPATH" "$SHARPA_DEPLOY_PYTHON" \
  -m sharpa_interface.dfm.server ur_duck_sim_real \
  flow-duck-sim500-real50-4gpu-10k-20261104-eval500-sim1000 \
  --asset-root /share/project/yenaisheng/DFM \
  --task-config-root "$PWD/config/model_tasks" \
  --calibration "$PWD/config/dfm_calibration.json" --check
```

## 物理接口

DFM adapter 在 `interface/sharpa_interface/dfm`，直接调用当前
`dfm.inference.factory.load_policy`，由 DFM processor/Policy 包装负责图像预处理、
状态归一化、Flow 解码和相对动作还原。工作台仍接收原有 canonical wrist 的
`sharpa_policy_action.v5`；不会将相机坐标直接标记为 canonical。

`runtime/config/dfm_calibration.json` 显式记录训练 manifest/URDF 和当前 UR geometry 的
SHA，以及双侧 `model_pose = A @ canonical_pose @ B`。A 包含训练相机/UR base 链，B
处理当前部署 wrist 和训练 hand-root 的终端轴差异；动作使用精确逆变换。
输入手关节按 DFM 的 index/middle/pinky/ring/thumb 次序重排，输出还原 interface 的
thumb/index/middle/ring/pinky 次序。所有位置为米，关节为弧度，旋转使用 column Rot6D。

这份标定与已保存的真实训练标定及当前代码链经过数值和原始 episode 数据核对；没有
重新测量当前物理相机。如果调整了相机、机械安装或几何标定，须重新生成并核对这份
边界标定，再更新推理 bundle。

服务保留 `/healthz`、MsgPack `/metadata`、`/reset` 和 WebSocket `/infer`。
metadata 的 `session_owner_token_required=true` 扩展使客户端使用同一随机 owner token
绑定 WebSocket 和 HTTP reset；同一连接可重置到新的 rollout session，其他连接不能
清除其历史。断线、预测失败或协议错误后必须重新连接/reset，不能重放旧 action。
停止、过期结果、执行保护和 dry-run 仍由现有节点管理。

## 网页测试流程

1. 按 [工作台文档](workbench.md) 构建并启动，打开 `http://127.0.0.1:8088/`。
2. 切换到 Policy，选择 BAAI 或 4090，打开模型清单，选择 `ur_duck_sim_real` 和一个
   checkpoint。页面按真实训练 run_id 加载，并显示加载状态和远端日志。
3. 等待权重加载、两轮推理预热完成和客户端身份校验；远端日志显示 `warmup complete` 后才显示
   `DFM deployment ready`。冷启动的编译和 CUDA 初始化在加载阶段完成。
4. 完成相机、状态、标定和复位条件后，先进行页面预览；预览沿用原规划链并设置
   `dry_run=true`，驱动不下发。确认后由操作者开始正式执行。
5. 停止执行、卸载当前模型，再选择另一 checkpoint；比较时使用相同起始条件和任务。

模型加载不启动机器人执行。独立模型 RPC smoke 只证明输入/输出和传输正确，不能证明
真实机器人动作或四个模型的任务成功率。执行 trace 位于 `runtime/log/execution`。


## 加载与预热

模型进程加载权重后，先在专用的同一个推理 worker 上执行两轮内部预测，覆盖新 episode 的
完整 chunk 和下一轮 cache append。真实请求、reset 和 invalidate 后续都使用这个 worker，
保留已准备的 CUDA/Triton 执行上下文。预热全部结束并清除临时 episode 后才提供 ready
健康状态和 `DFM deployment ready` 日志。预热失败会使启动失败，不会报告模型已就绪。

输入是按 checkpoint 保存的三相机像素预算生成的确定 RGB 和合法物理 pose；当前16:9
Ego/腕相机的 resize grids 与真实输入一致。使用 bundle 中已加载的真实标定完成原有 wire
转换；不读取采集 dataset、不连接机器人、不发执行命令。两轮结果只在进程内丢弃。完成或
失败后都会 reset、恢复 seed 42、清除生成历史和 client session，真实 episode 仍从第一个
观测开始。预热请求不进入网页的用户请求数、平均耗时或执行日志。

日志分别显示 `DFM model loaded`、`DFM inference warmup started/complete` 及预热耗时；
`/healthz` 的 `warmup` 字段提供两轮耗时和 context 诊断。它们是启动诊断，与真正请求的
RPC latency 分开。修改本节对应的 adapter 后仍须重新运行上面的 bundle 准备命令；准备
bundle 不启动模型或硬件，已经运行的模型不会热替换。


4090 的独立纯模型验收使用同一 checkpoint 和两份真实三相机观测：权重加载约7.32秒、
启动预热约11.73秒；ready 后第一请求从旧服务的11.677秒变为0.161秒，第二请求0.132秒，
reset 后新 rollout 第一请求0.150秒。冷初始化耗时仍属于启动时间，加载页会等待它完成。
这些数字是模型 RPC 测试，不代表真机任务效果。

fresh 同一模型进程、同一 worker 中，先执行冷真实请求，再 reset→两轮预热→reset 后
重复三个真实请求，输出逐元素完全相同；seed state 恢复，cache/reference/session 已清空。
两轮预热和真实请求的 context_tokens 为193/372，next_rope_position 为96/178。
独立的旧/新模型进程之间另有小幅数值差异：xyz 最大0.000473米、Rot6D 最大0.002595、
手关节最大0.001078弧度；这项比较不具有跨进程 bitwise 等价结论，具体 CUDA/backend
差异尚未单独定位。上述同进程回归用于验证预热的 seed/history 恢复。
