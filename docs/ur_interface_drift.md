# UR 并行副本漂移分析

对象：`ur-sharpa/src/nodes/sharpa_policy_v3_client/sharpa_policy_v3_client/`（下称 **U**）
与 `interface/sharpa_interface/`（下称 **I**，模块位于 `server/`）。

分类：**A** 本体真实需求（保留，迁入 UR 侧 adapter 如 `ur_common`，不进 frozen
interface）／**B** 陈旧遗留（可删）／**C** 违反 contract（潜在 bug）／**D** interface
缺失的真实改进（需 schema 变更签核，单列为提案，不静默移植）。

## 执行摘要

| 类别 | 数量 |
| --- | --- |
| A 本体需求 | 7 |
| B 陈旧遗留 | 4 |
| C 契约违反 | 12 |
| D 提案 | 18 |
| 合计 | 41 |

`serialization.py` 除 docstring／换行／尾逗号外 token 级完全一致，0 项行为差异，可直接删除。

### 最高风险（按顺序）

1. **`executed_steps` 为 slice-local，非累计。** U 的 `action_node.py:295` 把
   `hardware_execution.py:205,265` 从 0 开始的计数直接写入 feedback；
   `policy_client_node.py:629-632` 也按 `execute_length` 校验，而非
   `execute_start + execute_length`。这正是
   `server/executed_steps.py:8-9` 点名的违反项。迁移到
   `SyncExecutionGate.complete()`（`execution.py:206-209` 按 `execute_stop` 比较）后，
   任何 `execute_start > 0` 的 slice 会立刻抛
   `execution_done.executed_steps exceeds the active slice`。**先修此项，否则接入即失败。**
2. **U 不履行 ExecutableAction 截断契约。** `policy_client_node.py:783-788` 发布
   `action_length` 全长数组，截断下沉到 `hardware_execution.py:150-152`。契约要求
   `execution.execute_length` 等于 action arrays 第一维
   （`docs/interface_contract.md:144`，实现见 `action.py:86-124`）。切到
   `ParsedPolicyAction.execution_command()` 后数组首维变成 `execute_length`，
   `hardware_execution.py:144` 的 `(action_length, 9)` 形状断言必然失败。
3. **闭集 key 校验普遍违反「允许添加额外 key」。** `metadata.py:33-35`、
   `observation.py:105-110`、`transport.py:453,463` 对未知 key 直接报错，
   `metadata.py:169,232` 还在归一化时丢弃扩展 key。I 侧一律用
   `_required`／`_object` 只检查缺失（`metadata.py:39-41`）。任何 server 端新增
   诊断字段都会让 U 拒绝合法响应。

同级需注意：U buffers 的 RLock 线程安全与 byte 预算（D-B3／D-B2）在 I 侧完全缺失，
而 UR 用的是 ROS 多线程 executor，直接换用 `TemporalBuffer` 会引入数据竞争。

### 第三份副本

`/home/yns/deploy/sharpa_policy_v3_ws/.../` 严格更旧，**没有任何 Deploy-v2 副本丢失的修复**：
`action.py:15` 为 `sharpa_policy_action.v3`、带 `wrist_action_type`／44 维
`hand_joint`／`relative_eef`；`metadata.py:13` 为 v3；`observation.py:457` 等仍写
`"eef_frame": "robot_base"` 而非 `eef_def`；`mock_server.py:228` 同为 v3。
Deploy-v2 副本已把这些前推到 v4 与 `eef_def`。**结论：该副本可整体忽略。**

## observation.py（I 235 / U 852）

**不是同一角色。** I 的 `observation.py` 只是 validator（`validate_policy_observation`，
`observation.py:19`）；U 的是 builder（`build_observation`，`observation.py:711`）。
I 的 builder 在 `buffers.py:105` 的 `PolicyInputBuffers.build_observation`，且它在
`buffers.py:142` 出口调用 validator 自校验——U 的 builder 无此自校验。行数差主要来自
角色不同，非漂移。

| # | 差异 | 类 | 依据 |
| --- | --- | --- | --- |
| O1 | `execution_feedback` 要求 key 集合**恰好相等**，多一个 key 即拒 | **C** | `observation.py:105-110` vs I `execution.py:46` 只查缺失 |
| O2 | `validate_format_capacity()`：按 metadata 预检 frame 容量／buffer 字节／最小 wire 字节，抛带 required/available 的 `ObservationCapacityError` | **D** | `observation.py:184-296`；I 无对应物 |
| O3 | `ObservationNotReady(path, required, available)` 类型化欠数据错误 | **D** | `observation.py:35-45,131-145`；I `buffers.py:56` 只抛裸 `RuntimeError`。`state_node.py:519` 依赖此类型分流 |
| O4 | build 时预检 `max_message_size`，超限抛 `ObservationCapacityError` | **D** | `observation.py:795-813` |
| O5 | `_validate_state_joint_widths` 显式检测 history 内 joint 形状变化并报 timestamp | **B** | `observation.py:398-421`；I 侧 `np.stack`（`buffers.py:207`）+ `_optional_float_array`（`observation.py:157`）已等价覆盖，仅错误类型不同 |
| O6 | `_wire_int` 上界 `INT64_MAX` | **D** | `observation.py:83`；I `observation.py:222` 仅要求 >= 0 |
| O7 | eef 被请求时无条件写 `"eef_def": "absolute"` | **A** | `observation.py:457,478,534,555`。合法性来自 UR 约定 `StateFrame.eef_frame == "robot_base"`（`buffers.py:244`）即绝对系；该「robot_base ⇒ absolute」映射事实须记入 `ur_common` |
| O8 | I validator 额外校验 history 时间戳单调、`current` 不早于 history | — | I `observation.py:81,85,122`；U builder 无此检查，但迁移后由 I 补上，属收益 |

`sharpa_policy_observation.v3` 两侧一致（`observation.py:28` vs `metadata.py:15`）。
shape/dtype 两侧一致：hand 22（`observation.py:147` / U `buffers.py:214`）、
wrench (5,6)（`observation.py:59` / U `buffers.py:329`）、
deformation uint8 (5,240,240)（`observation.py:65` / U `buffers.py:390`）、
EEF (9,)（`observation.py:136` / U `buffers.py:201`）。**无 shape 漂移。**

## transport.py（I 820 / U 586）

**架构级不同，不是漂移。** I 是同步 + 手写 WebSocket 帧 + SSH relay
（`transport.py:123,287,504`）；U 是 aiohttp 异步（`transport.py:163`）。
U 的 ROS node 全异步，直接换用 I 的同步 `SharpaV3PolicyClient` 是设计变更，不是删除。

| # | 差异 | 类 | 依据 |
| --- | --- | --- | --- |
| T1 | 异步 API（`async def infer/reset/connect/close`） | **A** | U `transport.py:482,368,402,563`；I `transport.py:772` 同步。迁移须在 UR 侧包线程或 interface 增异步实现 |
| T2 | 单 inflight 断言 | **D** | U `transport.py:223-225,486`；I 无，仅靠 `SyncExecutionGate` 间接约束 |
| T3 | `GET /healthz` 前置探活 | **D** | U `transport.py:336-361`；I 只在 SSH relay 内探活（`transport.py:63-70`） |
| T4 | `max_message_size` 协商 `min(local, metadata)`，且连接中禁止重协商 | **D** | U `transport.py:323-333`；I `client.py:118-126` 固定构造值，`metadata.py:157` 校验后**从不下发给 transport** |
| T5 | 分离 http／connect／inference／heartbeat 超时 | **D** | U `transport.py:170-174`；I 单一 `timeout_s=60.0`（`transport.py:716`） |
| T6 | 超时／协议错误即 `_drop_websocket()`，下次快速失败 | **D** | U `transport.py:514,522,527`；I 出错后 `SimpleWebSocket` 状态未知 |
| T7 | 校验响应 `Content-Type == application/msgpack` | **D** | U `transport.py:287-292`；I 只发不校验（`transport.py:678`） |
| T8 | 分层异常（`PolicyStateError`／`PolicyConcurrencyError`／`PolicyProtocolError`／`PolicyTimeoutError`／`PolicyMessageTooLargeError`／`PolicyServerError`） | **D** | U `transport.py:31-95`；I 只有 `PolicyHttpError` + 裸 `RuntimeError`／`ConnectionError`／`TypeError`（`transport.py:810,813`）。`policy_client_node.py` 依赖此分层 |
| T9 | error 响应要求 key 集合恰好相等 | **C** | U `transport.py:453,463`；I `action.py:326` 可扩展 |
| T10 | HTTP body 分块读取 + 提前按 `content_length` 截断 | **D** | U `transport.py:242-255` |
| T11 | SSH relay／chunked HTTP／手写掩码帧 | — | 仅 I 有（`transport.py:38-91,570-612,453-467`）；UR 未用 SSH，属 I 侧多余能力，非 UR 丢失 |

两侧 URL 推导等价（U `transport.py:107-141` vs I `transport.py:504-523`），
U 额外限制 path 白名单（`transport.py:127`），差异可忽略。**两侧都没有重试／退避。**

## buffers.py（I 207 / U 707）

I 的 `TemporalBuffer` 是 dict 缓冲；U 是类型化 dataclass frame 缓冲。
七路固定容量与默认值两侧一致：camera 3／state 19／tau 19／wrench 19／deformation 3
（I `buffers.py:74-82` vs U `buffers.py:618-627`）。history/current 选择语义一致：
`count = history_len + int(current)`、current 取最后一帧、history 不含 current
（I `buffers.py:54,59-62` vs U `buffers.py:543,552-554`）。**无容量或选择语义漂移。**

| # | 差异 | 类 | 依据 |
| --- | --- | --- | --- |
| B1 | 乱序插入：按 timestamp `bisect` 插入并按 timestamp 去重／替换 | **D** | U `buffers.py:508-519`；I `buffers.py:47-48` 对非单调 push 直接 `ValueError`。ROS 多 producer 异步到达下这是实质能力差 |
| B2 | byte 容量 + frame 容量**双重驱逐** | **D** | U `buffers.py:557-564,502-506`；I 只有 `deque(maxlen)` |
| B3 | 每 buffer `threading.RLock` + `ObservationBuffers.locked()` 跨 buffer 一致快照 | **D** | U `buffers.py:470,654-657`，build 全程持锁（`observation.py:747`）；I 无任何锁 |
| B4 | push 时即校验 dtype/shape/有限性（tau (22,)、wrench (5,6)、deformation uint8 (5,240,240)、eef (9,)、hand (22,)） | **D** | U `buffers.py:274,329,390,201,211`；I 推迟到 build 出口（`observation.py:58-67`）。失败点更早、更易定位 |
| B5 | `StateFrame` 强制 `eef_frame == "robot_base"`（有 eef 时），无 eef 时必须为 None | **A**（含 **B** 成分） | U `buffers.py:234-247`，producer 侧 `state_node.py:263,280,287`。UR 坐标约定属 A，须迁入 `ur_common`；但字段名本身是 `eef_frame` 旧 schema 残留（第三副本 `observation.py:457` 仍外发该名）。**分类不完全确定**：建议保留语义检查、改名为本体内部字段，不要把 `eef_frame` 推入 interface |
| B6 | state EEF 的 rot6d 退化／共线校验 | **D** | U `buffers.py:119-133,226-229`；I 两侧 observation 与 action 均无 rot6d 校验 |
| B7 | 无效 camera frame 必须 `data` 为空 | **D** | U `buffers.py:162-163`；I 只禁「valid 且空」（`observation.py:96`） |
| B8 | `BufferUnderflowError(required, available)` | **D** | 同 O3，同一项，不重复计数 |
| B9 | `append()` 返回该帧是否仍被保留 | **D** | U `buffers.py:494,521-525`；I `push()` 返回 None |
| B10 | `clear_source(name)` 单路清空 | **D** | U `buffers.py:685-698`；I 只有全量 `clear()`（`buffers.py:98`） |

I 侧独有：push 时 deepcopy 隔离（`buffers.py:49`）；U 用 `copy()+setflags(write=False)`
（`buffers.py:88-90`）达到同等不可变性。等价，非差异。

## action.py（I 376 / U 452）

两侧 `ACTION_SCHEMA` 均为 `sharpa_policy_action.v4`（I `action.py:16` / U `action.py:15`）。

| # | 差异 | 类 | 依据 |
| --- | --- | --- | --- |
| A1 | `hand_joint` 左右拼成 44 维单数组 | **B** | U `action.py:370`、`HAND_JOINT_DIMENSION = 44`（`action.py:17`）。v3 形状遗留；v4 为 `{left, right}` 各 22（`docs/interface_contract.md:100-103`，I `action.py:226-239`）。注意 `hardware_execution.py:151` 仍按 44 维消费，删除前须改为消费 I 的 `HandAction` |
| A2 | `hand_joint` 左右**都不可为 None** | **C** | U `action.py:360-365` 无 None 分支；契约与 I `action.py:227` 允许 `None` |
| A3 | `diagnostics` 强制为含 4 个字段的对象，`policy_family`／`checkpoint_id` 必须非空 | **C** | U `action.py:387-418`；契约为 `dict | None`（`docs/interface_contract.md:117`），I `action.py:262-264` 允许 None |
| A4 | `next_metadata_format` 只做 `isinstance(dict)` + deepcopy，不做 schema 校验 | **C** | U `action.py:420-432`；契约要求非 None 时必须是完整格式（`docs/interface_contract.md:32`），I `action.py:266-268` 调 `validate_metadata_format` |
| A5 | 要求 joint／eef 至少一个非 None | **C**（更严，方向可取） | U `action.py:325-328`；I `action.py:292-293` 允许两者皆 None。技术上拒绝了合法负载，但更严格更安全，建议作为 D 提案上推而非在 UR 侧保留 |
| A6 | joint 存在时**优先返回 joint 并跳过 `eef_def` 校验** | **C** | U `action.py:329-336`：同时带 joint 与 relative eef 时 relative 被静默丢弃且不报错；I `action.py:213-219` 对任何非 None 的 eef 都校验 `eef_def` |
| A7 | 每腕折叠为单一数组 + `wrist_action_type` 字符串 | **B** | U `action.py:57-58,313-347`；v4 为 `{joint, eef, eef_def}` 并存（I `action.py:54-59`）。`policy_client_node.py:781-786` 依赖该形状 |
| A8 | `relative_eef` 分支 | **B** | U `action.py:168` 与 `hardware_execution.py:140-149`／`hardware_geometry.py:347`。parser 已不产出该值（`action.py:341` 只返回 `"eef"`），为**不可达死代码**，且其「本地把 relative 转 absolute」的意图与 `docs/interface_contract.md:122` 相悖，应随删除一并清掉 |
| A9 | 无 `executable_action()`／`execution_command()`，不产出 `sharpa_executable_action.v1` | **C** | U 全文无该 schema；I `action.py:86-124`。见摘要第 2 条 |
| A10 | `expected_left/right_joint_dimension`（UR 6-DOF 臂维度校验） | **A** | U `action.py:194-216,175-178`；I 无。属本体约束，迁入 `ur_common` |
| A11 | eef 数组逐行 rot6d 退化／共线校验 | **D** | U `action.py:143-156,171`；I 无 |
| A12 | `expected_action_id`／`expected_revision` 可选比对 | **D** | U `action.py:193-194,260-267`；I `action.py:140-146` 只比对 session/request |
| A13 | server error 在 transport 层处理而非 parser 层 | — | U `transport.py:547-558` vs I `action.py:148-149`。位置不同，覆盖等价，非差异 |
| A14 | `_object` 默认 `allow_unknown=True` | — | U `action.py:81`，与契约可扩展一致。第三副本为 `False`（旧行为），Deploy-v2 已修 |

I 侧独有：`EEF_DEFINITIONS` 含 `"relative"`（`action.py:19`）并由
`expected_eef_def="absolute"` 在公共边界上拒绝（`action.py:145,213-219`）——即
「server 内部可 relative，出口必须 absolute」的落点。U 对 eef 路径也拒绝非 absolute
（`action.py:337-340`），但受 A6 影响存在绕过路径。

## metadata.py（I 192 / U 261）

schema 常量全部一致（`SERVER/OBSERVATION/ACTION/METADATA_FORMAT/RESET/TRANSPORT`，
I `metadata.py:14-19` vs U `metadata.py:11-16`）。行数差主要是格式化与
`_wrist_selector`／`_hand_selector` 提取（U `metadata.py:72-86`，I 用内联闭包
`metadata.py:93-106`），属噪声。

| # | 差异 | 类 | 依据 |
| --- | --- | --- | --- |
| M1 | `_object` 对未知 key 报错 | **C** | U `metadata.py:33-35`；I `metadata.py:39-41` 只查缺失。契约明确「允许添加额外 key」（`docs/interface_contract.md:3-4`） |
| M2 | 归一化只回传已知 key，**丢弃扩展 key** | **C** | U `metadata.py:67-69,141-152,169,232-236`；I 全部 `**dict(obj)` 保留（`metadata.py:69,126,145`） |
| M3 | `validate_reset_response` 要求 `request_id` 为整数 | **C** | U `metadata.py:243`；I 允许 None（`metadata.py:173-178`）且 `client.py:65-72`／`transport.py:781` 实际会传 None。server 回显 null 时 UR 直接拒绝——**互操作断裂** |

## session.py（I 46 / U 263）

**同名不同物，不是漂移对。** I 的 `Session`／`SessionRegistry`（`session.py:17,31`）管
revision 与 execute_start；U 的 `PolicySessionRuntime`（`session.py:25`）是异步生命周期
编排器。I 侧对应物是 `policy.py:23` 的 `PolicyClientCore` + `execution.py:148` 的
`SyncExecutionGate`。

| # | 差异 | 类 | 依据 |
| --- | --- | --- | --- |
| S1 | 异步生命周期：`start`／`reset`／`reconnect`／`infer_once`／`infer_observation`／`disconnect`／`close`，`asyncio.Lock` 串行化 | **A** | U `session.py:115-263,65`；I `PolicyClientCore` 同步且无 reconnect／disconnect／snapshot |
| S2 | metadata format 变更时重跑容量校验 | **D** | U `session.py:101-108,246-249`；I `policy.py:89-90` 直接赋值 |
| S3 | 送出前预检 observation 身份（schema／session／request／format_id） | **A** | U `session.py:229-239`。UR 的 observation 由 `state_node` 跨进程构建（`policy_client_node.py:233-247`），此预检是该架构的必需品；I 侧 observation 自建故不需要 |
| S4 | `snapshot()` 只读状态视图、`clock_ns` 注入、`auto_reset` 开关 | **D** | U `session.py:34,59,79-91` |
| S5 | 新 session_id 不得与当前相同 | **D** | U `session.py:110-113`；I `session.py:36-38` 无限制 |
| S6 | `request_id` 仅在解析成功后 +1 | **D**（指向 I 侧 bug） | U `session.py:251`；I `execution.py:166-167` 在推理前就 +1，而 `cancel_inference`（`execution.py:181-187`）**不回滚** `next_request_id`。严格 server（`mock_server.py:216-218` 要求严格递增）会在一次传输失败后拒绝后续请求 |
| — | `revision` 从不递增 | 缺口 | U 全程传 `revision=0`（`session.py:123,144,166`），只回显 server 值；I `session.py:23-27` 的 `bump_revision` 在 UR 侧无实现。UR 的 safe-stop／replan 应产生 revision——**迁移时新增，不是删除前必须保留的东西** |

## serialization.py（I 144 / U 174）

Token 级比对（去注释／docstring）仅 3 处差异，全部为空白与字符串拼接形式：
`pass` vs docstring（U `serialization.py:16,20`）、集合尾逗号（U `serialization.py:83`）、
错误信息跨行拼接（U `serialization.py:118-119`）。`MAX_MESSAGE_SIZE`、ndarray 编解码、
描述符白名单、大小上限行为完全相同。**0 项行为差异，直接删除。**

## 迁移检查单

删除 U 副本**之前**必须完成（顺序即依赖顺序）：

| # | 动作 | 阻塞项 |
| --- | --- | --- |
| 1 | 改 `action_node.py:295` 与 `hardware_execution.py:205,265` 用 `server/executed_steps.py:make_progress()` 产出 `execute_start + executed_in_slice`；同步改 `policy_client_node.py:629-632` 按 `execute_start + execute_length` 校验 | C·摘要 1 |
| 2 | 把 `contract_tests/test_executed_steps.py` 接到 UR feedback builder（该文件头注释已要求 Phase 2/3 接入） | 验证 1 |
| 3 | 让 `policy_client` 改用 `ParsedPolicyAction.execution_command()`，`hardware_execution.prepare_executable_action` 改为消费 `execute_length` 首维、去掉本地切片（`hardware_execution.py:150-152`） | C·A9 |
| 4 | 消费端从 44 维拼接数组改为 `HandAction{left,right}`（各 22），并处理 None（`hardware_execution.py:151`、`policy_client_node.py:787-788`） | C·A2 / B·A1 |
| 5 | 消费端从 `wrist_action_type` + 单数组改为 `WristAction{joint,eef,eef_def}`（`policy_client_node.py:781-786`、`hardware_execution.py:136-144`） | B·A7 |
| 6 | 删除 `relative_eef` 全部分支：`hardware_execution.py:140-149`、`hardware_geometry.py:347-361` | B·A8 |
| 7 | 新建 `ur_common`，迁入：`hardware_geometry.py` 全量（UR base／RTDE pose／rot6d 行列约定）、`eef_frame == "robot_base"` 语义（`buffers.py:234-247`、`state_node.py:263,280,287`）与「robot_base ⇒ absolute」映射说明、UR 6-DOF joint 维度期望（`action.py:194-216`）、`max_frequency_hz` 限制（`hardware_execution.py:131`） | A·B5/A10/O7 |
| 8 | 在 `ur_common` 中提供 buffers 适配层：线程安全、byte 预算、乱序去重、`ObservationNotReady`／`ObservationCapacityError`、`validate_format_capacity`；或先在 interface 立项 D 提案再迁 | D·B1/B2/B3、O2/O3/O4 |
| 9 | 决定 transport 路线：interface 增异步实现，或在 UR 侧用线程包装同步 `SharpaV3PolicyClient`。同时补 T4（`max_message_size` 未下发，属 I 侧真实 bug）与 T8 异常分层，否则 `policy_client_node` 错误分流失效 | A·T1、D·T4/T8 |
| 10 | 修 `state_node.py:519,526` 的异常依赖（随 8 一并处理） | D·O3 |
| 11 | 修 I 侧 `cancel_inference` 不回滚 `next_request_id`（`execution.py:181-187`） | D·S6 |
| 12 | UR 侧新增 revision 递增（safe-stop／replan 调 `Session.bump_revision`） | 缺口·S6 |
| 13 | 删除 U 的 7 个副本模块；`serialization.py` 可在第 9 步前先删（0 差异） | — |
| 14 | 忽略并清理 `/home/yns/deploy/sharpa_policy_v3_ws/`（严格更旧，无独有修复） | — |

### D 提案清单（需 interface schema／API 签核，勿静默移植）

上推 interface（不改 wire schema，只补实现能力）：O2、O3、O4、O6、B1、B2、B3、B4、
B6、B7、B9、B10、T2、T3、T4、T5、T6、T7、T8、T10、A11、A12、S2、S4、S5、S6。

需改 wire schema、必须签核：A5（`joint`／`eef` 至少一个非 None——收紧
`docs/interface_contract.md:90-99`）。

### 分类不确定项（明确记录，勿当作「可安全删除」）

- **B5 `eef_frame == "robot_base"`**：语义（UR EEF 表达在 robot base 系）是 A；字段名
  `eef_frame` 是被 `eef_def` 取代的旧 schema 残留，属 B。两者混在同一处代码里。
  保守做法：保留语义检查、迁入 `ur_common` 并改名，不要把 `eef_frame` 推入 interface。
- **A5 joint／eef 至少一个非 None**：判为 C（拒绝了契约允许的负载），但方向正确。
  是「UR 违反契约」还是「契约过宽」需 interface owner 裁定。
- **B2 byte 容量驱逐**：不确定这是本体资源保护（A，留在 UR 侧）还是 interface 应有的
  通用能力（D）。倾向 D，但需 owner 确认 buffer 内存预算归属。
- **T11 SSH relay**：仅 I 有，UR 未用。未验证 PND 是否依赖；本报告不建议改动。
