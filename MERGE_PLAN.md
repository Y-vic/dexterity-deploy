# Interface migration plan

当前 repo 已冻结 v4 最小 dict contract、七路 buffer、action slice 和同步执行 gate。
完整字段见 `docs/interface_contract.md`，节点边界见 `docs/architecture.md`。

剩余集成顺序：

1. DreamZero server 的 template protocol/model adapters 迁移到 v4 output。
2. PND workstation 用 `PolicyClientCore` 替换旧的 model-specific client 和旧执行调度。
3. 建立 `policy_client -> action_ik -> action_execute -> execution_done` ROS topics。
4. UR 使用同一 policy client core，只替换 embodiment producers、IK 和 executor。
5. 两边完成 shadow、超时、stale done、部分执行失败和断线恢复测试后上真机。

非目标：共享 IK、共享硬件 driver、在 interface 中固定本体 base frame、支持 async。
