# Canonical wrist pose9

Public interface 的 EEF 是 PND 定义的 canonical pose9：

```text
[x, y, z, r00, r10, r20, r01, r11, r21]
```

- `xyz` 是 wrist 原点在 `canonical_root` 中的绝对位置（米）。
- `canonical_root`：`+X=left`、`+Y=up`、`+Z=forward`。
- `R` 是 root-from-wrist rotation；后 6 维取 R 的前两列，与 PND
  `mat_to_rot6d()` 一致。
- wrist child 顺序：`left_hand_C_MC`、`right_hand_C_MC`。

UR 的 29.5 mm TCP→hand_C_MC 偏移、左右 UR base 标定都属于 UR adapter，不能写进
interface。旧 `wire_pose` 的 server base correction 也只是兼容层，不进入 ROS
`UrStateFrame` 或 `ExecutableAction`。

当前远端 server 仍内置 PND deploy-physical→posttrain 转换。为避免二次换轴，只有
`policy_client_node` 的 transport boundary 会临时执行
`canonical ↔ legacy server pose`；ROS 内部和共享 interface 始终保持 canonical。
