# Shared robot assets

这里保存跨 embodiment 共用的机器人模型资源，避免在 `pnd-sharpa/`
和 `ur-sharpa/` 中复制同一份 URDF、mesh 或 MuJoCo 文件。

- `adam_sharpa_description/urdf/`：Adam Pro + SharpA URDF 及 mesh。
- `adam_sharpa_description/mujoco/`：与上述 URDF 配套的 MuJoCo 模型。

`adam_sharpa_description/` 本身是可共用的 ROS description package。
`pnd-sharpa/src/adam_sharpa_description` 是指向该 package 的相对软链接，
因此现有 `package://adam_sharpa_description/...` 引用保持不变。
