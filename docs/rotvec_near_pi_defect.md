# matrix_to_rotvec 近 π 失效（现存缺陷，非重构引入）

`matrix_to_rotvec` 的 `arccos/sin` 分支在旋转角接近 π 时条件数崩坏。近 π 保护阈值设在
`π - angle < 1e-6`，但实际失效点在 `1e-2`——**保护阈值比失效点小四个数量级**。

这不是重构引入的问题，是当前 deploy 路径里已经在跑的缺陷。

## float32 是放大器

单看 float64 输入时误差可忽略（gap 2.6e-3 处仅 1.99e-10），所以这个缺陷长期不可见。但
`ikfastpy.pyx` 声明 `vector[float]`，FK 输出经 float32 截断后才进 `matrix_to_rotvec`。

实测（600 次随机轴，float32 截断后输入）：

| π−angle | 原版（arccos/sin） | 四元数版 |
|---|---|---|
| 1e-2 | 1.21e-03 rad (0.07°) | 5.5e-08 |
| 2.6e-3 | **1.98e-02 rad (1.13°)** | 4.9e-08 |
| 1e-3 | **1.09e-01 rad (6.2°)** | 4.9e-08 |
| 1e-4 | **2.01 rad (115°)** | 5.0e-08 |

机理：`(r[2,1]-r[1,2], ...) / (2·sin(angle))`。float32 截断在分子引入 ~6e-8 噪声，除以趋零
的 `sin(angle)` 后放大。

## 为什么是常规工况

`ur_common` 移植时实测本 cell 的 UR5 FK 输出 `π − angle` 低到 **2.6e-3**，对应 1.13° 的 TCP
姿态误差。`transform_to_rtde_pose` 喂的是每一条 UR TCP 命令，所以这条路径每帧都过。

## 影响范围

| 位置 | 状态 |
|---|---|
| `ur-sharpa/src/nodes/sharpa_policy_v3_client/sharpa_policy_v3_client/hardware_geometry.py:216,234` | **活跃 deploy 路径** |
| `/home/yns/deploy/sharpa_policy_v3_ws/src/.../hardware_geometry.py:216,234` | 活跃副本 |
| `ur-sharpa/ur_sharpa/adapters/wire_geometry.py:167,174` | adapter，未接入 |
| `ur-sharpa/src/ur_common/ur_common/geometry.py` | **已修**（四元数路径，回归测试 `test_rotvec_is_accurate_near_pi`） |

PND 侧无此模式，不受影响。

## 修正会动标定

现有标定是在这个误差存在的前提下做的。修正 rotvec 会让 TCP 实际位姿变化约 1°，**标定可能
需要重做**。

因此建议：

1. `ur_common`（新链路）已修，保持。新链路本来就要重新标定。
2. 活跃 v3 路径**暂不动**，避免在旧链路上引入未标定的 1° 偏移。
3. P8 切换到 `actor_ik_node` 时统一处理——那时标定必然重做，一起验证。

如果在 P8 之前需要继续用旧链路做实验，注意它带这个误差；跨链路比对姿态数据时不可直接对齐。

## 附带发现

FK→IK→FK 闭环容差止步 **1.3e-05**（而非 1e-6），上限同样来自 `vector[float]` 绑定：
`forward()` 输出恰好是 float32 可表示值，其旋转正交性误差 5e-8 = float32 eps。要更紧需改
`vector[double]` 并重编 `.so`，会动标定，未做。
