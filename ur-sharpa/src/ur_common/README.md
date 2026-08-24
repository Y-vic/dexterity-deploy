# ur_common

UR5 + SharpA 的**唯一**正/逆运动学入口。teleop 和 deploy 都调这里。

两套运动学实现会让观测链路和执行链路在标定变化后逐渐对不上（calibration drift），本包就是为了杜绝这件事。不要在别处再写一份 FK/IK。

## 1. 硬件是 UR5，不是 UR5e

`ur_common/ikfastpy/` 里的编译产物是按 **UR5** 的 DH 参数生成的（`.so` 内部标识 `robot:GenericRobot - ur5`，肩部偏置 `0.089159`）。UR5e 的数值不同，换错型号不会报错，只会让 TCP 静默偏掉。

零位 TCP 位置可作为判据，三个分量在 UR5e 上都不一样：

| 分量 | UR5 | UR5e |
| --- | --- | --- |
| x = a2+a3 | -0.81725 | -0.81720 |
| y = -(d4+d6) | **-0.19145** | **-0.23290** |
| z = d1-d5 | **-0.005491** | **+0.062800** |

`test_arm_fk_is_ur5_not_ur5e` 就是卡这个的。

## 2. `.so` 只支持 Python 3.10

`ikfastpy.cpython-310-x86_64-linux-gnu.so` 是给 CPython 3.10 编的，换小版本必然 import 失败。

- `ikfast_available()` 判断当前解释器能不能用。
- 用不了时 `load_ikfast()` 抛 `IkfastUnavailable`，消息里带当前版本和排查建议，不做静默降级。
- 要换解释器就用 `ur_common/ikfastpy/setup.py` 重新编（需要 Cython + lapack）。

纯几何部分（`ur_common.geometry`）不依赖 `.so`，任意解释器可用。

手部指尖 FK 需要 `pinocchio`，用 `hand_fk_available()` 判断。

## 3. Public API

```python
from ur_common.kinematics import UrCommonKinematics, SafetyConfig

kin = UrCommonKinematics(SafetyConfig())   # 默认三个开关全关
```

### 手臂 FK / IK

| 方法 | 输入 → 输出 |
| --- | --- |
| `arm_fk(joints)` | UR5 6 关节角 → UR base 系 4x4 TCP |
| `arm_fk_rtde(joints)` | → `[xyz, rotvec]` (6,)，RTDE 约定 |
| `arm_fk_eef9(joints, side)` | → canonical EEF `xyz+rot6d` (9,)，PND column-Rot6D，absolute |
| `arm_ik(target, near_joints, side, ref_joint, allow_large)` | 4x4 或 `[xyz,rotvec]` + 种子角 → 6 关节角 |
| `arm_ik_eef9(eef9, near_joints, side, ...)` | canonical EEF → 6 关节角，`arm_fk_eef9` 的严格逆变换 |

`arm_ik` 无解时**保持**（返回上一帧下发目标，无则返回 `near_joints`），不会硬凑一个姿态。诊断看 `kin.solver(side).stats` / `.report()`。

### 手部 FK

| 方法 | 输入 → 输出 |
| --- | --- |
| `hand_fingertip_positions(q44)` | 44D 关节角 → `{'left': (5,3), 'right': (5,3)}` |
| `hand_fingertip_transforms(q44)` | → `{'left': (5,4,4), 'right': (5,4,4)}` |

44D 顺序 = interface 冻结顺序（左手 22 + 右手 22，每只手 thumb→pinky），见 `interface/sharpa_interface/sharpa.py`。左右 pinocchio 模型**只建一次**并缓存，`hand_model_cache_info()` 可查；这是原实现为了性能加的，不要去掉。

`assets/hand_urdf/` 只放了两个 `.urdf`，没带 31MB 的 STL —— mesh 是 `package://` 的 visual/collision 引用，FK 只用运动学树，不读几何。

### 坐标系转换

主链路：`ur_base → capture_root → SharpA hand_C_MC → canonical_root`。

| 方法 | 说明 |
| --- | --- |
| `ur_base_to_capture_root(t, side)` / `capture_root_to_ur_base` | 左右 UR base 相对公共 capture root |
| `capture_tcp_to_canonical_pose(t, side)` / `canonical_pose_to_capture_tcp` | capture ↔ canonical，出入 PND column-pose9 |
| `kin.geometry` | UR base 标定、TCP↔hand_C_MC 安装变换及显式 legacy wire adapter |

公共 root/wrist 轴和 pose9 顺序只由 `interface/sharpa_interface/sharpa.py` 定义。
`UR_TCP_FROM_SHARPA_HAND` 中的 29.5 mm 是这套 UR-SharpA 的安装偏移，只留在
`ur_common/geometry.py`，不属于 interface。`SERVER_DEPLOY_FROM_MODEL_BASE` 仅供显式
legacy wire 兼容，不进入 ROS 状态或动作主链。

### 旋转表示互转

`ur_common.geometry`：`matrix_to_column_rot6d` / `column_rot6d_to_matrix`
（canonical interface 使用的 PND 约定）、`matrix_to_row_rot6d` /
`row_rot6d_to_matrix`（legacy model 中间量）、`matrix_to_quat` /
`quat_to_matrix`、`matrix_to_rotvec` / `rotvec_to_matrix`，以及 v1 的
`pose_mat` / `pose_vector` / `pose_vector_quat`。

> `matrix_to_rotvec` 改用四元数中转，是相对原实现的**修正**。原来的 arccos/sin 形式除以 `sin(angle)`，角度接近 π 时误差放大；实测 400 个 UR5 位姿上最差 3.4e-03 rad（0.19°），改后 5.4e-08 rad。这条在 deploy 主链路上（`transform_to_rtde_pose` 喂每一条 UR TCP 指令），而本工位的 FK 输出经常落在 π 附近，不是边角情况。回归测试 `test_rotvec_is_accurate_near_pi`。

## 4. 三个安全开关

**全部默认关闭**。每个都是单一开关，打开即完整恢复 v1 行为；关闭时逻辑仍在，只是被绕过（代码没删）。

| 开关 | 默认 | 位置 | 关着的后果（源码记录的真机结果） |
| --- | --- | --- | --- |
| `enable_jump_gate` | `False` | `kin.jump_gate` → `gate_wrist()` | tracker 坏帧（几十厘米级毛刺）直接进 IK，解出错误位姿并朝其运动 |
| `enable_flip_hold` | `False` | `kin.solver(side)` → `ik_solver.py` | **真机撞机**。详见下方 |
| `enable_joint_clip` | `False` | `kin.joint_limiter` → `limit_joints()` | 原始 IK 输出直接进 servoJ，无步长/范围约束，一次大跳变就是一帧大动作 |

打开方式：

```python
SafetyConfig(enable_jump_gate=True, enable_flip_hold=True, enable_joint_clip=True)
SafetyConfig.v1_defaults()   # 等价，三个全开 + v1 阈值
SafetyConfig.from_env()      # 读 v1 环境变量，默认按 v1（即全开）
```

> `SafetyConfig()` 默认全关，`from_env()` 默认全开（对齐 v1 的 `TELEOP_JUMP_GATE=1`）。极性不同，别混。

当前 Vive Teleop 不使用 `enable_jump_gate`，但最终 `ur_node` 出口始终启用关节
绝对范围与逐命令步长保护。对齐后的 target 会先转换为实际 UR base/TCP，再以
实时测量 TCP 为基准按 `0.03 m` 平移范数和 `0.12 rad` 相对旋转角做 clip；IK
结果若相对实测关节超过 `0.35 rad` 也会拒绝。Deploy、Teleop 与 Bias 最终都经过
同一个 UR 出口，不能绕过这些限制。

### `enable_flip_hold` 为什么必须留意

v1 `robot_control/ik_solver.py` 明确记录过：曾把 `soft_flip_limit` 设成 2.8，想让"只超一点"的换腕解放行，靠下游 `limit_joint_vel`(0.35 rad/帧) 分帧过渡。真机结果是**撞机** —— 一次 SOFT-PASS(min_jump=2.321) 后，左臂位置偏差 9mm → 37.8mm → 186.2mm，两帧触发保护停。

原因：**clip 只限制每帧步长，不改变方向**。目标始终是那个换腕解，机械臂会以 0.35 rad/帧坚定地朝错误构型冲，并不会"平滑过渡"。hold 之所以安全，正是因为它根本不朝那个方向动。2.3 的换腕和 3.9 的换腕都是换腕，不能按数值远近放行。

所以：`enable_joint_clip` **不能**替代 `enable_flip_hold`。`soft_flip_limit` 默认 0（超阈值一律 hold），不要往上调。

### 阈值参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `jump_vel` | 3.5 m/s | 平移速度上限，@30Hz ≈ 117mm/帧 |
| `jump_rot_vel` | 1200 deg/s | 旋转速度上限，@30Hz ≈ 40°/帧 |
| `jump_max_reject` | 5 | 连续拒绝上限，超过强制放行一帧（防 tracker 真位移后永久卡死） |
| `jump_trans_thresh` / `jump_rot_thresh` | `None` | 直接指定每帧容差，优先于速度换算 |
| `flip_threshold` | `None` → 6× vel limit | 换腕判定阈值 |
| `joint_vel_limit` | 0.35 rad/帧 | |
| `joint_pos_limit` | ±6.28 ×6 | |
| `ur_max_joint_step_rad` | 0.35 rad/命令 | `ur_node` 最终出口与 Vive IK 后检查共用 |
| `ur_max_abs_joint_rad` | ±6.28 rad | `ur_node` 最终出口绝对范围，可由 launch 调整 |
| `frequency` | 30 Hz | 阈值按速度定义再按频率折算；固定"每帧 5cm"在 30Hz 和 60Hz 含义差一倍 |

阈值取得比人手更宽是刻意的：3.5 m/s 远超正常遥操作（快速挥动约 1.5–2.5 m/s），只拦真毛刺。v1 早期用 0.05m/帧（@30Hz 仅 1.5m/s），把正常快动作误判成坏帧而 hold，导致"快动作卡顿、慢动作正常"。

## 5. 测试

```bash
cd ur-sharpa/src/ur_common
python3 -m pytest test -q
```

无需硬件。solver 相关用例在非 3.10 解释器上自动 skip 并给出原因，不会伪造结果。

FK → IK → FK 闭合实测精度（3000 个随机可达位姿）：

| | median | p99 | max |
| --- | --- | --- | --- |
| 位置 | 3.0e-08 m | 5.8e-06 m | **1.3e-05 m** |
| 旋转 | 7.3e-08 rad | 5.2e-07 rad | **1.3e-05 rad** |

**达不到 1e-6 的 max，瓶颈是绑定层不是算法**：`ikfastpy.pyx` 声明的是 `vector[float]`，整个位姿以 float32 过 C++ 边界。已验证 `forward()` 输出恰好可被 float32 精确表示（截断误差 0.0），返回旋转的正交性误差约 5e-8，正是 float32 eps 量级。要做到 1e-6 得改成 `vector[double]` 重新生成绑定（见下方 TODO）。测试断言取在实测 max 之上一档（5e-05），没有进一步放宽。

## 6. TODO

- `ikfastpy` 绑定是 float32（`vector[float]`），把 FK/IK 闭合精度限在 ~1.3e-05。改 `vector[double]` 重新生成可到 float64，但需要重编 `.so`，且要在真机上复核标定，暂未做。
- `.so` 只有 cpython-310 一个构建产物，没有多版本 wheel。
- `package.xml` 里 `python3-pinocchio` 按 rosdep 名字写；本机 pinocchio 实际装在 conda env 里，`colcon build` 前需确认解释器能 import。
