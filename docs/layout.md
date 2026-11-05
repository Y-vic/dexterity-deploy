# 文件组织与迁移

仓库根目录 `/home/yns/deploy` 同时是唯一 ROS 工作区。
不再按机器人型号嵌套工作区，不保留旧路径兼容软链接。

| 内容 | 当前位置 |
| --- | --- |
| 业务节点 | `src/nodes/<package>` |
| 公共运动学、驱动辅助、网页资源 | `src/ur_common` |
| ROS 消息 | `src/ur_msgs` |
| 远端模型 Python 协议 | `interface/sharpa_interface` |
| 静态保护、模型配置 | `config` |
| 本机 SDK | `external` |
| 录制数据 | `data/episodes`、`data/demo_recordings` |
| colcon 输出 | `runtime/build`、`runtime/install`、`runtime/log/colcon` |
| ROS 与执行日志 | `runtime/log/ros`、`runtime/log/execution` |
| 标定与目录记忆 | `runtime/config` |
| 缓存与临时文件 | `runtime/cache`、`runtime/tmp` |

网页与手部 URDF 属于 `ur_common` 包，跟随包安装，不另建重复的顶层 assets。
`external/README.md` 是当前 SDK 安装说明；旧 deps 清单含过期版本，已移出。

2026-11-04 迁移时，PND、Astribot、Adam 模型、旧实验与历史文档先打包并用
`tar --compare` 校验，归档保存在仓库之外的
`/home/yns/deploy/.codex_backups/layout-20261104/legacy.tar.gz`。
同目录的 `migration.json` 记录路径映射，旧 build/install 仅供恢复，不再加载。
原来的采集数据完整迁移，旧运行日志保留在 `runtime/log`。

新增源文件放在所属包内，测试就近放在包的 `test`；运行产物只能放进 runtime，
录制文件只能放进 data 或用户明确指定的项目内目录。
