# System dependencies (documented, not automated)

System prerequisites for the UR + SharpA workstation. Install these separately;
the project build does not modify system packages.

## ROS and Python

- Ubuntu 22.04 (or the exact version confirmed against the machine you
  intend to reproduce; note here when it changes).
- ROS 2 Humble.
- `python3-yaml`, `python3-numpy`, `python3-msgpack`, `python3-aiohttp`,
  `python3-pil`, `python3-zmq`, and `python3-empy` (ROS interface
  generation).
- Build and run ROS with the system Python 3.10 after sourcing only
  `/opt/ros/humble/setup.bash`; a Conda 3.8 environment shadows ROS Python
  modules and breaks `ament_package`/`rosidl_adapter`.
- `curl`, `unzip`, `tar` for SDK archives.

## Hardware runtime

- CUDA runtime (for ZED). Pin the exact minor version used at the time you
  build ZED SDK.
- `libusb-1.0-0-dev` (RTDE communication).
- Vendor AE01 Avatar SDK, its Python 3.10 dependencies (including Zenoh), and
  `pyzmq` for Vive input. See `scripts/install_ae01_python.sh`.
- FFmpeg with the selected encoder; the workbench defaults to `hevc_nvenc`
  and needs a compatible NVIDIA driver/GPU. Policy inference itself stays remote.
- `ssh`, `tmux`, and `curl` for remote policy lifecycle; remote profiles start
  with empty model paths until the operator configures them.
- Pinocchio runtime for hand FK, available via the launch-time
  `pinocchio_python_path` and `pinocchio_library_path` arguments. The existing
  compiled IKFast binding targets system Python 3.10.

## 数据与运行目录

采集默认写入 `data/episodes`，Demo 写入 `data/demo_recordings`，构建与日志写入 `runtime`。
厂商 SDK 放在 `external`，见 [安装说明](../external/README.md)。
