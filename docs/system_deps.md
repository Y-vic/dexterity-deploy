# System dependencies (documented, not automated)

Not everything can be shipped in `deps/cache/`. The following must be
installed via the OS package manager on each machine. `deps/fetch.sh` does
NOT do this — it fails cleanly if these are missing, so you know to install.

## Common (both robots + workstation)

- Ubuntu 22.04 (or the exact version confirmed against the machine you
  intend to reproduce; note here when it changes).
- ROS 2 Humble.
- `python3-yaml`, `python3-numpy`, `python3-msgpack`, `python3-aiohttp`,
  `python3-pil`, `python3-zmq`, and `python3-empy` (ROS interface
  generation).
- Build and run ROS with the system Python 3.10 after sourcing only
  `/opt/ros/humble/setup.bash`; a Conda 3.8 environment shadows ROS Python
  modules and breaks `ament_package`/`rosidl_adapter`.
- `curl`, `unzip`, `tar` (for `fetch.sh`).

## UR host only

- CUDA runtime (for ZED). Pin the exact minor version used at the time you
  build ZED SDK.
- `libusb-1.0-0-dev` (RTDE communication).
- Vendor AE01 Avatar SDK, its Python 3.10 dependencies (including Zenoh), and
  `pyzmq` for Vive input. See `ur-sharpa/scripts/install_ae01_python.sh`.
- FFmpeg with the selected encoder; the workbench defaults to `hevc_nvenc`
  and needs a compatible NVIDIA driver/GPU. Policy inference itself stays remote.
- `ssh`, `tmux`, and `curl` for remote policy lifecycle; remote profiles start
  with empty model paths until the operator configures them.
- Pinocchio runtime for hand FK, available via the launch-time
  `pinocchio_python_path` and `pinocchio_library_path` arguments. The existing
  compiled IKFast binding targets system Python 3.10.

## PND robot only

- CUDA runtime (for ZED + WebRTC encoder).
- `ffmpeg` >= 5.x (RTP push).
- CycloneDDS or the DDS impl PND lowstate expects — check the SDK before
  installing.

## PND workstation only

- CUDA runtime (for policy inference + dashboard).
- `ffmpeg` >= 5.x (RTP receive).

## Recording disk

Both robots mount a dedicated NVMe at `/mnt/recording` (or whatever the
`recording_root` launch arg points to). See each embodiment's `README.md`
for the exact mount script. This is NOT installed by `deps/fetch.sh` because
disk layout is intrinsically per-machine.
