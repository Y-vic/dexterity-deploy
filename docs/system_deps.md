# System dependencies (documented, not automated)

Not everything can be shipped in `deps/cache/`. The following must be
installed via the OS package manager on each machine. `deps/fetch.sh` does
NOT do this — it fails cleanly if these are missing, so you know to install.

## Common (both robots + workstation)

- Ubuntu 22.04 (or the exact version confirmed against the machine you
  intend to reproduce; note here when it changes).
- ROS 2 Humble.
- `python3-yaml`, `python3-numpy`.
- `curl`, `unzip`, `tar` (for `fetch.sh`).

## UR host only

- CUDA runtime (for ZED). Pin the exact minor version used at the time you
  build ZED SDK.
- `libusb-1.0-0-dev` (RTDE communication).

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
