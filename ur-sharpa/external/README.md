# External runtime assets

This directory is a machine-local runtime cache. The repository tracks only
this manifest; SDK binaries, unpacked wheels, calibration files, and generated
logs must not be committed.

| Directory | Required source | Notes |
| --- | --- | --- |
| `sharpa-wave-sdk-5.0.7/` | [SharpA Wave SDK v5.0.7](https://github.com/sharpa-robotics/sharpa-wave-sdk/releases/tag/v5.0.7), asset `sharpa-wave-sdk_5.0.7_amd64.deb` | Current default for hand firmware 3.0.6 and 30 Hz tactile data. Extract the deb's `/opt/sharpa-wave-sdk/` tree here. |
| `sharpa-manus-sdk/` (legacy only) | SharpA/Manus vendor delivery | Must contain `SharpaManusClient.out`, `ManusSDK/lib/`, and the two `.mcal` calibration files. No public binary download is currently available. |
| `manus-retarget/` (legacy only) | SharpA/Manus vendor retarget release | Must contain `retargeting_manus_demo_multiprocess.py` and `include/proto_hand/sharpa_hand_pb2.py`. |
| `manus-python/` (legacy only) | Python 3.10 packages for the vendor retarget release | Install its vendor-pinned requirements into this target directory. |
| `pinocchio-runtime/` | [Pinocchio](https://github.com/stack-of-tasks/pinocchio) Python 3.10 runtime | Needed by hand FK and recorded fingertip positions. |
| `realsense-python/` | [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) Python 3.10 binding | System `pyrealsense2` may be used instead when its ABI matches. |

The workbench uses **AE01** for hand teleoperation. Install the vendor Avatar SDK
at `/opt/avatar-sdk` (or set `AE01_SDK_ROOT` / the `ae01_sdk_root` launch argument),
including `lib/python/avatar_sdk` and `share/sdk_config.json`. Run
`scripts/install_ae01_python.sh` after the vendor installer to install its Python
requirements and expose the SDK to system Python 3.10. The existing Manus
payloads above are only used by the retained standalone legacy node.

No local policy model runtime or checkpoints are needed. Remote services use
`config/model_servers.json` or the ignored `model_servers.local.json` override.

Example: download and unpack the current Wave SDK without overwriting another
installed SDK version:

```bash
cd "$(git rev-parse --show-toplevel)/ur-sharpa"
mkdir -p external/downloads external/sharpa-wave-sdk-5.0.7
curl -L -o external/downloads/sharpa-wave-sdk_5.0.7_amd64.deb \
  https://github.com/sharpa-robotics/sharpa-wave-sdk/releases/download/v5.0.7/sharpa-wave-sdk_5.0.7_amd64.deb
tmp_dir="$(mktemp -d)"
dpkg-deb -x external/downloads/sharpa-wave-sdk_5.0.7_amd64.deb "$tmp_dir"
cp -a "$tmp_dir/opt/sharpa-wave-sdk/." external/sharpa-wave-sdk-5.0.7/
rm -rf "$tmp_dir"
```

The release-published SHA-256 for the 5.0.7 CPU deb is
`194896e4b981ffa5494ff3f00500ce1e31c3b28a042e190454894658aad9e465`.
