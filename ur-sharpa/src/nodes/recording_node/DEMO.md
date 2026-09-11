# D435 Demo recording

Deploy Dashboard includes a **Demo** control and a live D435 view. Start records
sensor data and camera streams; Stop drains the writer and finalizes the files.
Recording does not start, stop, or otherwise command the robot. Policy, Replay,
and idle hardware feedback can all be recorded. Existing teleop episodes retain
their original format and recorder.

Before Start, the operator can enter an absolute recording root in the Dashboard.
Each recording creates a `YYYY-MM-DD_HH-MM-SS.ffffff` directory beneath that
root. The path field is locked while recording or saving. **Delete recent demo**
is enabled only after Stop has fully finalized the latest recording, asks for
confirmation, and can delete only that exact timestamp directory. Starting a new
recording clears the previous delete target.

## Launch

```bash
source ur-sharpa/scripts/setup_env.sh
cd ur-sharpa
colcon build --symlink-install --packages-select recording_node dashboard_node ur_launch
source install/setup.bash
ros2 launch ur_launch deploy.launch.py
```

The existing dashboard remains at `http://127.0.0.1:8088`. The demo service
listens on loopback port 8089; the dashboard proxies it on the same origin.
Launch arguments:

| Argument | Default |
| --- | --- |
| `demo_camera_serial` | Empty: select exactly one D435/D435i |
| `demo_camera_width` | `1920` |
| `demo_camera_height` | `1080` |
| `demo_camera_fps` | `30` |
| `demo_recording_root` | `ur-sharpa/demo_recordings` |
| `demo_port` | `8089` |

Set `demo_camera_serial:=<serial>` if multiple D435 cameras are connected.
The two existing D405 wrist cameras are excluded. Missing/busy D435 devices
retry every two seconds and do not prevent the rest of Deploy from running.
D435/D435i uses only 1920x1080 RGB at 30 Hz; depth and IMU are not enabled. USB bandwidth must support the
additional camera; device frame gaps and shared-memory camera overruns are
reported in the recording manifest. Runtime dependencies are `pyrealsense2`,
Pillow, NumPy, FFmpeg with libx264, and ROS 2 `rosbag2_py` with SQLite storage.

To add the recorder to an already running dashboard without launching robot nodes:

```bash
ros2 run recording_node demo_node --ros-args \
  -p root:=/absolute/path/to/demo_recordings \
  -p serial:="'<D435 serial>'"
```

The rebuilt dashboard must be restarted to load its new API routes. A single
demo service must own the camera. Do not launch a second service on the same
device/port.

## Outputs

```text
demo_recordings/YYYY-MM-DD_HH-MM-SS.ffffff/
  sensors/                 # ROS 2 sqlite3 bag + metadata.yaml
  d435_rgb.mp4             # RGB-only H.264 video
  demo.mp4                 # center-cropped RGB + right-side 2x6 tactile/torque panel
  frames.jsonl             # Exact camera times and overlay measurements
  manifest.json            # Calibration, format, counts, validity and errors
```

ROS bag topics retain full `/ur/state`, `/sharpa/state`, `/sharpa/tau`,
`/sharpa/wrench`, `/sharpa/deformation`, and `/camera/set` messages. Sensor
subscriptions use best-effort depth 30, compatible with both reliable and
best-effort publishers. The three existing cameras are drained from the
full-rate shared-memory ring and reconstructed as CameraSet messages, with
individual original JPEG payloads and source timestamps. ROS CameraSet is a
fallback when shared memory has not been detected; its configured preview
frequency may be lower. CameraSet.header is the reconstruction time. Repeated
view timestamps within aggregated sets do not denote new camera frames.

`sensors` is a raw sensor recording, not a teleop `data.json` replay sample.
No sensor is required to be valid before starting: missing or invalid data is
retained and reported, and `manifest.complete` is false for such recordings.
The bag records whole messages at received frequency, including measured joint
positions, velocities, torques and command validity present in those messages.
All cameras record RGB only. D435 color intrinsics are stored in the manifest.
D435 RGB is rotated 180 degrees clockwise before preview, RGB recording and
overlay composition. The tactile panel remains upright. Manifest intrinsics
refer to the native sensor image before rotation; the rotation is recorded
as `camera.rotation_clockwise_degrees`.

Frames enter the encoder at the configured nominal rate, then the completed MP4
timestamps are scaled without re-encoding to match the first-to-last D435
acquisition interval. Dropped frames therefore lower the resulting average frame
rate instead of accelerating the action. `frames.jsonl` preserves exact receive
time, RealSense color timestamps and clock domains, frame numbers, per-source
sensor timestamps, ten full wrenches, and ten joint metrics. RGB MP4 is lossy;
ROS message storage is lossless. Frames accepted before Stop are drained before
files are finalized.

## Overlay contract

- Order: left then right; thumb, index, middle, ring, pinky.
- Each hand has 22 measured joint angles in radians. Their five slices are
  `[0:5]`, `[5:9]`, `[9:13]`, `[13:17]`, `[17:22]`.
- Each finger displays `sum(q_i*q_i)` in rad². This is a scalar joint metric,
  without a square root, not an anatomical angle or an independent physical DoF.
- The measurement panel is a fixed 2-column x 6-row grid to the right of the
  camera image and has exactly the same height. The left column is the left hand
  and the right column is the right hand. Each column contains thumb, index,
  middle, ring and pinky deformation maps followed by that hand's torque chart.
- The composite retains the D435 source dimensions. At 1920x1080, the side panel
  is 360 pixels wide and the RGB view is center-cropped equally by 180 pixels on
  the left and right to 1560x1080. The result is 1920x1080 with no stretching.
- Each torque chart has exactly five lines in the same finger order. A line is
  `tau2`, the sum of that finger's measured joint torques squared from
  `/sharpa/tau`, in (N*m)^2, over the latest eight seconds. Torque uses the same
  five joint slices as q, without a square root. The value is saved as
  `joint_torque_square_sum_nm2` in each frame. All six fingertip wrench
  components remain in `frames.jsonl` and the raw bag.
- Wrench/deformation input is pinky-to-thumb, and is reordered to match the
  joint groups. Invalid or older-than-250-ms values display `--` / `NO DATA`.
- Pairing selects the latest sensor message received no later than the camera
  receive timestamp. This is software alignment, not hardware synchronization.

A timestamp-named `.partial` directory is visible while recording. Normal stop
renames it after writing a manifest, including when data quality is incomplete.
Errors or queue drops are never marked complete. A crash can leave `.partial`
and unfinished MP4 files; source files are retained for diagnosis.
