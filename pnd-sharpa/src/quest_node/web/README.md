# Quest WebVR

This directory is a build-free Quest browser client. It uses native WebXR and WebGL, sends PND-compatible tracking data to the same-origin `/vrwebsocket`, and consumes the existing ZED producer through the same-origin `/webrtc` GStreamer signaling endpoint.

The page intentionally does not connect to `/nuc_ws` or `/jetson_ws`. The Quest stack owns tracking, while `zed_node` owns the camera pipeline and the existing root-level `/gstwebrtc-api-3.0.0.esm.js` asset.

## Controller payload

- `Head`, `LeftHand`, and `RightHand` contain WebXR `local-floor` position, XYZ Euler degrees, and quaternion.
- `Joy.axes` is `[L_x, L_y, L_trigger, L_grip, R_x, R_y, R_trigger, R_grip]`.
- `Joy.buttons` contains six `[value, touched]` pairs in `[L_thumb, L_X, L_Y, R_thumb, R_A, R_B]` order.
- Right A maps the current arms-forward pose to the Adam Pro reference; right B
  clears that calibration on the ROS side.
- The ZED video consumer starts automatically and reconnects after interruption.
- Top-bottom ZED frames are cropped and rendered on WebXR left/right eye layers.
- Low-poly virtual hands follow the left and right controller `gripSpace` poses.
