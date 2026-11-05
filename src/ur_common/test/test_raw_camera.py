import numpy as np
import pytest
from ur_common.raw_camera import pack_bgr, unpack_bgr, RAW_ENCODING
from ur_common.camera_transport import SharedCameraReader, SharedCameraWriter
from ur_common.drivers import _ThreadedCamera
import json
import time


def test_raw_ring_preserves_strided_pixels_and_detects_overrun(tmp_path):
    image = np.arange(6*8*3, dtype=np.uint8).reshape(6, 8, 3)[:, ::2]
    payload = pack_bgr(image)
    writer = SharedCameraWriter("raw_test", len(payload), tmp_path, slot_count=2)
    reader = SharedCameraReader("raw_test", tmp_path, encoding=RAW_ENCODING)
    try:
        writer.write("ego", 10, payload)
        frame = reader.read_pending()[0]
        w, h, bgr = unpack_bgr(frame.ego)
        assert (w, h) == (4, 6)
        assert bytes(bgr) == image.tobytes()
        assert frame.encoding == RAW_ENCODING
        for stamp in (11, 12, 13):
            writer.write("ego", stamp, payload)
        assert [f.ego_timestamp_ns for f in reader.read_pending()] == [12, 13]
        assert reader.dropped_frames["ego"] == 1
        writer.write("ego", 14, payload[:-1])
        assert reader.read_pending() == []
        assert reader.dropped_frames["ego"] == 2
    finally:
        reader.close()
        writer.close()


def test_raw_header_rejects_invalid_size():
    payload = pack_bgr(np.zeros((4, 6, 3), np.uint8))
    for malformed in (payload[:-1], payload + b"x", b"", b"X" + payload[1:]):
        with pytest.raises(ValueError):
            unpack_bgr(malformed)


def test_stalled_preview_does_not_discard_recording_frames():
    camera = _ThreadedCamera(raw_queue_size=64)
    camera.configure_raw_recording(preview_fps=10)
    image = np.zeros((4, 6, 3), np.uint8)
    # Never run the JPEG consumer: preview remains stalled for 60 captures.
    for i in range(60):
        started = 1_000_000_000 + i*33_333_333
        camera._accept_raw_capture(image, i+1, started, started+1000)
    assert camera.read_raw_latest()[0] == 60
    assert [t for t, _ in camera.read_raw_pending()] == list(range(1, 61))
    # Draining the recording queue does not consume the inference snapshot.
    assert camera.read_raw_latest()[0] == 60
    assert len(camera._raw_queue) == 1
    assert camera.recording_raw_drop_count == 0
    assert camera.preview_skipped_count >= 59
    for i in range(65):
        camera._accept_raw_capture(image, i+1, 4_000_000_000+i*33_333_333, 4_000_001_000+i*33_333_333)
    assert camera.recording_raw_drop_count == 1
    assert len(camera.read_raw_pending()) == 64


@pytest.mark.parametrize('raw_recording', [False, True])
def test_frame_timing_correlates_capture_and_publish_without_consuming_frames(raw_recording):
    source = time.time_ns()-70_000_000
    image = np.zeros((4, 6, 3), np.uint8)

    class OneFrameCamera(_ThreadedCamera):
        def _grab_rgb(self):
            self._stop.set()
            return image
        def _sample_timestamp_ns(self):
            return source

    camera = OneFrameCamera()
    if raw_recording:
        camera.configure_raw_recording()
    camera._capture_loop()
    assert 'frame_timing_events' not in camera.diagnostics()
    events = camera.diagnostics(include_frame_timings=True)['frame_timing_events']
    assert len(events) == 1
    capture = events[0]
    assert capture['stage'] == 'capture_ready' and capture['source_timestamp_ns'] == source
    assert capture['at_ns'] >= source+70_000_000
    assert capture['monotonic_ns'] >= capture['grab_finished_monotonic_ns'] >= capture['grab_started_monotonic_ns']
    camera.record_raw_publish(source, capture['at_ns']+8_000_000, capture['monotonic_ns']+8_000_000, 1.5)
    published = camera.diagnostics(include_frame_timings=True)['frame_timing_events']
    assert published == [dict(stage='raw_published', source_timestamp_ns=source,
                             at_ns=capture['at_ns']+8_000_000, monotonic_ns=capture['monotonic_ns']+8_000_000,
                             pack_write_ms=1.5)]
    assert camera.diagnostics(include_frame_timings=True)['frame_timing_events'] == []
    assert camera.read_raw_latest()[0] == source
    if raw_recording:
        assert camera.read_raw_pending()[0][0] == source
    json.dumps(events+published, allow_nan=False)


def test_frame_timing_queue_reports_overflow_without_blocking_camera():
    camera = _ThreadedCamera()
    for i in range(520):
        camera.record_raw_publish(i, i+1, i+2, 0.1)
    d = camera.diagnostics(include_frame_timings=True)
    assert len(d['frame_timing_events']) == 512
    assert d['frame_timing_dropped_count'] == 8
    assert d['frame_timing_events'][0]['source_timestamp_ns'] == 8
    assert d['frame_timing_events'][-1]['source_timestamp_ns'] == 519
