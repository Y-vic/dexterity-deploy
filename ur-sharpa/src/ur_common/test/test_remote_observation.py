import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ur_common.raw_camera import pack_bgr
from ur_common.remote_observation import RemoteCameraEncoder


def frame(stamp=10):
    data = np.zeros((40, 64, 3), dtype=np.uint8)
    data[:] = [15, 40, 220]
    message = SimpleNamespace()
    for source in ('ego', 'left_wrist', 'right_wrist'):
        setattr(message, source, pack_bgr(data))
        setattr(message, source + '_valid', True)
        setattr(message, source + '_timestamp_ns', stamp)
    return message


def test_raw_images_keep_native_resolution_orientation_and_rgb_channels():
    message = frame()
    result = RemoteCameraEncoder().encode(message)
    assert [name for name, _, _ in result] == ['ego_cam', 'left_wrist_cam', 'right_wrist_cam']
    for _, stamp, jpeg in result:
        image = Image.open(io.BytesIO(jpeg))
        assert image.size == (64, 40)
        assert stamp == 10
        assert np.allclose(np.asarray(image)[20, 30], [220, 40, 15], atol=3)


def test_timestamps_advance_independently_and_invalid_samples_are_retryable():
    encoder = RemoteCameraEncoder()
    message = frame()
    message.left_wrist_valid = False
    assert len(encoder.encode(message)) == 2
    message.left_wrist_valid = True
    assert [name for name, _, _ in encoder.encode(message)] == ['left_wrist_cam']
    assert encoder.encode(message) == []
    message.ego_timestamp_ns = 9
    message.right_wrist_timestamp_ns = 11
    assert [(name, stamp) for name, stamp, _ in encoder.encode(message)] == [('right_wrist_cam', 11)]


def test_malformed_camera_does_not_starve_other_cameras_or_advance_its_stamp():
    encoder = RemoteCameraEncoder()
    message = frame()
    message.ego = b'broken'
    assert len(encoder.encode(message)) == 2
    assert 'ego' in encoder.errors
    assert 'ego' not in encoder.stamps
    message.ego = frame().ego
    assert [name for name, _, _ in encoder.encode(message)] == ['ego_cam']
    assert encoder.errors == {}
