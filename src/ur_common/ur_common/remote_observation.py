"""Encode policy JPEGs from raw camera frames independently of UI preview cadence."""
import io

import numpy as np
from PIL import Image

from .raw_camera import unpack_bgr


class RemoteCameraEncoder:
    def __init__(self):
        self.stamps = {}
        self.errors = {}

    def encode(self, message):
        frames = []
        self.errors = {}
        if message is None:
            return frames
        for source, name in (
            ('ego', 'ego_cam'),
            ('left_wrist', 'left_wrist_cam'),
            ('right_wrist', 'right_wrist_cam'),
        ):
            try:
                stamp = int(getattr(message, source + '_timestamp_ns'))
                if not getattr(message, source + '_valid') or stamp <= self.stamps.get(source, 0):
                    continue
                width, height, raw = unpack_bgr(getattr(message, source))
                bgr = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                buffer = io.BytesIO()
                Image.fromarray(bgr[:, :, ::-1]).save(buffer, format='JPEG', quality=90)
                frames.append((name, stamp, buffer.getvalue()))
                self.stamps[source] = stamp
            except (AttributeError, TypeError, ValueError) as exc:
                # Each camera has its own clock and validity. A broken packet
                # from one camera must not starve the other observation queues.
                self.errors[source] = str(exc)
        return frames
