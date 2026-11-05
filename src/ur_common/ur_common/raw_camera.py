"""Versioned packed BGR frames for the local recording and inference camera ring."""
import struct

RAW_ENCODING = "bgr8_v1"
_HEADER = struct.Struct("<8sIII")
_MAGIC = b"USBGRV1\0"


def pack_bgr(image):
    import numpy as np
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("raw camera frame must be uint8 HxWx3 BGR")
    height, width = image.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("empty raw camera frame")
    return _HEADER.pack(_MAGIC, width, height, width * 3) + image.tobytes(order="C")


def unpack_bgr(data):
    if len(data) < _HEADER.size:
        raise ValueError("truncated raw camera header")
    magic, width, height, stride = _HEADER.unpack_from(data)
    if (magic != _MAGIC or not 0 < width <= 8192 or not 0 < height <= 8192
            or stride != width * 3 or len(data) != _HEADER.size + height * stride):
        raise ValueError("invalid raw camera dimensions or payload length")
    return width, height, memoryview(data)[_HEADER.size:]


def complete_bgr(data):
    try:
        unpack_bgr(data)
        return True
    except (ValueError, TypeError, struct.error):
        return False
