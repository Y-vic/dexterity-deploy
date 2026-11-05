"""Loss-detecting local tactile ring, preserving the ROS left/right wire order."""
import struct
from types import SimpleNamespace

from .camera_transport import SharedCameraReader, SharedCameraWriter

DEFAULT_PREFIX = "ur_sharpa_deformation_v1"
HAND_BYTES = 5 * 240 * 240
_HEADER = struct.Struct("<8sII10?")
_MAGIC = b"USDEFM1\0"


def pack_deformation(msg):
    left, right = bytes(msg.left), bytes(msg.right)
    if len(left) not in (0, HAND_BYTES) or len(right) not in (0, HAND_BYTES):
        raise ValueError("deformation must contain 5x240x240 bytes per hand or be empty")
    if (getattr(msg, "fingertip_count", 5), getattr(msg, "height", 240),
            getattr(msg, "width", 240)) != (5, 240, 240):
        raise ValueError("invalid deformation geometry")
    return _HEADER.pack(_MAGIC, len(left), len(right),
                        *msg.left_valid, *msg.right_valid) + left + right


def complete_deformation(data):
    if len(data) < _HEADER.size:
        return False
    magic, left, right, *_ = _HEADER.unpack_from(data)
    return (magic == _MAGIC and left in (0, HAND_BYTES) and right in (0, HAND_BYTES)
            and len(data) == _HEADER.size + left + right)


def unpack_deformation(stamp, data):
    if not complete_deformation(data):
        raise ValueError("invalid deformation payload")
    _, left, right, *flags = _HEADER.unpack_from(data)
    view = memoryview(data)
    return SimpleNamespace(timestamp_ns=int(stamp), fingertip_count=5, height=240, width=240,
                           left=view[_HEADER.size:_HEADER.size + left],
                           right=view[_HEADER.size + left:], left_valid=flags[:5],
                           right_valid=flags[5:])


class DeformationWriter:
    def __init__(self, prefix=DEFAULT_PREFIX, directory="/dev/shm", slot_count=16):
        self.ring = SharedCameraWriter(prefix, _HEADER.size + 2 * HAND_BYTES, directory,
                                       slot_count, reserve_space=True, names=("deform",))

    def write(self, msg):
        self.ring.write("deform", msg.timestamp_ns, pack_deformation(msg))

    def close(self):
        self.ring.close()


class DeformationReader:
    def __init__(self, prefix=DEFAULT_PREFIX, directory="/dev/shm"):
        self.ring = SharedCameraReader(prefix, directory, encoding="deformation_v1",
                                       names=("deform",), validate_payload=complete_deformation)

    @property
    def dropped_frames(self):
        return self.ring.dropped_frames["deform"]

    def read_pending(self):
        return [unpack_deformation(stamp, value[1])
                for stamp, _, value in self.ring.read_entries() if value[2]]

    def close(self):
        self.ring.close()
