from types import SimpleNamespace
import pytest

from ur_common.deformation_transport import (
    DeformationReader, DeformationWriter, HAND_BYTES, pack_deformation,
    unpack_deformation, complete_deformation,
)


def frame(stamp, left=11, right=22):
    return SimpleNamespace(timestamp_ns=stamp, left=bytes([left])*HAND_BYTES,
                           right=bytes([right])*HAND_BYTES,
                           left_valid=[True, False, True, True, True], right_valid=[True]*5)


def test_ring_burst_keeps_pixels_order_flags_and_owned_buffers(tmp_path):
    writer = DeformationWriter(directory=tmp_path, slot_count=4)
    reader = DeformationReader(directory=tmp_path)
    try:
        for i in range(3):
            writer.write(frame(100+i, i, i+10))
        result = reader.read_pending()
        assert [m.timestamp_ns for m in result] == [100, 101, 102]
        for i, msg in enumerate(result):
            assert bytes(msg.left) == bytes([i])*HAND_BYTES
            assert bytes(msg.right) == bytes([i+10])*HAND_BYTES
            assert msg.left_valid == [True, False, True, True, True]
        for i in range(10):
            writer.write(frame(200+i))
        assert bytes(result[0].left) == bytes(HAND_BYTES)  # Cannot tear on slot reuse.
        assert len(reader.read_pending()) == 4
        assert reader.dropped_frames == 6
        assert reader.read_pending() == []
    finally:
        reader.close()
        writer.close()


def test_empty_hand_is_preserved_and_truncated_data_is_rejected():
    msg = frame(1)
    msg.left = b""
    msg.left_valid = [False]*5
    payload = pack_deformation(msg)
    restored = unpack_deformation(1, payload)
    assert len(restored.left) == 0 and not any(restored.left_valid)
    assert len(restored.right) == HAND_BYTES
    assert not complete_deformation(payload[:-1])
    with pytest.raises(ValueError):
        unpack_deformation(1, payload[:-1])
    msg.right = b"bad"
    with pytest.raises(ValueError):
        pack_deformation(msg)


def test_reader_survives_writer_restart_without_stale_frames(tmp_path):
    reader = DeformationReader(directory=tmp_path)
    writer = DeformationWriter(directory=tmp_path)
    try:
        writer.write(frame(10))
        assert reader.read_pending()[0].timestamp_ns == 10
        writer.close()
        writer = DeformationWriter(directory=tmp_path)
        writer.write(frame(20))
        assert reader.read_pending()[0].timestamp_ns == 20
    finally:
        reader.close()
        writer.close()
