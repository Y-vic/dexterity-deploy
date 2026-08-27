import os

from ur_common.camera_transport import SharedCameraReader, SharedCameraWriter


def jpeg(payload: bytes) -> bytes:
    return b"\xff\xd8" + payload + b"\xff\xd9"


def test_latest_camera_frames_round_trip(tmp_path):
    writer = SharedCameraWriter(prefix="test_camera", max_payload_bytes=64, directory=tmp_path)
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        writer.write("ego", 11, jpeg(b"ego"))
        writer.write("left_wrist", 12, jpeg(b"left"))
        writer.write("right_wrist", 13, jpeg(b"right"))
        frame = reader.read()
        assert (frame.ego, frame.left_wrist, frame.right_wrist) == (
            jpeg(b"ego"), jpeg(b"left"), jpeg(b"right")
        )
        assert (frame.ego_timestamp_ns, frame.left_wrist_timestamp_ns, frame.right_wrist_timestamp_ns) == (11, 12, 13)
        writer.write("right_wrist", 0, b"", valid=False)
        assert reader.read().right_wrist_valid is False
    finally:
        reader.close()
        writer.close()


def test_pending_reader_preserves_burst_frames_in_order(tmp_path):
    writer = SharedCameraWriter(
        prefix="test_camera",
        max_payload_bytes=64,
        directory=tmp_path,
        slot_count=4,
    )
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        for timestamp in (11, 12, 13):
            writer.write("ego", timestamp, jpeg(str(timestamp).encode()))

        frames = reader.read_pending()

        assert [frame.ego_timestamp_ns for frame in frames] == [11, 12, 13]
        assert [frame.ego for frame in frames] == [
            jpeg(b"11"),
            jpeg(b"12"),
            jpeg(b"13"),
        ]
        assert reader.read_pending() == []
        assert reader.read().ego_timestamp_ns == 13
    finally:
        reader.close()
        writer.close()


def test_pending_reader_reports_ring_overrun(tmp_path):
    writer = SharedCameraWriter(
        prefix="test_camera",
        max_payload_bytes=64,
        directory=tmp_path,
        slot_count=2,
    )
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        writer.write("ego", 11, jpeg(b"11"))
        assert [frame.ego_timestamp_ns for frame in reader.read_pending()] == [11]

        for timestamp in (12, 13, 14):
            writer.write("ego", timestamp, jpeg(str(timestamp).encode()))

        frames = reader.read_pending()
        assert [frame.ego_timestamp_ns for frame in frames] == [13, 14]
        assert reader.dropped_frames["ego"] == 1
    finally:
        reader.close()
        writer.close()


def test_pending_camera_sets_never_move_another_view_backwards(tmp_path):
    writer = SharedCameraWriter(
        prefix="test_camera",
        max_payload_bytes=64,
        directory=tmp_path,
        slot_count=4,
    )
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        writer.write("ego", 20, jpeg(b"ego-20"))
        writer.write("left_wrist", 10, jpeg(b"left-10"))
        writer.write("ego", 30, jpeg(b"ego-30"))
        writer.write("left_wrist", 25, jpeg(b"left-25"))

        frames = reader.read_pending()
        for name in ("ego", "left_wrist"):
            timestamps = [
                getattr(frame, f"{name}_timestamp_ns")
                for frame in frames
                if getattr(frame, f"{name}_valid")
            ]
            assert timestamps == sorted(timestamps)
    finally:
        reader.close()
        writer.close()


def test_reader_reopens_restarted_writer(tmp_path):
    writer = SharedCameraWriter(prefix="test_camera", max_payload_bytes=64, directory=tmp_path)
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        writer.write("ego", 11, jpeg(b"old"))
        assert reader.read().ego == jpeg(b"old")
        writer.close()
        writer = SharedCameraWriter(prefix="test_camera", max_payload_bytes=64, directory=tmp_path)
        writer.write("ego", 21, jpeg(b"new"))
        frame = reader.read()
        assert frame.ego == jpeg(b"new")
        assert frame.ego_timestamp_ns == 21
    finally:
        reader.close()
        writer.close()


def test_new_writer_atomically_replaces_files_without_breaking_live_reader(tmp_path):
    old_writer = SharedCameraWriter(
        prefix="test_camera", max_payload_bytes=64, directory=tmp_path
    )
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    new_writer = None
    try:
        old_writer.write("ego", 11, jpeg(b"old"))
        assert reader.read().ego == jpeg(b"old")
        new_writer = SharedCameraWriter(
            prefix="test_camera", max_payload_bytes=128, directory=tmp_path
        )
        new_writer.write("ego", 21, jpeg(b"new"))
        assert reader.read().ego == jpeg(b"new")
        old_writer.close()
        new_writer.write("ego", 22, jpeg(b"still-new"))
        frame = reader.read()
        assert frame.ego == jpeg(b"still-new")
        assert frame.ego_timestamp_ns == 22
    finally:
        reader.close()
        old_writer.close()
        if new_writer is not None:
            new_writer.close()


def test_reader_survives_legacy_in_place_truncation(tmp_path):
    writer = SharedCameraWriter(
        prefix="test_camera", max_payload_bytes=64, directory=tmp_path
    )
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        writer.write("ego", 11, jpeg(b"old"))
        assert reader.read().ego == jpeg(b"old")
        os.truncate(tmp_path / "test_camera_ego", 0)
        frame = reader.read()
        assert frame is not None
        assert frame.ego_valid is False
    finally:
        reader.close()
        writer.close()


def test_reader_rejects_truncated_jpeg_payload(tmp_path):
    writer = SharedCameraWriter(
        prefix="test_camera", max_payload_bytes=64, directory=tmp_path
    )
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        writer.write("ego", 11, b"\xff\xd8truncated")
        frame = reader.read()
        assert frame is not None
        assert frame.ego_valid is False
        assert frame.ego == b""
    finally:
        reader.close()
        writer.close()
