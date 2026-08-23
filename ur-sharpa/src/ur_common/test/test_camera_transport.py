import os

from ur_common.camera_transport import SharedCameraReader, SharedCameraWriter


def test_latest_camera_frames_round_trip(tmp_path):
    writer = SharedCameraWriter(prefix="test_camera", max_payload_bytes=64, directory=tmp_path)
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        writer.write("ego", 11, b"ego")
        writer.write("left_wrist", 12, b"left")
        writer.write("right_wrist", 13, b"right")
        frame = reader.read()
        assert (frame.ego, frame.left_wrist, frame.right_wrist) == (b"ego", b"left", b"right")
        assert (frame.ego_timestamp_ns, frame.left_wrist_timestamp_ns, frame.right_wrist_timestamp_ns) == (11, 12, 13)
        writer.write("right_wrist", 0, b"", valid=False)
        assert reader.read().right_wrist_valid is False
    finally:
        reader.close()
        writer.close()


def test_reader_reopens_restarted_writer(tmp_path):
    writer = SharedCameraWriter(prefix="test_camera", max_payload_bytes=64, directory=tmp_path)
    reader = SharedCameraReader(prefix="test_camera", directory=tmp_path)
    try:
        writer.write("ego", 11, b"old")
        assert reader.read().ego == b"old"
        writer.close()
        writer = SharedCameraWriter(prefix="test_camera", max_payload_bytes=64, directory=tmp_path)
        writer.write("ego", 21, b"new")
        frame = reader.read()
        assert frame.ego == b"new"
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
        old_writer.write("ego", 11, b"old")
        assert reader.read().ego == b"old"
        new_writer = SharedCameraWriter(
            prefix="test_camera", max_payload_bytes=128, directory=tmp_path
        )
        new_writer.write("ego", 21, b"new")
        assert reader.read().ego == b"new"
        old_writer.close()
        new_writer.write("ego", 22, b"still-new")
        frame = reader.read()
        assert frame.ego == b"still-new"
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
        writer.write("ego", 11, b"old")
        assert reader.read().ego == b"old"
        os.truncate(tmp_path / "test_camera_ego", 0)
        frame = reader.read()
        assert frame is not None
        assert frame.ego_valid is False
    finally:
        reader.close()
        writer.close()
