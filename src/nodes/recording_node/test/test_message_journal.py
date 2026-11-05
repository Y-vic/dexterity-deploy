import json
import sqlite3
import threading
import time

import pytest

from recording_node.message_journal import MessageJournal
from recording_node.recording_node import RecorderCore


def records(directory):
    with sqlite3.connect(directory / "messages.sqlite3") as db:
        return db.execute("SELECT topic, receive_sequence, source_timestamp_ns, cdr "
                          "FROM messages ORDER BY id").fetchall()


def test_burst_preserves_every_payload_and_duplicate_source_timestamp(tmp_path):
    journal = MessageJournal(tmp_path)
    try:
        for i in range(1000):
            assert journal.append("/sharpa/state", "ur_msgs/msg/HandStateFrame",
                                  i.to_bytes(4, "little"), source_timestamp_ns=12)
        assert journal.close()
        rows = records(tmp_path)
        assert len(rows) == 1000
        assert [r[1] for r in rows] == list(range(1, 1001))
        assert [r[3] for r in rows] == [i.to_bytes(4, "little") for i in range(1000)]
        assert {r[2] for r in rows} == {12}
        assert journal.status()["complete"]
        assert not journal.append("/sharpa/state", "test", b"after stop")
    finally:
        journal.close()


def test_background_commit_window_batches_paced_events_and_drains_on_close(tmp_path):
    journal = MessageJournal(tmp_path)
    for index in range(60):
        assert journal.append('/event', 'test', index.to_bytes(4, 'little'))
        time.sleep(.001)
    assert journal.close()
    assert [row[3] for row in records(tmp_path)] == [i.to_bytes(4, 'little') for i in range(60)]
    assert journal.status()['max_batch_size'] > 1
    assert journal.status()['commits'] < 60
    assert journal.status()['complete']


def test_sequence_is_per_topic_and_invalid_bytes_are_not_rewritten(tmp_path):
    journal = MessageJournal(tmp_path)
    for topic, payload in [("/state", b"\x00\x00\xc0\x7f"), ("/command", b"bad"), ("/state", b"x")]:
        assert journal.append(topic, "test", payload, received_unix_ns=100,
                              received_monotonic_ns=200)
    assert journal.close()
    assert [(r[0], r[1]) for r in records(tmp_path)] == [("/state", 1), ("/command", 1), ("/state", 2)]
    assert records(tmp_path)[0][3] == b"\x00\x00\xc0\x7f"


def test_slow_disk_budget_failure_is_explicit_and_accepted_data_drains(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = MessageJournal._write_batch

    def blocked(connection, batch):
        entered.set()
        assert release.wait(5)
        original(connection, batch)

    monkeypatch.setattr(MessageJournal, "_write_batch", staticmethod(blocked))
    journal = MessageJournal(tmp_path, max_pending_bytes=1024)
    try:
        assert journal.append("/state", "test", b"a" * 100)
        assert entered.wait(2)
        assert not journal.append("/state", "test", b"b" * 100)
        assert not journal.close(timeout=0)
        assert not (tmp_path / "messages_meta.json").exists()
    finally:
        release.set()
        assert journal.close()
    summary = json.loads((tmp_path / "messages_meta.json").read_text())
    assert summary["received"] == {"/state": 2}
    assert summary["written"] == {"/state": 1}
    assert summary["rejected"] == {"/state": 1}
    assert not summary["complete"]
    assert summary["peak_pending_bytes"] <= 1024
    assert records(tmp_path)[0][3] == b"a" * 100


def test_disk_failure_never_reports_success(tmp_path, monkeypatch):
    def broken(connection, batch):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(MessageJournal, "_write_batch", staticmethod(broken))
    journal = MessageJournal(tmp_path)
    assert journal.append("/state", "test", b"hello")
    assert journal.close()
    assert not journal.status()["complete"]
    assert "disk full" in journal.status()["errors"][0]
    assert records(tmp_path) == []


def test_overload_does_not_disable_future_recording(tmp_path):
    journal = MessageJournal(tmp_path, max_pending_bytes=1024)
    assert not journal.append("/state", "test", b"x" * 1024)
    assert journal.append("/state", "test", b"recovered")
    assert journal.close()
    assert records(tmp_path) == [("/state", 2, 0, b"recovered")]
    assert not journal.status()["complete"]


def test_database_start_failure_is_reported(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("read only")

    monkeypatch.setattr(sqlite3, "connect", broken)
    with pytest.raises(RuntimeError, match="read only"):
        MessageJournal(tmp_path)


def test_rejected_training_sample_still_preserves_received_messages(tmp_path):
    core = RecorderCore(tmp_path)
    try:
        assert core.start()
        assert core.record_message("/sharpa/state", "ur_msgs/msg/HandStateFrame", b"original")
        assert not core.sample()  # Other modalities absent; old filter still rejects.
        core.stop(save=True)
        saved = core.writer.last_result_dir
        assert records(saved / "raw") == [("/sharpa/state", 1, 0, b"original")]
        summary = json.loads((saved / "manifest.json").read_text())
        assert summary["quality"]["raw_messages"]["complete"]
        assert summary["quality"]["frame_count"] == 0
    finally:
        core.writer.close()


def test_raw_overflow_marks_saved_episode_incomplete(tmp_path):
    core = RecorderCore(tmp_path, raw_message_budget_bytes=512)
    try:
        core.start()
        assert not core.record_message("/sharpa/state", "test", b"large" * 100)
        core.sample()
        core.stop(save=True)
        saved = core.writer.last_result_dir
        quality = json.loads((saved / "manifest.json").read_text())["quality"]
        assert not quality["ok"]
        assert any("raw message journal incomplete" in e["message"] for e in quality["errors"])
        assert not (saved / "COMPLETE").exists()
    finally:
        core.writer.close()


def test_consecutive_sessions_do_not_share_messages(tmp_path):
    core = RecorderCore(tmp_path)
    try:
        saved = []
        for payload in (b"one", b"two"):
            assert core.start()
            core.record_message("/state", "test", payload)
            core.stop(save=True)
            saved.append(core.writer.last_result_dir)
        assert [records(p / "raw")[0][3] for p in saved] == [b"one", b"two"]
    finally:
        core.writer.close()


def test_drain_timeout_keeps_episode_in_place_and_allows_retry(tmp_path, monkeypatch):
    core = RecorderCore(tmp_path)
    try:
        assert core.start()
        core.record_message("/state", "test", b"pending")
        partial = core.writer.episode_dir
        journal = core.writer.message_journal
        close = journal.close
        monkeypatch.setattr(journal, "close", lambda timeout=30: False)
        with pytest.raises(RuntimeError, match="still draining"):
            core.stop()
        assert partial.is_dir()
        assert core.writer.active
        assert not core.start()
        monkeypatch.setattr(journal, "close", close)
        core.stop()
        assert records(core.writer.last_result_dir / "raw")[0][3] == b"pending"
        assert not partial.exists()
    finally:
        core.writer.close()
