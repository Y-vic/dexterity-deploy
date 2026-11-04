from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from recording_node.recording_node import RecorderCore
from test_recording_core import msgset


def event(kind, request_id='request', rollout_id='episode'):
    return json.dumps(dict(event=kind, request_id=request_id, rollout_id=rollout_id,
                           submitted_unix_ns=123, completed_unix_ns=456,
                           result={'actions': {'shape': [1, 2], 'dtype': '<f4', 'data': [[1, 2]]}}))


def test_inference_rows_bypass_invalid_training_frames_and_preserve_raw_cdr(tmp_path):
    core = RecorderCore(tmp_path, rollout_recording=True)
    try:
        assert core.start()
        core.inference_rollout_id = 'episode'
        assert core.record_inference(event('requested'), b'request cdr')
        assert core.record_inference(event('completed'), b'result cdr')
        assert not core.sample()
        core.stop()
        saved = core.writer.last_result_dir
        rows = [json.loads(line) for line in (saved / 'raw/policy_inference.jsonl').read_text().splitlines()]
        assert [row['event'] for row in rows] == ['requested', 'completed']
        assert rows[-1]['result']['actions']['data'] == [[1, 2]]
        with sqlite3.connect(saved / 'raw/messages.sqlite3') as database:
            records = database.execute('SELECT topic, cdr FROM messages ORDER BY id').fetchall()
        assert records == [('/policy/inference', b'request cdr'), ('/policy/inference', b'result cdr')]
        assert core.writer.message_journal.status()['complete']
    finally:
        core.writer.close()


def test_stop_drains_inflight_result_without_accepting_another_episode(tmp_path, monkeypatch):
    core = RecorderCore(tmp_path, rollout_recording=True)
    entered = threading.Event()
    wait_for = core._inference_condition.wait_for
    def waiting(predicate, timeout):
        entered.set()
        return wait_for(predicate, timeout)
    monkeypatch.setattr(core._inference_condition, 'wait_for', waiting)
    try:
        core.start()
        core.inference_rollout_id = 'episode'
        core.record_inference(event('requested'), b'start')
        with ThreadPoolExecutor(max_workers=1) as pool:
            saving = pool.submit(core.stop)
            assert entered.wait(2)
            assert not core.recording and not saving.done()
            assert not core.record_inference(event('completed', rollout_id='other'), b'wrong episode')
            assert not core.record_inference(event('requested', request_id='other'), b'late request')
            assert core.record_inference(event('completed'), b'output')
            saving.result(timeout=5)
        saved = core.writer.last_result_dir
        assert len((saved / 'raw/policy_inference.jsonl').read_text().splitlines()) == 2
        assert not core._pending_inferences
        assert not core.record_inference(event('completed'), b'after close')
        assert core.start()
        core.inference_rollout_id = 'next'
        assert not core.record_inference(event('completed'), b'old result')
        core.stop()
        assert not (core.writer.last_result_dir / 'raw/policy_inference.jsonl').exists()
    finally:
        core.writer.close()


def test_missing_result_marks_episode_incomplete(tmp_path, monkeypatch):
    core = RecorderCore(tmp_path, rollout_recording=True)
    try:
        core.start()
        core.inference_rollout_id = 'episode'
        core.record_inference(event('requested'), b'start')
        monkeypatch.setattr(core._inference_condition, 'wait_for', lambda *args, **kwargs: False)
        core.stop()
        quality = json.loads((core.writer.last_result_dir / 'manifest.json').read_text())['quality']
        assert not quality['ok']
        assert any('policy inference drain timed out' in error['message'] for error in quality['errors'])
    finally:
        core.writer.close()


def test_inference_queue_budget_failure_is_reported(tmp_path):
    core = RecorderCore(tmp_path, raw_message_budget_bytes=512, rollout_recording=True)
    try:
        core.start()
        core.inference_rollout_id = 'episode'
        assert not core.record_inference(event('requested'), b'request')
        assert not core.record_inference(event('failed'), b'failure')
        core.stop()
        assert not core.writer.message_journal.status()['complete']
        assert 'policy inference journal rejected an event' in core.quality_errors
    finally:
        core.writer.close()




def test_jsonl_write_failure_marks_raw_capture_incomplete(tmp_path, monkeypatch):
    core = RecorderCore(tmp_path, rollout_recording=True)
    original = Path.open
    def open_file(path, *args, **kwargs):
        if path.name == 'policy_inference.jsonl':
            raise OSError('inference disk full')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_file)
    try:
        core.start()
        core.inference_rollout_id = 'episode'
        core.record_inference(event('requested'), b'start')
        core.record_inference(event('completed'), b'output')
        core.stop()
        status = core.writer.message_journal.status()
        assert not status['complete']
        assert any('inference disk full' in error for error in status['errors'])
    finally:
        core.writer.close()
