import hashlib
import json

import numpy as np
from rclpy.serialization import serialize_message
from ur_msgs.msg import HandStateFrame, TauFrame, WrenchFrame

from recording_node.message_journal import MessageJournal
from recording_node.recording_audit import audit_episode, nearest_unique


def test_recovery_matches_original_cdr_once_without_changing_episode(tmp_path):
    (tmp_path/'timeline').mkdir()
    stamps = 1_000_000_000+np.rint(np.arange(6)*1e9/30).astype(np.int64)
    np.savez(tmp_path/'timeline/clock.npz', time_unix_ns=stamps)
    np.savez(tmp_path/'timeline/source_quality.npz', source_names=['camera', 'ur', 'hand', 'tau', 'wrench', 'deform'],
             source_timestamp_ns=np.full((6, 6), stamps[0], dtype=np.int64), valid=np.ones((6, 6), bool))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(sample_rate_hz=30., quality=dict(
        ok=True, frame_count=2, dropped=4, warnings=[dict(message='stale')]*4))))
    (tmp_path/'validation.json').write_text(json.dumps(dict(ok=False, items=[])))
    journal = MessageJournal(tmp_path/'raw')
    for index, stamp in enumerate(stamps):
        messages = [('/sharpa/state', HandStateFrame(timestamp_ns=int(stamp), q=[float(index)]*44,
            q_cmd=[float(index)+.5]*44, q_cmd_timestamp_ns=int(stamp)-1000,
            q_cmd_valid=True, q_cmd_source=3, is_hil=index >= 3, left_valid=True, right_valid=True)),
            ('/sharpa/tau', TauFrame(timestamp_ns=int(stamp), left=[float(index)]*22,
                                   right=[-float(index)]*22, left_valid=True, right_valid=True)),
            ('/sharpa/wrench', WrenchFrame(timestamp_ns=int(stamp), left=[float(index)]*30,
                right=[-float(index)]*30, left_valid=[True]*5, right_valid=[True]*5))]
        for topic, message in messages:
            assert journal.append(topic, 'ur_msgs/msg/'+type(message).__name__, serialize_message(message),
                                  int(stamp), received_unix_ns=int(stamps[-1])+100_000_000)
    assert journal.close()
    def hashes():
        return {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tmp_path.rglob('*') if p.is_file()}
    before = hashes()
    report, arrays = audit_episode(tmp_path, recover=True)
    assert before == hashes()
    assert report['manifest_quality_ok'] and not report['validation_ok']
    assert report['warning_counts'] == {'stale': 4}
    assert report['sampled']['hand']['repeated_valid_rows'] == 5
    assert report['sampled']['hand']['source_age_ms']['max'] == 166.667
    assert report['feedback_alignment']['source_skew_ms']['max'] == 0.
    for key in ('hand', 'tau', 'wrench'):
        assert report['recovery'][key]['unique_frames'] == report['recovery'][key]['matched_frames'] == 6
    np.testing.assert_array_equal(arrays['q'][:, 0], np.arange(6))
    np.testing.assert_array_equal(arrays['q_cmd'][:, 0], np.arange(6)+.5)
    np.testing.assert_array_equal(arrays['force'][:, 1, 0, 0], -np.arange(6))
    assert arrays['is_hil'].tolist() == [False]*3+[True]*3
    assert arrays['q_cmd_valid'].all() and arrays['wrench_valid'].all()


def test_recovery_does_not_reuse_source_or_invent_a_late_sample():
    stamps = np.array([1_000_000_000, 1_033_333_333], dtype=np.int64)
    targets = np.array([1_000_000_000, 1_001_000_000, 1_033_333_333], dtype=np.int64)
    received = stamps+300_000_000
    args = dict(max_skew_ns=20_000_000, delay_ns=500_000_000)
    assert nearest_unique(stamps, received, targets, **args).tolist() == [0, -1, 1]
    args['delay_ns'] = 100_000_000
    assert nearest_unique(stamps, received, targets, **args).tolist() == [-1, -1, -1]


def test_duplicate_source_timestamp_does_not_hide_a_new_frame():
    stamps = np.array([100, 100, 110], dtype=np.int64)
    assert nearest_unique(stamps, stamps, np.array([100, 101]),
                          max_skew_ns=20, delay_ns=20).tolist() == [0, 2]
