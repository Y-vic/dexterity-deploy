"""Read-only audit of arrival bursts, sampled cache repeats and source recovery."""
from collections import Counter
import json
from pathlib import Path
import sqlite3

import numpy as np


def percentiles(values):
    values = np.asarray(values)
    return dict(zip(('min', 'p50', 'p95', 'p99', 'max'),
                    np.percentile(values, [0, 50, 95, 99, 100]).round(3).tolist())) if values.size else {}


def nearest_unique(stamps, received, targets, *, max_skew_ns, delay_ns):
    """Match real source frames once; missing data stays missing, never interpolated."""
    stamps, received, targets = (np.asarray(v, dtype=np.int64) for v in (stamps, received, targets))
    if (len(stamps) != len(received) or np.any(np.diff(stamps) < 0)
            or np.any(np.diff(targets) < 0) or max_skew_ns <= 0 or delay_ns < 0):
        raise ValueError('source frames must be in timestamp order')
    indices = np.full(len(targets), -1, dtype=np.int64)
    last = -1
    for row, target in enumerate(targets):
        lower = max(last+1, int(np.searchsorted(stamps, int(target)-max_skew_ns)))
        if last >= 0:
            lower = max(lower, int(np.searchsorted(stamps, stamps[last], side='right')))
        upper = int(np.searchsorted(stamps, int(target)+max_skew_ns, side='right'))
        candidates = [i for i in range(lower, upper) if received[i] <= target+delay_ns]
        if candidates:
            chosen = min(candidates, key=lambda i: abs(int(stamps[i])-int(target)))
            if last >= 0 and stamps[chosen] <= stamps[last]:
                continue
            indices[row] = last = chosen
    return indices


def audit_episode(directory, *, recover=False, delay_s=.5, max_skew_ms=20.):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    validation = json.loads((directory/'validation.json').read_text())
    with np.load(directory/'timeline/clock.npz') as data:
        clock = data['time_unix_ns'].copy()
    with np.load(directory/'timeline/source_quality.npz') as data:
        names = data['source_names'].tolist()
        sampled = data['source_timestamp_ns'].copy()
        sampled_valid = data['valid'].copy()
    frequency = float(manifest['sample_rate_hz'])
    duration = (int(clock[-1])-int(clock[0]))/1e9+1/frequency if len(clock) else 0.
    quality = manifest['quality']
    result = dict(input_directory=str(directory), input_modified=False, duration_s=duration,
        timeline_rows=len(clock), training_rows=quality['frame_count'], dropped=quality['dropped'],
        manifest_quality_ok=quality['ok'], validation_ok=validation['ok'],
        warning_counts=dict(Counter(row['message'] for row in quality.get('warnings', []))),
        sampling_work=quality.get('sampling_work', {}), video=quality.get('video_encoding', {}),
        validation_items=validation['items'], raw={}, sampled={},
        recovery=dict(delay_s=delay_s, max_skew_ms=max_skew_ms, interpolation=False))
    for column, name in enumerate(names):
        stamps = sampled[:, column]
        valid = sampled_valid[:, column] & (stamps > 0)
        unique = len(np.unique(stamps[valid]))
        result['sampled'][name] = dict(valid_rows=int(valid.sum()), unique_valid_frames=unique,
            unique_valid_hz=unique/duration if duration else 0.,
            repeated_valid_rows=int(valid.sum())-unique,
            source_age_ms=percentiles((clock[valid]-stamps[valid])/1e6))
    columns = [names.index(key) for key in ('ur', 'hand', 'tau', 'wrench')]
    valid = np.all(sampled_valid[:, columns] & (sampled[:, columns] > 0), axis=1)
    result['feedback_alignment'] = dict(valid_rows=int(valid.sum()),
        source_skew_ms=percentiles(np.ptp(sampled[valid][:, columns], axis=1)/1e6))
    arrays = dict(time_unix_ns=clock)
    connection = sqlite3.connect((directory/'raw/messages.sqlite3').as_uri()+'?mode=ro', uri=True)
    try:
        topics = [row[0] for row in connection.execute('SELECT DISTINCT topic FROM messages')]
        for topic in topics:
            rows = connection.execute('SELECT received_unix_ns,source_timestamp_ns FROM messages WHERE topic=? ORDER BY id', (topic,)).fetchall()
            received, stamps = np.asarray(rows, dtype=np.int64).T
            gaps = np.diff(received)/1e6
            summary = dict(messages=len(rows), received_hz=len(rows)/duration,
                receive_gap_ms=percentiles(gaps), receive_bursts_lt_1ms=int(np.count_nonzero(gaps < 1.)),
                receive_stalls_gt_100ms=int(np.count_nonzero(gaps > 100.)))
            if np.all(stamps > 0):
                summary.update(unique_source_frames=len(np.unique(stamps)),
                    source_gap_ms=percentiles(np.diff(stamps)/1e6),
                    delivery_latency_ms=percentiles((received-stamps)/1e6))
            key = {'/ur/state': 'ur', '/sharpa/state': 'hand', '/sharpa/tau': 'tau', '/sharpa/wrench': 'wrench'}.get(topic)
            if key:
                last = np.searchsorted(received, clock, side='right')-1
                valid = last >= 0
                expected = np.zeros(len(clock), dtype=np.int64)
                expected[valid] = stamps[last[valid]]
                summary['saved_matches_latest_arrival_percent'] = (float(np.mean(
                    sampled[:, names.index(key)][valid] == expected[valid])*100) if valid.any() else None)
            result['raw'][topic] = summary
        if recover:
            from rclpy.serialization import deserialize_message
            from ur_msgs.msg import HandStateFrame, TauFrame, WrenchFrame
            for key, topic, cls in (('hand', '/sharpa/state', HandStateFrame),
                                    ('tau', '/sharpa/tau', TauFrame), ('wrench', '/sharpa/wrench', WrenchFrame)):
                rows = connection.execute('SELECT source_timestamp_ns,received_unix_ns,cdr FROM messages WHERE topic=? ORDER BY source_timestamp_ns,id', (topic,)).fetchall()
                if not rows:
                    raise ValueError('no raw feedback messages for '+topic)
                stamps = np.asarray([row[0] for row in rows], dtype=np.int64)
                received = np.asarray([row[1] for row in rows], dtype=np.int64)
                match = nearest_unique(stamps, received, clock, max_skew_ns=round(max_skew_ms*1e6), delay_ns=round(delay_s*1e9))
                matched = match >= 0
                arrays[key+'_matched'] = matched
                arrays[key+'_source_timestamp_ns'] = np.where(matched, stamps[np.maximum(match, 0)], 0)
                arrays[key+'_received_unix_ns'] = np.where(matched, received[np.maximum(match, 0)], 0)
                if key == 'hand':
                    arrays['q'] = np.full((len(clock), 44), np.nan, np.float32)
                    arrays['q_cmd'] = np.full((len(clock), 44), np.nan, np.float32)
                    arrays['q_valid'] = np.zeros(len(clock), bool)
                    arrays['q_cmd_valid'] = np.zeros(len(clock), bool)
                    arrays['q_cmd_timestamp_ns'] = np.zeros(len(clock), np.int64)
                    arrays['q_cmd_source'] = np.zeros(len(clock), np.uint8)
                    arrays['q_cmd_invalid_reason'] = np.full(len(clock), '', dtype='U32')
                    arrays['is_hil'] = np.zeros(len(clock), bool)
                else:
                    shape = (len(clock), 2, 5, 6) if key == 'wrench' else (len(clock), 44)
                    arrays[key] = np.full(shape, np.nan, np.float32)
                    arrays[key+'_valid'] = np.zeros((len(clock), 2, 5) if key == 'wrench' else len(clock), bool)
                for row in np.flatnonzero(matched):
                    message = deserialize_message(rows[match[row]][2], cls)
                    if key == 'hand':
                        arrays['q'][row], arrays['q_cmd'][row] = message.q, message.q_cmd
                        arrays['q_valid'][row] = message.left_valid and message.right_valid and np.isfinite(message.q).all()
                        arrays['q_cmd_valid'][row] = bool(arrays['q_valid'][row] and message.q_cmd_valid
                            and np.isfinite(message.q_cmd).all() and message.q_cmd_source in (2, 3)
                            and 0 < message.q_cmd_timestamp_ns <= message.timestamp_ns)
                        arrays['q_cmd_timestamp_ns'][row] = message.q_cmd_timestamp_ns
                        arrays['q_cmd_source'][row] = message.q_cmd_source
                        arrays['q_cmd_invalid_reason'][row] = message.q_cmd_invalid_reason
                        arrays['is_hil'][row] = bool(arrays['q_cmd_valid'][row] and message.is_hil)
                    elif key == 'tau':
                        arrays[key][row] = [*message.left, *message.right]
                        arrays[key+'_valid'][row] = message.left_valid and message.right_valid and np.isfinite(arrays[key][row]).all()
                    else:
                        arrays[key][row] = np.asarray([message.left, message.right]).reshape(2, 5, 6)
                        arrays[key+'_valid'][row] = np.asarray([message.left_valid, message.right_valid]) & np.isfinite(arrays[key][row]).all(axis=2)
                result['recovery'][key] = dict(matched_frames=int(matched.sum()),
                    unique_frames=len(np.unique(arrays[key+'_source_timestamp_ns'][matched])),
                    matched_hz=float(matched.sum()/duration), missing_frames=int((~matched).sum()))
                flags = arrays['q_valid' if key == 'hand' else key+'_valid']
                valid_rows = np.all(flags, axis=(1, 2)) if key == 'wrench' else flags
                result['recovery'][key]['valid_frames'] = int(valid_rows.sum())
            arrays['force'], arrays['torque'] = arrays['wrench'][..., :3], arrays['wrench'][..., 3:]
    finally:
        connection.close()
    return result, arrays
