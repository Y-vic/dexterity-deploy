import json

import pytest
from recording_node.execution_audit import archive_actor_trace
from recording_node.message_journal import MessageJournal
from ur_common.rollout_capture import RolloutCapture


def trace(tmp_path):
    source, episode = tmp_path / 'trace', tmp_path / 'episode'
    source.mkdir(); episode.mkdir()
    (source / 'manifest.json').write_text(json.dumps(dict(schema='sharpa.actor_ik.execution_trace.v2', state='complete')))
    rows = [dict(event='action_proposed', action_id='a', phase='handback', request_id='request',
                 chunk_id='chunk', chunk_step=3, control_observation=dict(rollout_id='episode')),
            dict(event='command_published', action_id='a', revision=0, step_index=0,
                 timestamp_ns=123, expected_devices=['ur', 'sharpa'],
                 left_arm_joint=[0]*6, right_arm_joint=[0]*6, hand_joint=[0]*44),
            dict(event='hardware_command_result_received', action_id='a', device='ur', success=True)]
    (source / 'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return source, episode


def test_archive_joins_final_raw_receipt_and_records_missing_per_device(tmp_path):
    from rclpy.serialization import serialize_message
    from ur_msgs.msg import HardwareCommandResult
    from std_msgs.msg import String
    source, episode = trace(tmp_path)
    journal = MessageJournal(episode / 'raw')
    msg = HardwareCommandResult(source=3, action_id='a', device='sharpa', success=False,
                                failure_code='sdk_failed', timestamp_ns=456)
    journal.append('/sharpa/command_result', 'ur_msgs/msg/HardwareCommandResult', serialize_message(msg))
    for side in ('left', 'right'):
        event = dict(event='ur_sdk_arm_command_completed', action_id='a', side=side, success=True,
                     target_rad=[0]*6, timestamp_ns=124)
        journal.append('/deploy/execution_event', 'std_msgs/msg/String', serialize_message(String(data=json.dumps(event))))
    assert journal.close()
    result = archive_actor_trace(source, episode, 'episode')
    assert result['commands'] == 1 and not result['missing_receipts']
    assert not result['missing_device_receipts']  # A failed pair never fabricates side success.
    row = json.loads((episode / 'actor/execution.jsonl').read_text())
    assert row['receipts']['sharpa']['success'] is False
    assert row['device_receipts']['left_arm']['target_rad'] == [0]*6
    assert row['device_receipts']['right_hand'] is None
    assert row['context']['chunk_step'] == 3 and row['context']['phase'] == 'handback'
    assert (episode / 'actor/events.jsonl').read_bytes() == (source / 'events.jsonl').read_bytes()


def test_missing_receipts_are_unknown_and_foreign_rollout_is_excluded(tmp_path):
    source, episode = trace(tmp_path)
    result = archive_actor_trace(source, episode, 'episode')
    assert result['missing_receipts'] == [dict(action_id='a', revision=0, step_index=0, device='sharpa')]
    assert len(result['missing_device_receipts']) == 2
    assert archive_actor_trace(source, episode, 'other')['commands'] == 0


def test_partial_trace_is_an_explicit_archive_failure(tmp_path):
    source, episode = trace(tmp_path)
    with (source / 'events.jsonl').open('a') as stream:
        stream.write('{"partial":')
    with pytest.raises(OSError, match='incomplete JSONL'):
        archive_actor_trace(source, episode, 'episode')


def test_termination_preserves_annotation_and_action_identity():
    requests = []
    capture = RolloutCapture(requests.append)
    capture.tick(True, now=1)
    identity = capture.rollout_id
    capture.was_ready = True
    capture.finish('fail', action_id='last', source='learning_command')
    capture.tick(False, now=2)
    terminal = requests[-1]
    assert terminal['finished_rollout_id'] == identity
    assert terminal['termination']['outcome'] == 'failure'
    assert terminal['termination']['action_id'] == 'last'
    capture.tick(True, now=3)
    assert requests[-1]['termination'] == {} and requests[-1]['category'] == ''


def test_recorder_archives_trace_checksums_and_preserves_end_time_on_classification(tmp_path, monkeypatch):
    from recording_node.recording_node import RecorderCore
    from test_recording_core import msgset
    import recording_node.recording_node as module
    monkeypatch.setattr(module, '_fk', lambda joints: ([0.0]*15, [0.0]*15))
    source, _ = trace(tmp_path)
    with (source / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event='hardware_command_result_received', action_id='a', device='sharpa', success=True))+'\n')
    sides = [dict(event=f'{device}_sdk_{group}_command_completed', action_id='a', side=side, success=True)
             for device, group in (('ur', 'arm'), ('sharpa', 'hand')) for side in ('left', 'right')]
    (source / 'node_events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in sides))
    core = RecorderCore(tmp_path / 'recordings', record_tactile=False, rollout_recording=True)
    try:
        messages = msgset()
        messages['hand'].q_cmd_source = 3
        messages['hand'].is_hil = True
        for name, message in messages.items():
            core.update(name, message, received=1_000_000_000)
        core.start()
        core.inference_rollout_id = 'episode'
        core.actor_trace_path = str(source)
        core.execution_audit_required = True
        core.termination = dict(outcome='truncated', timestamp_ns=789, action_id='a',
                                annotation_source='actor_stop', annotated_unix_ns=790)
        assert core.sample(now=1_000_000_000)
        saved = core.stop()
        assert saved is not None
        checksums = (saved / 'checksums.sha256').read_text()
        assert 'actor/execution.jsonl' in checksums and 'termination.json' in checksums
        classified = core.classify('success')
        termination = json.loads((classified / 'termination.json').read_text())
        assert termination['outcome'] == 'success' and termination['timestamp_ns'] == 789
        assert termination['action_id'] == 'a' and termination['last_command']['timestamp_ns'] == 123
        assert termination['annotation_source'] == 'recording_classification'
        assert termination['previous_annotation']['outcome'] == 'truncated'
    finally:
        core.writer.close()


def test_ros_adapter_registers_and_serializes_both_command_and_receipt_topics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rclpy.serialization import deserialize_message
    import recording_node.recording_node as module
    from ur_msgs.msg import ArmCommand, HandCommand, HardwareCommandResult, WrenchFrame
    parameters = dict(recording_root=str(tmp_path), video_backend='mjpeg', shared_memory_prefix='audit_test_no_camera',
                      deformation_shm_prefix='audit_test_no_deformation')
    subscriptions = {}
    monkeypatch.setattr(module.Node, '__init__', lambda self, name: None)
    monkeypatch.setattr(module.Node, 'declare_parameter', lambda self, name, value: parameters.setdefault(name, value))
    monkeypatch.setattr(module.Node, 'get_parameter', lambda self, name: SimpleNamespace(value=parameters[name]))
    monkeypatch.setattr(module.Node, 'create_publisher', lambda self, cls, topic, qos: SimpleNamespace(publish=lambda msg: None))
    monkeypatch.setattr(module.Node, 'create_subscription',
        lambda self, cls, topic, callback, qos: subscriptions.setdefault(topic, (cls, callback)))
    monkeypatch.setattr(module.Node, 'create_timer', lambda self, period, callback: None)
    node = module.RecordingNode()
    received = []
    node.core.record_message = lambda *args: received.append(args)
    node.core.recording = True
    try:
        assert parameters['skip_tactile'] is False
        assert subscriptions['/sharpa/wrench'][0] is WrenchFrame
        wrench = WrenchFrame(timestamp_ns=123, left=[float(i) for i in range(30)],
                             right=[-float(i) for i in range(30)], left_valid=[True]*5, right_valid=[True]*5)
        subscriptions['/sharpa/wrench'][1](wrench)
        captured = deserialize_message(received[-1][2], WrenchFrame)
        assert list(captured.left[3:6]) == [3., 4., 5.]  # Tx/Ty/Tz survive raw CDR too.
        for topic, cls in (('/ur/command', ArmCommand), ('/sharpa/command', HandCommand),
                           ('/ur/command_result', HardwareCommandResult), ('/sharpa/command_result', HardwareCommandResult)):
            assert subscriptions[topic][0] is cls
            subscriptions[topic][1](cls(action_id='command-1'))
            assert received[-1][0] == topic
            assert received[-1][1] == 'ur_msgs/msg/' + cls.__name__
            assert deserialize_message(received[-1][2], cls).action_id == 'command-1'
    finally:
        node.core.recording = False
        node._camera_poll_stop.set()
        node._camera_poll_thread.join()
        node.camera_reader.close()
        node.deformation_reader.close()
        node.core.writer.close()


def test_remote_chunk_steps_stay_with_their_rollout_across_pause(tmp_path):
    source, episode = trace(tmp_path)
    rows = []
    for step, rollout in enumerate(('first', 'second')):
        identity = dict(action_id='same-remote-chunk', revision=0, step_index=step)
        rows.extend([
            dict(event='action_proposed', **identity, control_observation=dict(rollout_id=rollout)),
            dict(event='command_published', **identity, timestamp_ns=100+step, expected_devices=['sharpa']),
            dict(event='hardware_command_result_received', **identity, device='sharpa', success=False),
        ])
    (source/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for rollout, expected_step in (('first',0), ('second',1)):
        result = archive_actor_trace(source, episode, rollout)
        assert result['commands'] == 1
        event = json.loads((episode/'actor/execution.jsonl').read_text())
        assert event['command']['step_index'] == expected_step
        assert event['context']['control_observation']['rollout_id'] == rollout
