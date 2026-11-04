"""Archive the local Actor journal and preserve explicit episode outcomes."""
import json
from pathlib import Path
import time
import sqlite3


def write_termination(directory, value):
    path = Path(directory) / 'termination.json'
    path.write_text(json.dumps({'schema': 'ur_sharpa.termination.v1', **value}, ensure_ascii=False, indent=2), encoding='utf-8')


def archive_actor_trace(source, directory, rollout_id):
    """Copy bounded file prefixes; never follow arbitrary files from the journal.

    A still-active run is explicitly a snapshot, not a finalized Actor run.
    Missing receipts are unknown (not successful and not failed).
    """
    source = Path(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('schema') != 'sharpa.actor_ik.execution_trace.v2':
        raise ValueError('unsupported Actor trace schema')
    target = Path(directory) / 'actor'
    target.mkdir(exist_ok=True)
    files = {}
    for name in ('manifest.json', 'summary.json', 'final_state.json', 'events.jsonl', 'node_events.jsonl', 'trajectories.jsonl'):
        path = source / name
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            size = path.stat().st_size
            # Stream large trajectory logs; retain complete JSONL records only.
            with (target / name).open('wb') as output:
                remaining = size
                while remaining:
                    data = stream.read(min(1024 * 1024, remaining))
                    if not data:
                        raise OSError('Actor trace shrank during archive')
                    output.write(data)
                    remaining -= len(data)
        if name.endswith('.jsonl') and size:
            with (target / name).open('rb+') as output:
                output.seek(-1, 2)
                if output.read(1) != b'\n':
                    raise OSError('Actor trace contains an incomplete JSONL record')
        files[name] = size
    commands, receipts, proposals, side_receipts = {}, {}, {}, {}
    last = None
    events = target / 'events.jsonl'
    if not events.exists():
        raise ValueError('Actor event journal missing')

    def consume(row):
        nonlocal last
        kind = row.get('event')
        key = (row.get('action_id'), row.get('revision', 0), row.get('step_index', 0))
        if kind == 'action_proposed' and row.get('control_observation', {}).get('rollout_id') == rollout_id:
            # Remote chunks may span a pause and two recordings. New events
            # bind each step; legacy one-step local events use action identity.
            proposals[key if 'step_index' in row else row['action_id']] = row
        if kind == 'command_published' and (key in proposals or key[0] in proposals):
            commands[key] = row
            last = dict(action_id=key[0], timestamp_ns=row.get('command_timestamp_ns', row['timestamp_ns']))
        elif kind in ('hardware_command_result_received', 'hardware_command_result_published'):
            receipts.setdefault(key, {})[row['device']] = row
        elif kind in ('ur_sdk_arm_command_completed', 'sharpa_sdk_hand_command_completed'):
            group = row['side'] + ('_arm' if kind.startswith('ur_') else '_hand')
            side_receipts.setdefault(key, {})[group] = row

    for name in ('events.jsonl', 'node_events.jsonl'):
        path = target / name
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    consume(json.loads(line))
    # Receipts can arrive after Actor closes its run; the recorder's final drain
    # is authoritative for those messages. CDR remains preserved in raw/.
    database = Path(directory) / 'raw/messages.sqlite3'
    if database.exists():
        from rclpy.serialization import deserialize_message
        from std_msgs.msg import String
        from ur_msgs.msg import HardwareCommandResult
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
            for topic, payload in connection.execute(
                    "SELECT topic, cdr FROM messages WHERE topic IN "
                    "('/ur/command_result', '/sharpa/command_result', '/deploy/execution_event') ORDER BY id"):
                if topic == '/deploy/execution_event':
                    consume(json.loads(deserialize_message(payload, String).data))
                else:
                    msg = deserialize_message(payload, HardwareCommandResult)
                    if msg.source == 3:
                        consume(dict(event='hardware_command_result_received', action_id=msg.action_id,
                            revision=msg.revision, step_index=msg.step_index, device=msg.device,
                            timestamp_ns=msg.timestamp_ns, success=msg.success,
                            failure_code=msg.failure_code, failure_message=msg.failure_message, dry_run=msg.dry_run))
    missing, missing_sides = [], []
    with (target / 'execution.jsonl').open('w') as output:
        for key, command in commands.items():
            expected = command.get('expected_devices', ['ur', 'sharpa'])
            ack = receipts.get(key, {})
            sides = side_receipts.get(key, {})
            for device in expected:
                if device not in ack:
                    missing.append(dict(action_id=key[0], revision=key[1], step_index=key[2], device=device))
                elif ack[device]['success'] and not command.get('dry_run'):
                    for side in ('left', 'right'):
                        group = side + ('_arm' if device == 'ur' else '_hand')
                        if group not in sides:
                            missing_sides.append(dict(action_id=key[0], group=group))
            output.write(json.dumps(dict(schema='ur_sharpa.execution_step.v1',
                action_id=key[0], context=proposals.get(key, proposals.get(key[0])), command=command,
                receipts={device: ack.get(device) for device in expected},
                device_receipts={group: sides.get(group) for group in
                    ('left_arm', 'right_arm', 'left_hand', 'right_hand')})) + '\n')
    result = dict(schema='ur_sharpa.actor_archive.v1', rollout_id=rollout_id,
        source_path=str(source), archived_unix_ns=time.time_ns(),
        source_state=manifest.get('state'), files=files, commands=len(commands),
        missing_receipts=missing, missing_device_receipts=missing_sides, last_command=last)
    (target / 'archive.json').write_text(json.dumps(result, indent=2))
    return result
