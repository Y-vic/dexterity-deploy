import threading

from quest_node.webvr_protocol import parse_webvr_message
from quest_node.webvr_state import WebVRReceiverState, calibration_is_stale


def sample_payload(timestamp):
    pose = {
        "position": {"x": 0.0, "y": 1.0, "z": -0.5},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    return {
        "timestamp": timestamp,
        "Head": pose,
        "LeftHand": pose,
        "RightHand": pose,
        "Joy": {
            "axes": [0.0] * 8,
            "buttons": [[0, False]] * 6,
        },
    }


def test_fast_disconnect_reconnect_clears_old_sample_and_records_epoch():
    state = WebVRReceiverState()
    old_sample = parse_webvr_message(sample_payload(1))
    state.set_connection(True)
    state.record_sample(old_sample, 10.0)

    state.set_connection(False)
    state.set_connection(True)
    snapshot = state.snapshot()

    assert snapshot.connected
    assert snapshot.connection_version == 3
    assert snapshot.disconnect_epoch == 1
    assert snapshot.sample is None
    assert snapshot.received_at is None

    new_sample = parse_webvr_message(sample_payload(2))
    state.record_sample(new_sample, 10.1)
    snapshot = state.snapshot()
    assert snapshot.sequence == 2
    assert snapshot.sample is new_sample
    assert snapshot.received_at == 10.1


def test_process_if_current_rejects_snapshot_after_disconnect():
    state = WebVRReceiverState()
    sample = parse_webvr_message(sample_payload(1))
    state.set_connection(True)
    state.record_sample(sample, 10.0)
    snapshot = state.snapshot()
    state.set_connection(False)
    callbacks = []

    assert not state.process_if_current(snapshot, lambda: callbacks.append(True))
    assert callbacks == []


def test_receiver_counts_source_sequence_gaps_and_resets():
    state = WebVRReceiverState()
    state.set_connection(True)
    for sequence in (10, 11, 14, 2):
        payload = sample_payload(sequence)
        payload["sequence"] = sequence
        state.record_sample(parse_webvr_message(payload), float(sequence))

    snapshot = state.snapshot()

    assert snapshot.source_sequence_gaps == 2
    assert snapshot.source_sequence_resets == 1


def test_process_if_current_serializes_disconnect_after_callback():
    state = WebVRReceiverState()
    sample = parse_webvr_message(sample_payload(1))
    state.set_connection(True)
    state.record_sample(sample, 10.0)
    snapshot = state.snapshot()
    callback_started = threading.Event()
    disconnect_started = threading.Event()
    disconnect_finished = threading.Event()
    callback_observations = []
    process_results = []

    def callback():
        callback_started.set()
        disconnect_started.wait(timeout=1.0)
        callback_observations.append(disconnect_finished.is_set())

    def process_snapshot():
        process_results.append(state.process_if_current(snapshot, callback))

    def disconnect():
        disconnect_started.set()
        state.set_connection(False)
        disconnect_finished.set()

    process_thread = threading.Thread(target=process_snapshot)
    process_thread.start()
    assert callback_started.wait(timeout=1.0)
    disconnect_thread = threading.Thread(target=disconnect)
    disconnect_thread.start()
    process_thread.join(timeout=1.0)
    disconnect_thread.join(timeout=1.0)

    assert not process_thread.is_alive()
    assert not disconnect_thread.is_alive()
    assert process_results == [True]
    assert callback_observations == [False]
    assert disconnect_finished.is_set()
    assert not state.snapshot().connected


def test_calibration_only_expires_after_connected_tracking_stalls():
    assert not calibration_is_stale(
        calibrated=True,
        connected=True,
        last_frame_time=10.0,
        now=10.9,
        timeout=1.0,
    )
    assert calibration_is_stale(
        calibrated=True,
        connected=True,
        last_frame_time=10.0,
        now=11.1,
        timeout=1.0,
    )
    assert not calibration_is_stale(
        calibrated=False,
        connected=True,
        last_frame_time=10.0,
        now=20.0,
        timeout=1.0,
    )
    assert not calibration_is_stale(
        calibrated=True,
        connected=False,
        last_frame_time=10.0,
        now=20.0,
        timeout=1.0,
    )
