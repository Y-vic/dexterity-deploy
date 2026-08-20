from io import BytesIO
from pathlib import Path

from monitor_node.recording_monitor import RecordingMonitor, RecordingSession


def _session() -> RecordingSession:
    return RecordingSession(
        index=1,
        sample_name="sample_0001",
        partial_dir=Path("sample_0001.partial"),
        final_dir=Path("sample_0001"),
        started_unix_ns=1,
        started_mono=1.0,
        start_status={},
        raw_handle=BytesIO(),
    )


def _event(status: dict) -> dict:
    return {"valid": True, "status": status}


def test_quest_quality_labels_tracking_gate_and_transport_failures():
    session = _session()
    session.quest_webvr_events = [
        _event(
            {
                "connected": True,
                "hand_position_tracking": {
                    "LeftHand": "Known",
                    "RightHand": "Known",
                },
                "hand_gates": {
                    "LeftHand": {"state": "Normal"},
                    "RightHand": {"state": "Normal"},
                },
                "receive_age_ms": 10.0,
                "source_sequence_gaps": 4,
            }
        ),
        _event(
            {
                "connected": True,
                "hand_position_tracking": {
                    "LeftHand": "Lost",
                    "RightHand": "Known",
                },
                "hand_gates": {
                    "LeftHand": {"state": "Recovering"},
                    "RightHand": {"state": "Normal"},
                },
                "receive_age_ms": 250.0,
                "source_sequence_gaps": 6,
            }
        ),
    ]
    errors = []
    warnings = []

    RecordingMonitor._append_quest_quality_errors(session, errors, warnings)

    assert {error["code"] for error in errors} == {
        "quest_openxr_position_lost",
        "quest_hand_gate_held",
        "quest_transport_frame_gap",
    }


def test_quest_quality_warns_on_large_retarget_residual():
    session = _session()
    session.quest_webvr_events = [_event({"connected": True})]
    session.quest_retarget_events = [
        _event(
            {
                "wrist_position_residual_mm": {
                    "Left": 12.0,
                    "Right": 120.0,
                }
            }
        )
    ]
    errors = []
    warnings = []

    RecordingMonitor._append_quest_quality_errors(session, errors, warnings)

    assert not errors
    assert [warning["code"] for warning in warnings] == [
        "quest_retarget_residual_high"
    ]


def test_non_quest_recording_is_not_failed_by_missing_quest_status():
    session = _session()
    errors = []
    warnings = []

    RecordingMonitor._append_quest_quality_errors(session, errors, warnings)

    assert not errors
    assert not warnings
