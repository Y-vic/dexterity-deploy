"""Validate this server's payloads with the WORKSTATION's own validators.

The point of these tests is that they do not re-implement the contract: they
import `sharpa_interface` from the deploy workspace and hand it exactly what the
server would send. A hand-written assertion about a field name can agree with a
mistake; `validate_server_metadata` cannot.

These run on the workstation, offline, with no model and no robot.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "interface"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import server as srv
from sharpa_interface.server.action import parse_policy_action
from sharpa_interface.server.execution import ExecutionValidationError
from sharpa_interface.server.metadata import (
    validate_metadata_format,
    validate_reset_response,
    validate_server_metadata,
)


FORMAT_ID = "mot_action_v5.test"


def make_metadata(prompt: str = "") -> dict:
    return srv.build_server_metadata(
        format_id=FORMAT_ID,
        model_id="sharpa_policy_server",
        checkpoint_id="030000",
        checkpoint_path="/share/project/zjk/orca/outputs/x/checkpoints/030000",
        task_id="ur_orca_action",
        run_id="mot_v5_head100h_030000",
        prompt=prompt,
        dataset_path="",
        host="127.0.0.1",
        port=5500,
    )


def test_metadata_passes_the_workstation_validator():
    """The whole /metadata body, through the client's own strict check."""
    validated = validate_server_metadata(make_metadata())
    assert validated["schema"] == srv.SERVER_SCHEMA
    assert validated["action_schema"] == srv.ACTION_SCHEMA
    assert validated["port"] == 5500


def test_metadata_format_passes_and_requests_only_what_is_used():
    validated = validate_metadata_format(srv.build_metadata_format(FORMAT_ID))
    # One ego frame, no history: the model reads a single current image.
    assert validated["image"]["ego_cam"] == {"history_len": 0, "current": True}
    for name in ("left_wrist_cam", "right_wrist_cam"):
        assert validated["image"][name]["current"] is False
        assert validated["image"][name]["history_len"] == 0
    # STATE_ANCHOR_OFFSET frames of history plus current, eef only, both hands.
    # history_len must NOT be 0: the four state rows and their anchor are read
    # out of this history so their spacing is the training rate. Requesting 0
    # and buffering per-request on the server stretched the rows ~5x.
    assert validated["state"]["current"] is True
    assert validated["state"]["history_len"] == srv.STATE_ANCHOR_OFFSET == 12
    for side in ("left_wrist", "right_wrist"):
        assert validated["state"][side]["eef"] is True
        assert validated["state"][side]["joint"] is False
    assert validated["state"]["hand_joint"] == {"left": True, "right": True}
    # No tactile at all -- it is not in the 56D representation.
    for name in ("tau", "wrench", "deformation"):
        assert validated["sensor"][name] == {"history_len": 0, "current": False}


def test_eef_contract_matches_what_policy_client_demands():
    """Copied from policy_client_node.validate_server_eef_contract.

    If these five values drift, the node refuses to connect. Asserting them
    here means the failure shows up in a test rather than as a dashboard fault.
    """
    expected = {
        "schema": "sharpa62_direct_absolute_eef.v1",
        "observation_eef_def": "absolute",
        "action_eef_def": "absolute",
        "eef_pose_layout": "xyz,r00,r10,r20,r01,r11,r21",
        "server_xyz_direction_transform": "none",
    }
    assert make_metadata()["interface"]["eef_contract"] == expected


def test_reset_response_passes_the_validator():
    policy_server = srv.PolicyServer(
        srv.MotPolicy("", "", ""),
        metadata=make_metadata(),
        format_id=FORMAT_ID,
        prompt="",
        dry_run=True,
    )
    response = policy_server.handle_reset({"session_id": "abc123"})
    validated = validate_reset_response(response, expected_session_id="abc123")
    assert validated["reset"] is True
    assert validated["metadata_format"]["format_id"] == FORMAT_ID


def test_reset_generates_a_session_id_when_none_is_given():
    policy_server = srv.PolicyServer(
        srv.MotPolicy("", "", ""),
        metadata=make_metadata(),
        format_id=FORMAT_ID,
        prompt="",
        dry_run=True,
    )
    response = policy_server.handle_reset({})
    assert response["session_id"]
    validate_reset_response(response, expected_session_id=response["session_id"])


# --- a full cycle ---------------------------------------------------------
def identity_pose9(x: float = 0.3, y: float = 0.1, z: float = 0.5) -> np.ndarray:
    """An absolute pose with identity rotation, in canonical column layout."""
    return np.array([x, y, z, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def make_state_history(frames: int = 12, step_m: float = 0.002) -> dict:
    """A state history stacked oldest-first, as the workstation's buffers emit it.

    ``step_m`` advances x by 2 mm per frame, so a 12-frame span is 24 mm -- the
    order of magnitude training measured over the same span. A test that used
    identical frames would pass even if the server read the wrong index.
    """
    xs = [0.3 - step_m * (frames - i) for i in range(frames)]
    return {
        "timestamp_ns": np.arange(frames, dtype=np.int64),
        "left_wrist": {
            "joint": None,
            "eef": np.stack([identity_pose9(x, 0.1, 0.5) for x in xs]),
            "eef_def": "absolute",
        },
        "right_wrist": {
            "joint": None,
            "eef": np.stack([identity_pose9(-x, 0.1, 0.5) for x in xs]),
            "eef_def": "absolute",
        },
        "hand_joint": {
            "left": np.full((frames, 22), 0.1, dtype=np.float32),
            "right": np.full((frames, 22), 0.2, dtype=np.float32),
        },
        "valid": np.ones(frames, dtype=np.bool_),
    }


def make_observation(session_id: str, request_id: int = 0) -> dict:
    """One observation shaped exactly as the workstation's buffers emit it."""
    return {
        "schema": srv.OBSERVATION_SCHEMA,
        "metadata_format_id": FORMAT_ID,
        "session_id": session_id,
        "request_id": request_id,
        "timestamp_ns": 1_700_000_000_000_000_000,
        "prompt": "",
        "image": {
            "ego_cam": {"history": [], "current": {"data": b"not-a-real-jpeg"}},
            "left_wrist_cam": {"history": [], "current": None},
            "right_wrist_cam": {"history": [], "current": None},
        },
        "state": {
            # STATE_ANCHOR_OFFSET frames, oldest first, excluding current --
            # exactly what select(history_len=12, current=True) produces. The
            # server reads rows out of this at STATE_HISTORY_OFFSETS.
            "history": make_state_history(),
            "current": {
                "timestamp_ns": 1_700_000_000_000_000_000,
                "valid": True,
                "left_wrist": {
                    "joint": None,
                    "eef": identity_pose9(0.3, 0.1, 0.5),
                    "eef_def": "absolute",
                },
                "right_wrist": {
                    "joint": None,
                    "eef": identity_pose9(-0.3, 0.1, 0.5),
                    "eef_def": "absolute",
                },
                "hand_joint": {
                    "left": np.full(22, 0.1, dtype=np.float32),
                    "right": np.full(22, 0.2, dtype=np.float32),
                },
            },
        },
        "sensor": {
            name: {"history": None, "current": None}
            for name in ("tau", "wrench", "deformation")
        },
        "execution_feedback": {
            "last_action_id": None,
            "executed_steps": 0,
            "success": True,
        },
    }


def dry_run_server() -> srv.PolicyServer:
    policy_server = srv.PolicyServer(
        srv.MotPolicy("", "", ""),
        metadata=make_metadata(),
        format_id=FORMAT_ID,
        prompt="",
        dry_run=True,
    )
    policy_server.ready = True
    return policy_server


def test_action_response_passes_parse_policy_action():
    """The response the client actually consumes, through its own parser.

    This is the test that would catch a missing `rtc`, a wrong array width, a
    pre-sliced array, or a relative eef_def leaking onto the boundary.
    """
    policy_server = dry_run_server()
    session_id = policy_server.handle_reset({})["session_id"]
    response = policy_server.handle_infer(make_observation(session_id, request_id=7))

    parsed = parse_policy_action(
        response, expected_session_id=session_id, expected_request_id=7
    )
    assert parsed.execution.action_length == srv.EXECUTION_HORIZON
    # NOT the whole chunk: the first EXECUTE_START steps are expired intent by
    # the time the chunk takes over. See the RTC_STEPS derivation in server.py.
    assert parsed.execution.execute_length == srv.EXECUTE_LENGTH
    assert parsed.execution.execute_start == srv.EXECUTE_START
    assert parsed.execution.rtc == srv.RTC_STEPS
    # The arrays themselves stay full length; the client does the slicing.
    assert parsed.left_wrist.eef.shape == (srv.EXECUTION_HORIZON, 9)
    assert parsed.execution.frequency_hz == pytest.approx(10.0)


def test_dry_run_chunk_holds_the_current_pose():
    """A zero relative chunk must decode to exactly where the arm already is.

    This is the safety property of --dry-run: the transport can be exercised
    against real hardware in preview without commanding motion.
    """
    policy_server = dry_run_server()
    session_id = policy_server.handle_reset({})["session_id"]
    observation = make_observation(session_id)
    response = policy_server.handle_infer(observation)

    current = observation["state"]["current"]
    for side in ("left", "right"):
        commanded = response["action"][f"{side}_wrist"]["eef"]
        measured = current[f"{side}_wrist"]["eef"]
        assert commanded.shape == (srv.EXECUTION_HORIZON, 9)
        for step in commanded:
            assert np.allclose(step, measured, atol=1e-5)


def test_dry_run_holds_the_measured_hand_pose():
    policy_server = dry_run_server()
    session_id = policy_server.handle_reset({})["session_id"]
    observation = make_observation(session_id)
    response = policy_server.handle_infer(observation)

    for side in ("left", "right"):
        commanded = response["action"]["hand_joint"][side]
        measured = observation["state"]["current"]["hand_joint"][side]
        assert commanded.shape == (srv.EXECUTION_HORIZON, 22)
        for step in commanded:
            assert np.allclose(step, measured, atol=0)


def test_infer_rejects_a_foreign_session():
    """A session mismatch must raise here, not be answered.

    The client drops an identity mismatch silently, so answering a foreign
    session would present as a hang with no error anywhere.
    """
    policy_server = dry_run_server()
    policy_server.handle_reset({"session_id": "real"})
    with pytest.raises(ValueError, match="does not match"):
        policy_server.handle_infer(make_observation("forged"))


def test_infer_requires_a_reset_first():
    policy_server = dry_run_server()
    with pytest.raises(ValueError, match="no active session"):
        policy_server.handle_infer(make_observation("whatever"))


def test_response_survives_a_msgpack_roundtrip():
    """What the client decodes must equal what the server built.

    numpy arrays cross this boundary as a descriptor, not a list; a mismatch
    here would surface downstream as a type error naming the wrong cause.
    """
    policy_server = dry_run_server()
    session_id = policy_server.handle_reset({})["session_id"]
    response = policy_server.handle_infer(make_observation(session_id, request_id=3))

    decoded = srv.unpackb(srv.packb(response))
    parsed = parse_policy_action(
        decoded, expected_session_id=session_id, expected_request_id=3
    )
    assert parsed.execution.rtc == srv.RTC_STEPS
    for side in ("left", "right"):
        original = response["action"][f"{side}_wrist"]["eef"]
        assert np.allclose(decoded["action"][f"{side}_wrist"]["eef"], original, atol=0)
        assert decoded["action"][f"{side}_wrist"]["eef"].dtype == original.dtype


# --- the four-row state tensor --------------------------------------------
def test_state_tensor_has_the_row_count_the_expert_demands():
    """Four rows, not one.

    The expert raises if the count disagrees with its state_history_offsets,
    and this was a real defect: the dataclass DEFAULT is (0,), but this run's
    hydra config sets (9, 6, 3, 0). A one-row tensor reached the real model and
    was rejected there -- offline nothing had caught it.
    """
    import repr_bridge as rb

    assert rb.STATE_HISTORY_OFFSETS == (9, 6, 3, 0)
    assert rb.STATE_ANCHOR_OFFSET == 12
    assert rb.STATE_ANCHOR_OFFSET > max(rb.STATE_HISTORY_OFFSETS)




# --- state rows read from the observation's history -----------------------
def test_state_rows_come_from_the_history_at_the_right_offsets():
    """The rows must be the frames at STATE_HISTORY_OFFSETS, not any 4 frames.

    The fixture advances x by 2 mm per frame, so each offset has a distinct
    value and an off-by-one or reversed index shows up as a wrong number rather
    than as a plausible-looking one.
    """
    import repr_bridge as rb

    obs = make_observation("s", 0)
    anchor, rows = srv.extract_state_rows(obs)
    assert len(rows) == 4

    history = obs["state"]["history"]["left_wrist"]["eef"]
    current = obs["state"]["current"]["left_wrist"]["eef"]
    # history is oldest-first with 12 entries; offset k back is index 12-k,
    # and offset 0 is `current`.
    for row_index, offset in enumerate(rb.STATE_HISTORY_OFFSETS):
        expected = current if offset == 0 else history[rb.STATE_ANCHOR_OFFSET - offset]
        assert np.allclose(rows[row_index][0]["left"], expected, atol=0), (
            f"row {row_index} (offset {offset}) read the wrong frame"
        )
    # The anchor is the oldest frame in the history: offset 12 -> index 0.
    assert np.allclose(anchor["left"], history[0], atol=0)


def test_state_row_span_matches_the_training_magnitude():
    """The whole point of reading history: displacement must be ~cm, not ~dm.

    The fixture moves 2 mm/frame, so the newest row sits 12 frames = 24 mm from
    the anchor. Under the old server-side ring the same four rows spanned one
    inference each and this number came out several times larger.
    """
    import repr_bridge as rb

    anchor, rows = srv.extract_state_rows(make_observation("s", 0))
    state = rb.build_state_rows(anchor, rows)
    newest_span = np.linalg.norm(state[-1][rb.SLICES["left_wrist_trans"]])
    assert 0.020 < newest_span < 0.030, f"span {newest_span*1000:.1f} mm"
    # Rows must be strictly ordered oldest-to-newest in displacement.
    spans = [
        float(np.linalg.norm(row[rb.SLICES["left_wrist_trans"]])) for row in state
    ]
    assert spans == sorted(spans), f"rows not oldest-first: {spans}"


def test_missing_history_is_rejected_not_guessed():
    """A None history must name the cause, not silently fall back."""
    obs = make_observation("s", 0)
    obs["state"]["history"] = None
    with pytest.raises(ValueError, match="history_len"):
        srv.extract_state_rows(obs)


def test_short_history_is_rejected():
    obs = make_observation("s", 0)
    obs["state"]["history"]["left_wrist"]["eef"] = np.zeros((5, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="need at least"):
        srv.extract_state_rows(obs)


def test_relative_history_is_rejected_at_the_boundary():
    obs = make_observation("s", 0)
    obs["state"]["history"]["left_wrist"]["eef_def"] = "relative"
    with pytest.raises(ValueError, match="must be absolute"):
        srv.extract_state_rows(obs)


def test_action_anchor_is_current_not_the_state_anchor():
    """The two anchors differ, and mixing them is the likeliest silent error.

    A dry-run chunk is zero relative motion, so it must decode to the CURRENT
    pose. If the action were composed onto the state anchor instead, it would
    decode 24 mm behind where the arm actually is -- a small, plausible offset
    that would quietly bias every command backwards.
    """
    policy_server = dry_run_server()
    session_id = policy_server.handle_reset({})["session_id"]
    obs = make_observation(session_id, 0)
    response = policy_server.handle_infer(obs)

    current = obs["state"]["current"]["left_wrist"]["eef"]
    state_anchor = obs["state"]["history"]["left_wrist"]["eef"][0]
    commanded = response["action"]["left_wrist"]["eef"][0]

    assert np.allclose(commanded, current, atol=1e-5)
    assert not np.allclose(commanded, state_anchor, atol=1e-4)


def test_metadata_satisfies_the_policy_client_identity_check():
    """The dashboard verifies identity separately from validate_server_metadata.

    ``metadata_identity_error`` reads ``model_id``/``model_name``, ``task_id``,
    ``run_id`` and ``prompt``, and treats a MISSING field as a mismatch -- so a
    metadata body that passes the schema validator can still be refused. It was:
    the server reported only ``policy_family`` and the dashboard showed
    "模型身份不匹配：model <missing> != sharpa_policy_server" with the model
    running and idle. Checked here with the node's own function.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ur-sharpa/src/nodes/policy_client_node"))
    from policy_client_node.policy_client_node import metadata_identity_error

    error = metadata_identity_error(
        make_metadata(),
        expected_model="sharpa_policy_server",
        expected_task_id="ur_orca_action",
        expected_run_id="mot_v5_head100h_030000",
        expected_prompt="",
    )
    assert error == "", error


# --- the anchor-lag timing --------------------------------------------------
# These model TIME, which no other test here does. The defect they pin produced
# perfectly valid chunks -- correct shape, correct anchor, green /healthz, 0.87 ms
# inter-chunk gap -- while the arm made 13-22% of the ground-truth progress per
# cycle, because the first steps of every chunk commanded a return to a pose the
# arm had already left.
def test_execute_start_equals_the_lag_between_anchoring_and_take_over():
    """The dropped prefix must be exactly the steps that expire before take-over.

    RTC asks the client to fetch with RTC_STEPS of the current slice still to
    run, so a chunk begins executing RTC_STEPS periods after its observation was
    taken. Dropping fewer steps leaves a retraction; dropping more discards real
    intent. Equality is the whole property.
    """
    assert srv.EXECUTE_START == srv.RTC_STEPS


def test_the_first_executed_step_is_the_one_that_means_now():
    """Step k of a chunk is (k+1)*ACTION_STRIDE frames ahead of the observation.

    At take-over the arm is RTC_STEPS periods past that observation, so the step
    whose horizon matches "now" is the one this slice starts on. Derived from the
    representation's own constants rather than restated, so a stride or fps change
    cannot leave this passing while the alignment breaks.
    """
    import repr_bridge as rb

    period_s = 1.0 / srv.FREQUENCY_HZ
    lag_s = srv.RTC_STEPS * period_s
    # Horizon of the first step this slice executes, in seconds.
    first = srv.EXECUTE_START
    horizon_s = (first + 1) * rb.ACTION_STRIDE / rb.TRAIN_FPS
    # It should be the first step whose horizon reaches the lag.
    assert horizon_s > lag_s
    previous_s = first * rb.ACTION_STRIDE / rb.TRAIN_FPS
    assert previous_s <= lag_s, (
        f"step {first - 1} at {previous_s * 1000:.0f} ms already reaches the "
        f"{lag_s * 1000:.0f} ms lag; the slice starts one step too late"
    )


def test_diagnostic_mode_is_strictly_sequential():
    """No step is dropped and no inference is prefetched while executing."""
    assert srv.RTC_STEPS == 0
    assert srv.EXECUTE_START == 0
    assert srv.EXECUTE_LENGTH == srv.EXECUTION_HORIZON


def test_the_slice_fits_inside_the_chunk_the_server_sends():
    assert srv.EXECUTE_START + srv.EXECUTE_LENGTH <= srv.EXECUTION_HORIZON


def test_client_executes_the_slice_the_server_asked_for():
    """Through the client's own slicing, not a re-implementation of it.

    ``executable_action`` is what the ROS node publishes, so this is the number of
    steps the arm will actually run and the pose it will start from.
    """
    policy_server = dry_run_server()
    session_id = policy_server.handle_reset({})["session_id"]
    obs = make_observation(session_id, request_id=11)
    response = policy_server.handle_infer(obs)
    parsed = parse_policy_action(
        response, expected_session_id=session_id, expected_request_id=11
    )

    executable = parsed.executable_action()
    for side in ("left", "right"):
        assert executable[f"{side}_wrist"]["eef"].shape == (srv.EXECUTE_LENGTH, 9)
        assert executable["hand_joint"][side].shape == (srv.EXECUTE_LENGTH, 22)
    # And it must be the TAIL of the full array, not the head.
    full = response["action"]["left_wrist"]["eef"]
    assert np.allclose(
        executable["left_wrist"]["eef"], full[srv.EXECUTE_START :], atol=0
    )


def test_sequential_gate_waits_for_done_before_the_next_inference():
    """The next request cannot begin until the full active chunk reports done."""
    from sharpa_interface.server.execution import SyncExecutionGate

    policy_server = dry_run_server()
    session_id = policy_server.handle_reset({})["session_id"]
    gate = SyncExecutionGate()

    def chunk(request_id: int):
        response = policy_server.handle_infer(
            make_observation(session_id, request_id)
        )
        return parse_policy_action(
            response,
            expected_session_id=session_id,
            expected_request_id=request_id,
        )

    request_id, _ = gate.begin_inference()
    command = gate.accept_action(chunk(request_id))
    execution = command["execution"]

    with pytest.raises(RuntimeError, match="cannot begin inference"):
        gate.begin_inference()
    with pytest.raises(ExecutionValidationError, match="rtc > 0"):
        gate.begin_prefetch(
            request_id=request_id,
            action_id=command["action_id"],
            revision=command["revision"],
            execute_start=execution["execute_start"],
            execute_length=execution["execute_length"],
            executed_steps=execution["execute_start"],
        )

    feedback = gate.complete(
        {
            "schema": "sharpa_execution_done.v1",
            "session_id": session_id,
            "request_id": request_id,
            "action_id": command["action_id"],
            "revision": command["revision"],
            "execute_start": execution["execute_start"],
            "execute_length": execution["execute_length"],
            "executed_steps": execution["execute_start"]
            + execution["execute_length"],
            "success": True,
            "done": True,
            "error": None,
        }
    )
    assert feedback["success"]
    next_request_id, _ = gate.begin_inference()
    assert next_request_id == request_id + 1
