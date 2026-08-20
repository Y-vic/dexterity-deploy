from __future__ import annotations

from collections import deque
from collections.abc import Callable
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import ws_core.policy_client as policy_client_module
from ws_core.policy_client import ObsSample, PolicyClient, PolicyInputSnapshot
from ws_core.sharpa_v3 import (
    ACTION_SCHEMA,
    ERROR_SCHEMA,
    METADATA_FORMAT_SCHEMA,
    OBSERVATION_SCHEMA,
    SERVER_SCHEMA,
    SharpaV3Frame,
    SharpaV3History,
    SharpaV3ProtocolError,
)
from ws_core.websocket_rpc import PolicyRpcResult


def metadata_format(
    format_id: str,
    *,
    image_current: bool,
    state_current: bool,
) -> dict:
    return {
        "schema": METADATA_FORMAT_SCHEMA,
        "format_id": format_id,
        "image": {
            "ego_cam": {"history_len": 0, "current": image_current},
            "left_wrist_cam": {"history_len": 0, "current": False},
            "right_wrist_cam": {"history_len": 0, "current": False},
        },
        "state": {
            "history_len": 0,
            "current": state_current,
            "left_wrist": {"joint": False, "eef": True},
            "right_wrist": {"joint": False, "eef": True},
            "hand_joint": {"left": True, "right": True},
        },
        "sensor": {
            "tau": {"history_len": 0, "current": False},
            "wrench": {"history_len": 15, "current": True},
            "deformation": {"history_len": 0, "current": True},
        },
    }


def slow_format() -> dict:
    return metadata_format(
        "trex_slow_v1",
        image_current=True,
        state_current=True,
    )


def fast_format() -> dict:
    return metadata_format(
        "trex_fast_v1",
        image_current=False,
        state_current=False,
    )


def current_only_format() -> dict:
    result = metadata_format(
        "dreamzero_current_v1",
        image_current=True,
        state_current=True,
    )
    result["sensor"]["wrench"] = {"history_len": 0, "current": False}
    result["sensor"]["deformation"] = {
        "history_len": 0,
        "current": False,
    }
    return result


def server_metadata(*, prompt: str = "Fold the towel.") -> dict:
    return {
        "schema": SERVER_SCHEMA,
        "transport": "websocket+binary_msgpack",
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "infer_path": "/infer",
        "metadata_path": "/metadata",
        "reset_path": "/reset",
        "policy_family": "trex",
        "prompt": prompt,
        "metadata_format": slow_format(),
    }


def frame(index: int) -> SharpaV3Frame:
    value = float(index)
    return SharpaV3Frame(
        obs_seq=index,
        timestamp_ns=1_000_000 + index,
        image_jpeg=f"jpeg-{index}".encode(),
        image_valid=True,
        left_eef=np.full(9, value, dtype=np.float32),
        right_eef=np.full(9, value + 10, dtype=np.float32),
        left_hand=np.full(22, value + 20, dtype=np.float32),
        right_hand=np.full(22, value + 30, dtype=np.float32),
        state_valid=True,
        left_tau=np.zeros(22, dtype=np.float32),
        right_tau=np.zeros(22, dtype=np.float32),
        left_tau_valid=np.ones(22, dtype=bool),
        right_tau_valid=np.ones(22, dtype=bool),
        left_wrench=np.full((5, 6), value + 40, dtype=np.float32),
        right_wrench=np.full((5, 6), value + 50, dtype=np.float32),
        left_wrench_valid=np.ones(5, dtype=bool),
        right_wrench_valid=np.ones(5, dtype=bool),
        left_deformation=np.full((5, 240, 240), index, dtype=np.uint8),
        right_deformation=np.full((5, 240, 240), index + 10, dtype=np.uint8),
        left_deformation_valid=np.ones(5, dtype=bool),
        right_deformation_valid=np.ones(5, dtype=bool),
    )


@pytest.fixture(scope="module")
def history() -> tuple[SharpaV3Frame, ...]:
    return tuple(frame(index) for index in range(1, 19))


def latest_observation(index: int = 18) -> ObsSample:
    return ObsSample(
        seq=index,
        provider="obs_sync",
        payload_json=json.dumps({"schema": "ws.policy_obs.v1"}),
        image_rgb=b"",
        tactile_data=b"",
        recv_time=1.0,
        timestamp_unix_s=2.0,
        stamp_ns=1_000_000 + index,
    )


def snapshot(
    history: tuple[SharpaV3Frame, ...],
    *,
    active_format: dict | None = None,
) -> PolicyInputSnapshot:
    history_buffer = SharpaV3History()
    history_buffer.configure(active_format or slow_format())
    for item in history:
        history_buffer.append(item)
    history_snapshot = history_buffer.snapshot()
    assert history_snapshot is not None
    return PolicyInputSnapshot(
        window=(latest_observation(),),
        sharpa_v3=history_snapshot,
    )


def inference_request(
    request_id: int,
    *,
    feedback: dict | None = None,
    reason: str = "action_complete",
) -> SimpleNamespace:
    payload = {} if feedback is None else {"execution_feedback": feedback}
    return SimpleNamespace(
        request_id=request_id,
        request_stamp_ns=900_000 + request_id,
        trigger_action_seq=request_id - 1,
        reason=reason,
        payload_json=json.dumps(payload),
    )


def action_response(
    request_id: int,
    *,
    next_format: dict | None = None,
    revision: int = 0,
) -> dict:
    left = np.arange(16 * 9, dtype=np.float32).reshape(16, 9)
    right = left + 1_000
    hand = np.arange(16 * 44, dtype=np.float32).reshape(16, 44) + 2_000
    return {
        "schema": ACTION_SCHEMA,
        "session_id": "episode-v3",
        "request_id": request_id,
        "action_id": "trex-action-1",
        "revision": revision,
        "timestamp_ns": 2_000_000 + request_id,
        "execution": {
            "frequency_hz": 15.0,
            "action_length": 16,
            "execute_start": revision * 4,
            "execute_length": 4,
        },
        "action": {
            "left_wrist": {"joint": None, "eef": left, "eef_def": "absolute"},
            "right_wrist": {"joint": None, "eef": right, "eef_def": "absolute"},
            "hand_joint": {"left": hand[:, :22], "right": hand[:, 22:]},
        },
        "auxiliary": {
            "video": {"ego": None, "left_wrist": None, "right_wrist": None},
            "tactile": {"deformation": None, "wrench": None, "hand_tau": None},
        },
        "diagnostics": {
            "policy_family": "trex",
            "checkpoint_id": "trex-test",
            "checkpoint_path": "/checkpoints/trex-test",
            "inference_latency_ms": 5.0,
        },
        "next_metadata_format": next_format,
    }


def make_client(*, prompt: str = "") -> PolicyClient:
    client = object.__new__(PolicyClient)
    client.policy_protocol = "sharpa_v3"
    client.provider = "trex"
    client.server_url = "ws://127.0.0.1:5500/infer"
    client.request_timeout_s = 3.0
    client.ssh_host = ""
    client.ssh_remote_host = "127.0.0.1"
    client.ssh_remote_port = 5500
    client.session_id = "episode-v3"
    client.prompt = prompt
    client.client_lock = threading.Lock()
    client.lock = threading.Lock()
    client.client = None
    client.sharpa_v3_metadata = None
    client.sharpa_v3_active_format = None
    client.sharpa_v3_effective_prompt = ""
    client.sharpa_v3_reset_pending = False
    client.sharpa_v3_force_initial_feedback = True
    client.sharpa_v3_history = SharpaV3History()
    client.sharpa_v3_history_resets = 0
    client.sharpa_v3_history_last_reset_reason = ""
    client.last_sharpa_v3_obs_seq = None
    client.last_sharpa_v3_obs_time = None
    client.obs_buffer = deque(maxlen=128)
    client.latest_obs = None
    client.v3_history_max_gap_s = 0.25
    client.v3_image_max_age_ms = 150.0
    client.v3_joint_max_age_ms = 150.0
    client.v3_wrench_max_age_ms = 150.0
    client.v3_deformation_max_age_ms = 150.0
    client.obs_received = 0
    client.last_obs_seq = None
    client.last_obs_time = None
    client.last_request_id = 0
    client.last_pred_request_id = None
    client.last_pred_msg = None
    client.pending_request = None
    client.request_inflight = False
    client.last_trigger_action_seq = None
    client.request_event = threading.Event()
    client.stop_event = threading.Event()
    client._context = object()
    client.dry_run = False
    client.pred_seq = 0
    client.request_index = 0
    client.policy_calls = 0
    client.policy_failures = 0
    client.obs_dropped = 0
    client.last_latency_s = None
    client.last_error = ""
    client.pred_pub = SimpleNamespace(published=[])
    client.pred_pub.publish = client.pred_pub.published.append
    return client


def install_fake_transport(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict,
    *,
    infer_results: tuple[object, ...] = (),
    reset_format: dict | None = None,
    on_infer: Callable[[dict], None] | None = None,
):
    class FakeSharpaV3PolicyClient:
        instances: list[FakeSharpaV3PolicyClient] = []
        queued_results = deque(infer_results)

        def __init__(self, url: str, **kwargs) -> None:
            self.url = url
            self.kwargs = kwargs
            self.metadata = metadata
            self.closed = False
            self.infer_calls: list[dict] = []
            self.reset_calls: list[tuple[str, int | None]] = []
            type(self).instances.append(self)

        def infer(self, request: dict) -> PolicyRpcResult:
            self.infer_calls.append(request)
            if on_infer is not None:
                on_infer(request)
            result = type(self).queued_results.popleft()
            if isinstance(result, BaseException):
                raise result
            if isinstance(result, PolicyRpcResult):
                return result
            return PolicyRpcResult(payload=result, latency_s=0.01)

        def reset(self, session_id: str, request_id: int | None = None) -> dict:
            self.reset_calls.append((session_id, request_id))
            return {
                "schema": "sharpa_policy_reset.v1",
                "session_id": session_id,
                "request_id": request_id,
                "reset": True,
                "metadata_format": reset_format or slow_format(),
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        policy_client_module,
        "SharpaV3PolicyClient",
        FakeSharpaV3PolicyClient,
    )
    return FakeSharpaV3PolicyClient


def response_payload(
    client: PolicyClient,
    response: dict,
    request: SimpleNamespace,
) -> dict:
    return client._policy_response_payload(
        PolicyRpcResult(payload=response, latency_s=0.01),
        latest_observation(),
        pred_seq=request.request_id,
        stamp_ns=3_000_000 + request.request_id,
        request_info={"metadata_format_id": "trex_slow_v1"},
        request_started_ns=3_000_000,
        response_received_ns=3_010_000,
        request=request,
    )


def test_v3_metadata_builds_observation_and_converts_action(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    transport = install_fake_transport(monkeypatch, server_metadata())
    client = make_client()

    client._ensure_sharpa_v3_client()

    assert transport.instances[0].url.endswith("/infer")
    assert client.sharpa_v3_effective_prompt == "Fold the towel."
    assert client.sharpa_v3_active_format["format_id"] == "trex_slow_v1"
    request = inference_request(
        7,
        feedback={
            "last_action_id": "ignored-on-first-request",
            "executed_steps": 12,
            "success": True,
        },
    )
    observation, info = client._build_policy_request(
        snapshot(history),
        inference_request=request,
    )
    assert observation["metadata_format_id"] == "trex_slow_v1"
    assert observation["prompt"] == "Fold the towel."
    assert observation["sensor"]["wrench"]["history"]["left"].shape == (
        15,
        5,
        6,
    )
    assert observation["execution_feedback"] == {
        "last_action_id": None,
        "executed_steps": 0,
        "success": True,
    }
    assert info["provider"] == "trex"

    payload = response_payload(client, action_response(7), request)

    assert payload["action_hand_pose_62d"].shape == (4, 62)
    np.testing.assert_array_equal(
        payload["action_hand_pose_62d"][0, :9],
        action_response(7)["action"]["left_wrist"]["eef"][0],
    )
    assert payload["_ws_sharpa_v4"]["execute_length"] == 4
    assert not client.sharpa_v3_force_initial_feedback


def test_v3_client_is_not_created_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    client.stop_event.set()
    constructor_calls = []

    def unexpected_constructor(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        raise AssertionError("transport must not be created during shutdown")

    monkeypatch.setattr(
        policy_client_module,
        "SharpaV3PolicyClient",
        unexpected_constructor,
    )

    with pytest.raises(ConnectionError, match="shutting down"):
        client._ensure_sharpa_v3_client()

    assert constructor_calls == []


def test_v3_client_created_during_shutdown_is_closed_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()

    class ShutdownRaceTransport:
        instances = []

        def __init__(self, *args, **kwargs) -> None:
            self.metadata = server_metadata()
            self.closed = False
            type(self).instances.append(self)
            client.stop_event.set()

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        policy_client_module,
        "SharpaV3PolicyClient",
        ShutdownRaceTransport,
    )

    with pytest.raises(ConnectionError, match="shutting down"):
        client._ensure_sharpa_v3_client()

    assert len(ShutdownRaceTransport.instances) == 1
    assert ShutdownRaceTransport.instances[0].closed is True
    assert client.client is None


def test_destroy_closes_client_before_joining_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    order = []

    class FakeTransport:
        def close(self) -> None:
            order.append("close")

    class FakeWorker:
        def join(self) -> None:
            assert order == ["close"]
            order.append("join")

    client.client = FakeTransport()
    client.worker = FakeWorker()
    monkeypatch.setattr(
        policy_client_module.Node,
        "destroy_node",
        lambda self: order.append("destroy"),
    )

    client.destroy_node()

    assert client.stop_event.is_set()
    assert client.request_event.is_set()
    assert client.client is None
    assert order == ["close", "join", "destroy"]


def test_destroy_interrupts_blocked_v3_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    started = threading.Event()
    closed = threading.Event()
    errors = []

    class BlockingTransport:
        def infer(self, request: dict) -> None:
            started.set()
            assert closed.wait(timeout=1.0)
            raise ConnectionError("closed for shutdown")

        def close(self) -> None:
            closed.set()

    monkeypatch.setattr(
        policy_client_module,
        "SharpaV3PolicyClient",
        BlockingTransport,
    )
    monkeypatch.setattr(
        policy_client_module.Node,
        "destroy_node",
        lambda self: None,
    )
    client.client = BlockingTransport()

    def infer() -> None:
        try:
            client._infer_remote({"schema": "test"})
        except Exception as exc:
            errors.append(exc)

    client.worker = threading.Thread(target=infer)
    client.worker.start()
    assert started.wait(timeout=1.0)

    client.destroy_node()

    assert not client.worker.is_alive()
    assert closed.is_set()
    assert len(errors) == 1
    assert isinstance(errors[0], ConnectionError)


def test_v3_uses_fixed_stream_capacities_and_metadata_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_transport(monkeypatch, server_metadata())
    client = make_client()

    client._ensure_sharpa_v3_client()

    assert client.sharpa_v3_history.stream_capacities() == {
        "ego_cam": 3,
        "left_wrist_cam": 3,
        "right_wrist_cam": 3,
        "state": 19,
        "tau": 19,
        "wrench": 19,
        "deformation": 3,
    }
    assert client.sharpa_v3_history.stream_required_lengths() == {
        "ego_cam": 1,
        "left_wrist_cam": 0,
        "right_wrist_cam": 0,
        "state": 1,
        "tau": 0,
        "wrench": 16,
        "deformation": 1,
    }
    assert not client.sharpa_v3_history.ready


def test_v3_prebuffers_all_streams_before_metadata_arrives(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    install_fake_transport(monkeypatch, server_metadata())
    client = make_client()
    for item in history:
        client.sharpa_v3_history.append(item)
    client.latest_obs = latest_observation()
    client.obs_buffer.append(client.latest_obs)
    assert not client.sharpa_v3_history.configured

    client._ensure_sharpa_v3_client()

    assert client.sharpa_v3_history.ready
    selected = client._select_obs_window()
    assert selected is not None
    assert selected.sharpa_v3 is not None
    assert selected.sharpa_v3.anchor_obs_seq == 18
    assert len(selected.sharpa_v3.frames) == 16


def test_v3_current_only_format_is_ready_after_one_tick() -> None:
    client = make_client()
    client._set_sharpa_v3_format(current_only_format())
    client.sharpa_v3_history.append(frame(18))
    client.latest_obs = latest_observation()
    client.obs_buffer.append(client.latest_obs)

    selected = client._select_obs_window()

    assert selected is not None
    assert selected.sharpa_v3 is not None
    assert len(selected.sharpa_v3.frames) == 1
    assert selected.sharpa_v3.stream_capacities["ego_cam"] == 3
    assert selected.sharpa_v3.stream_capacities["wrench"] == 19
    assert selected.sharpa_v3.stream_required_lengths["ego_cam"] == 1
    assert selected.sharpa_v3.stream_required_lengths["wrench"] == 0


def test_v3_callback_keeps_only_latest_raw_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    client._set_sharpa_v3_format(current_only_format())
    client._image_from_obs = lambda _sample, _wrapper: (
        np.zeros((160, 320, 3), dtype=np.uint8),
        {"fallback": False},
    )
    client._encode_jpeg = lambda _image: b"jpeg"
    monkeypatch.setattr(
        policy_client_module,
        "extract_sharpa_v3_frame",
        lambda *_args, obs_seq, **_kwargs: frame(obs_seq),
    )

    for index in (1, 2):
        client._on_obs(
            SimpleNamespace(
                seq=index,
                provider="obs_sync",
                payload_json=json.dumps(
                    {
                        "stamp_ns": 1_000_000 + index,
                        "model_image": {"valid": True, "age_ms": 1.0},
                    }
                ),
                image_rgb=b"rgb",
                tactile_data=b"tactile",
            )
        )

    assert len(client.obs_buffer) == 1
    assert client.obs_buffer[-1].seq == 2
    assert client.sharpa_v3_history.stream_lengths() == {
        "ego_cam": 2,
        "left_wrist_cam": 2,
        "right_wrist_cam": 2,
        "state": 2,
        "tau": 2,
        "wrench": 2,
        "deformation": 2,
    }


def test_v3_slow_format_waits_for_only_sixteen_wrench_ticks() -> None:
    client = make_client()
    client._set_sharpa_v3_format(slow_format())
    for index in range(1, 16):
        client.sharpa_v3_history.append(frame(index))
    client.latest_obs = latest_observation(15)
    client.obs_buffer.append(client.latest_obs)
    assert client._select_obs_window() is None

    client.sharpa_v3_history.append(frame(16))
    client.latest_obs = latest_observation(16)
    client.obs_buffer.clear()
    client.obs_buffer.append(client.latest_obs)

    selected = client._select_obs_window()
    assert selected is not None
    assert selected.sharpa_v3 is not None
    assert selected.sharpa_v3.anchor_obs_seq == 16
    assert len(selected.sharpa_v3.frames) == 16


def test_v3_rejects_conflicting_fixed_prompt_and_closes_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = install_fake_transport(monkeypatch, server_metadata())
    client = make_client(prompt="Pick up the cup.")

    with pytest.raises(SharpaV3ProtocolError, match="prompt disagrees"):
        client._ensure_sharpa_v3_client()

    assert transport.instances[0].closed
    assert client.client is None
    assert client.sharpa_v3_metadata is None


def test_v3_rejects_wrong_policy_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = server_metadata()
    metadata["policy_family"] = "gcc_n17"
    transport = install_fake_transport(monkeypatch, metadata)
    client = make_client()

    with pytest.raises(SharpaV3ProtocolError, match="policy_family"):
        client._ensure_sharpa_v3_client()

    assert transport.instances[0].closed
    assert client.client is None


def test_v3_rejects_metadata_larger_than_fixed_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = server_metadata()
    metadata["metadata_format"]["sensor"]["wrench"]["history_len"] = 19
    transport = install_fake_transport(monkeypatch, metadata)
    client = make_client()

    with pytest.raises(
        SharpaV3ProtocolError,
        match="wrench requires 20 frames, fixed workstation capacity is 19",
    ):
        client._ensure_sharpa_v3_client()

    assert transport.instances[0].closed
    assert client.client is None


def test_trex_action_switches_slow_fast_and_back(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    install_fake_transport(monkeypatch, server_metadata())
    client = make_client()
    client._ensure_sharpa_v3_client()
    for item in history:
        client.sharpa_v3_history.append(item)
    client.latest_obs = latest_observation()
    client.obs_buffer.append(client.latest_obs)
    assert client._select_obs_window() is not None

    first_request = inference_request(8)
    response_payload(
        client,
        action_response(8, next_format=fast_format()),
        first_request,
    )
    assert client.sharpa_v3_active_format["format_id"] == "trex_fast_v1"

    feedback = {
        "last_action_id": "trex-action-1",
        "executed_steps": 4,
        "success": True,
    }
    second_request = inference_request(9, feedback=feedback)
    fast_snapshot = client._select_obs_window()
    assert fast_snapshot is not None
    fast_observation, _ = client._build_policy_request(
        fast_snapshot,
        inference_request=second_request,
    )
    assert fast_observation["metadata_format_id"] == "trex_fast_v1"
    assert fast_observation["image"]["ego_cam"]["current"] is None
    assert fast_observation["state"]["current"] is None
    assert fast_observation["execution_feedback"] == feedback

    response_payload(
        client,
        action_response(9, next_format=slow_format(), revision=1),
        second_request,
    )
    assert client.sharpa_v3_active_format["format_id"] == "trex_slow_v1"
    assert client.sharpa_v3_history.ready
    slow_snapshot = client._select_obs_window()
    assert slow_snapshot is not None
    assert slow_snapshot.sharpa_v3 is not None
    assert slow_snapshot.sharpa_v3.metadata_format["format_id"] == "trex_slow_v1"


def test_structured_server_error_does_not_publish_cached_action(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    error_response = {
        "schema": ERROR_SCHEMA,
        "request_id": 10,
        "error": {
            "code": "INVALID_POLICY_MESSAGE",
            "message": "bad metadata format",
            "retryable": False,
        },
    }
    install_fake_transport(
        monkeypatch,
        server_metadata(),
        infer_results=(error_response,),
    )
    monkeypatch.setattr(policy_client_module.rclpy, "ok", lambda **_: True)
    client = make_client()
    client._ensure_sharpa_v3_client()
    for item in history:
        client.sharpa_v3_history.append(item)
    client.latest_obs = latest_observation()
    client.obs_buffer.append(client.latest_obs)
    selected = client._select_obs_window()
    assert selected is not None
    cached_action = object()
    client.last_pred_msg = cached_action
    request = inference_request(10)

    success = client._run_policy_request(selected, request, 4_000_000)

    assert not success
    assert client.pred_pub.published == []
    assert client.last_pred_msg is cached_action
    assert client.policy_calls == 1
    assert client.policy_failures == 1
    assert "INVALID_POLICY_MESSAGE" in client.last_error


def test_stale_snapshot_is_not_sent_after_history_reset(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    transport = install_fake_transport(
        monkeypatch,
        server_metadata(),
        infer_results=(action_response(11),),
    )
    client = make_client()
    client._ensure_sharpa_v3_client()
    for item in history:
        client.sharpa_v3_history.append(item)
    client.latest_obs = latest_observation()
    client.obs_buffer.append(client.latest_obs)
    selected = client._select_obs_window()
    assert selected is not None

    with client.lock:
        client._clear_sharpa_v3_history_locked("pipeline_reset")

    assert not client._run_policy_request(
        selected,
        inference_request(11),
        4_100_000,
    )
    assert transport.instances[0].infer_calls == []
    assert client.pred_pub.published == []


def test_new_observation_after_snapshot_does_not_discard_request(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    transport = install_fake_transport(
        monkeypatch,
        server_metadata(),
        infer_results=(action_response(12),),
    )
    monkeypatch.setattr(policy_client_module.rclpy, "ok", lambda **_: True)
    monkeypatch.setattr(policy_client_module, "set_header", lambda *_: None)
    monkeypatch.setattr(PolicyClient, "get_clock", lambda _self: None)
    client = make_client()
    client.pred_published = 0
    client._ensure_sharpa_v3_client()
    for item in history:
        client.sharpa_v3_history.append(item)
    client.latest_obs = latest_observation()
    client.obs_buffer.append(client.latest_obs)
    selected = client._select_obs_window()
    assert selected is not None
    assert selected.sharpa_v3 is not None
    assert selected.sharpa_v3.anchor_obs_seq == 18

    client.sharpa_v3_history.append(frame(19))
    client.latest_obs = latest_observation(19)
    client.obs_buffer.clear()
    client.obs_buffer.append(client.latest_obs)

    assert client._run_policy_request(
        selected,
        inference_request(12),
        4_200_000,
    )
    assert len(transport.instances[0].infer_calls) == 1
    assert len(client.pred_pub.published) == 1
    assert client.last_pred_request_id == 12


def test_pipeline_reset_during_rpc_discards_response_and_next_format(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    client = make_client()

    def reset_pipeline(_request: dict) -> None:
        client._on_inference_request(
            inference_request(13, reason="pipeline_reset")
        )

    transport = install_fake_transport(
        monkeypatch,
        server_metadata(),
        infer_results=(action_response(12, next_format=fast_format()),),
        on_infer=reset_pipeline,
    )
    monkeypatch.setattr(policy_client_module.rclpy, "ok", lambda **_: True)
    client._ensure_sharpa_v3_client()
    for item in history:
        client.sharpa_v3_history.append(item)
    client.latest_obs = latest_observation()
    client.obs_buffer.append(client.latest_obs)
    selected = client._select_obs_window()
    assert selected is not None
    assert selected.sharpa_v3 is not None
    client.last_request_id = 12
    client.request_inflight = True

    assert not client._run_policy_request(
        selected,
        inference_request(12),
        4_300_000,
    )
    assert len(transport.instances[0].infer_calls) == 1
    assert client.pred_pub.published == []
    assert client.last_pred_request_id is None
    assert client.sharpa_v3_active_format["format_id"] == "trex_slow_v1"
    assert client.sharpa_v3_force_initial_feedback
    assert client.sharpa_v3_reset_pending
    assert client.pending_request.request_id == 13


def test_v3_status_reports_retained_capacity_required_per_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_transport(monkeypatch, server_metadata())
    client = make_client()
    client._ensure_sharpa_v3_client()
    for index in (1, 2):
        client.sharpa_v3_history.append(frame(index))
    client.action_horizon = 16
    client.actor_send_hz = 15.0
    client.obs_rate_hz = 30.0
    client.policy_window_frames = 4
    client.policy_window_stride = 2
    client.initial_window_frames = 1
    client.baseline_history_max_gap_s = 0.25
    client.baseline_image_max_age_ms = 150.0
    client.baseline_wrench_max_age_ms = 150.0
    client.baseline_deformation_max_age_ms = 150.0
    client.gcc_history_frames = 17
    client.gcc_history_max_gap_s = 0.25
    client.gcc_joint_max_age_ms = 150.0
    client.gcc_wrench_max_age_ms = 150.0
    client.gcc_deformation_max_age_ms = 150.0
    client.baseline_history = deque()
    client.baseline_history_resets = 0
    client.baseline_history_last_reset_reason = ""
    client.gcc_history = deque()
    client.gcc_history_resets = 0
    client.gcc_history_last_reset_reason = ""
    client.pred_published = 0
    client.last_request_start_time = None
    client.last_published_pred_seq = None
    client.last_request_started_ns = None
    client.last_response_received_ns = None
    client.last_pred_time = None
    client.last_request_info = {}
    client.status_pub = SimpleNamespace(published=[])
    client.status_pub.publish = client.status_pub.published.append
    monkeypatch.setattr(PolicyClient, "get_clock", lambda _self: None)
    monkeypatch.setattr(
        policy_client_module,
        "make_status",
        lambda _clock, _name, _ok, payload: payload,
    )

    client._publish_status()

    status = client.status_pub.published[0]["sharpa_v3"]
    assert status["history_retained"] == 2
    assert status["history_capacity"] == 19
    assert status["history_required"] == 16
    assert "history_len" not in status
    assert status["streams"] == {
        "ego_cam": {
            "length": 2,
            "capacity": 3,
            "required": 1,
            "ready": True,
        },
        "left_wrist_cam": {
            "length": 2,
            "capacity": 3,
            "required": 0,
            "ready": True,
        },
        "right_wrist_cam": {
            "length": 2,
            "capacity": 3,
            "required": 0,
            "ready": True,
        },
        "state": {
            "length": 2,
            "capacity": 19,
            "required": 1,
            "ready": True,
        },
        "tau": {
            "length": 2,
            "capacity": 19,
            "required": 0,
            "ready": True,
        },
        "wrench": {
            "length": 2,
            "capacity": 19,
            "required": 16,
            "ready": False,
        },
        "deformation": {
            "length": 2,
            "capacity": 3,
            "required": 1,
            "ready": True,
        },
    }


def test_reconnect_restores_initial_execution_feedback(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    transport = install_fake_transport(
        monkeypatch,
        server_metadata(),
        infer_results=(ConnectionError("connection lost"),),
    )
    client = make_client()
    client._ensure_sharpa_v3_client()
    first_transport = transport.instances[0]
    client.sharpa_v3_force_initial_feedback = False

    with pytest.raises(ConnectionError, match="connection lost"):
        client._infer_remote({"schema": OBSERVATION_SCHEMA})

    assert first_transport.closed
    assert client.client is None
    assert client.sharpa_v3_metadata is None
    assert client.sharpa_v3_active_format is None
    assert client.sharpa_v3_effective_prompt == ""
    assert client.sharpa_v3_force_initial_feedback

    client._ensure_sharpa_v3_client()
    request = inference_request(
        11,
        feedback={
            "last_action_id": "stale-action",
            "executed_steps": 8,
            "success": True,
        },
    )
    observation, _ = client._build_policy_request(
        snapshot(history),
        inference_request=request,
    )
    assert len(transport.instances) == 2
    assert observation["execution_feedback"] == {
        "last_action_id": None,
        "executed_steps": 0,
        "success": True,
    }


def test_pipeline_reset_restores_format_and_initial_feedback(
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SharpaV3Frame, ...],
) -> None:
    transport = install_fake_transport(
        monkeypatch,
        server_metadata(),
        reset_format=slow_format(),
    )
    client = make_client()
    client._ensure_sharpa_v3_client()
    client._set_sharpa_v3_format(fast_format())
    client.sharpa_v3_force_initial_feedback = False
    for item in history:
        client.sharpa_v3_history.append(item)
    client.last_sharpa_v3_obs_seq = history[-1].obs_seq
    client.last_sharpa_v3_obs_time = 1.0

    reset_request = inference_request(12, reason="pipeline_reset")
    client._on_inference_request(reset_request)

    assert not any(client.sharpa_v3_history.stream_lengths().values())
    assert client.sharpa_v3_history_last_reset_reason == "pipeline_reset"
    assert client.sharpa_v3_reset_pending
    client._ensure_sharpa_v3_client()
    assert transport.instances[0].reset_calls == [("episode-v3", 12)]
    assert client.sharpa_v3_active_format["format_id"] == "trex_slow_v1"
    assert not client.sharpa_v3_reset_pending

    request = inference_request(
        13,
        feedback={
            "last_action_id": "pre-reset-action",
            "executed_steps": 16,
            "success": True,
        },
    )
    observation, _ = client._build_policy_request(
        snapshot(history),
        inference_request=request,
    )
    assert observation["execution_feedback"] == {
        "last_action_id": None,
        "executed_steps": 0,
        "success": True,
    }
