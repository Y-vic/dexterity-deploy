"""Native selection, exact identity and retired interface proposals in real ROS wrappers."""

import json
from copy import deepcopy

import pytest
from policy_client_node import policy_client_node as policy
from std_msgs.msg import String
from test_policy_client_core import _metadata_format
from test_remote_callbacks import control, response
from test_remote_callbacks import run_node as run_node


class NativeClient:
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.closed = False
        self.current_metadata = {
            "model_id": "dfm_flow",
            "task_id": "task",
            "run_id": "run",
            "prompt": "Pick ducks.",
            "checkpoint_path": "/dfm/logs/run/checkpoints/checkpoint-45000",
            "policy_transport": "dfm_native",
            "native_interface": {"camera_bindings": {"ego": "ego_cam"}},
            "interface_revision": 1,
            "pending_interface_update": None,
            "metadata_format": _metadata_format("native-canonical"),
        }

    def connect(self):
        return deepcopy(self.current_metadata)

    def reset(self, session_id, request_id):
        self.current_metadata["pending_interface_update"] = None
        return {"metadata_format": self.current_metadata["metadata_format"]}

    def update_interface(self, interface):
        self.current_metadata["pending_interface_update"] = {"revision": 2, "interface": interface}
        return {"queued": True}

    def infer_action(self, payload, **kwargs):
        raise AssertionError("Tests supply the response explicitly")

    def close(self):
        self.closed = True


@pytest.fixture
def native_class(monkeypatch):
    import sharpa_interface.dfm.client

    monkeypatch.setattr(sharpa_interface.dfm.client, "NativePolicyClient", NativeClient)
    return NativeClient


def selected_status(**changes):
    return {
        "generation": 7,
        "phase": "running",
        "model": "dfm_flow",
        "task_id": "task",
        "run_id": "run",
        "prompt": "Pick ducks.",
        "checkpoint_path": "/dfm/logs/run/checkpoints/checkpoint-45000",
        "server_url": "ws://127.0.0.1:5500/infer",
        "ssh_host": "",
        "local": True,
        "policy_transport": "dfm_native",
        **changes,
    }


def connected(node):
    node._model_status(String(data=json.dumps(selected_status())))
    node._tick()
    node._inference_pool.finish()
    node._tick()
    node._control_status(control())
    node._tick()
    node._inference_pool.finish()
    node._tick()
    assert node._started and node._model_identity_ok and node._control_ready
    assert isinstance(node._client, NativeClient)


def proposal(node, **changes):
    return String(
        data=json.dumps(
            {
                "model_generation": node._model_generation,
                "session_id": node._core.session_id,
                "interface_revision": 1,
                "interface": {"declaration": "next"},
                **changes,
            }
        )
    )


def latest(node):
    return json.loads(node.messages["/policy/status"][-1].data)


def test_explicit_native_transport_selects_isolated_client_and_local_connection(run_node, native_class):
    def check(node):
        connected(node)
        assert node._client.config.ssh_host == ""
        assert node._client.kwargs["python"].endswith("dfm-serving/bin/python")
        assert node._policy_transport == latest(node)["policy_transport"] == "dfm_native"
        assert latest(node)["native_interface"] == node._client.current_metadata["native_interface"]
        assert latest(node)["interface_revision"] == 1

    run_node(check)


@pytest.mark.parametrize("change", [{"prompt": ""}, {"checkpoint_path": "wrong-checkpoint"}, {"checkpoint_path": ""}])
def test_native_ready_requires_exact_prompt_and_checkpoint_identity(run_node, native_class, change):
    def check(node):
        node._model_status(String(data=json.dumps(selected_status())))
        node._client.current_metadata.update(change)
        node._tick()
        node._inference_pool.finish()
        node._tick()
        assert not node._model_identity_ok and not node._model_available
        assert node._client.closed and node._phase == "fault"
        assert not node.messages["/policy/executable_action"]

    run_node(check)


@pytest.mark.parametrize(
    "changes", [{"model_generation": 6}, {"session_id": "retired-rollout"}, {"interface_revision": 99}]
)
def test_proposals_reject_retired_model_session_and_revision_without_rpc(run_node, native_class, changes):
    def check(node):
        connected(node)
        node._interface_update(proposal(node, **changes))
        assert not node._inference_pool.jobs and node._service_future is None
        assert latest(node)["interface_update_error"] and not node._rpc_faulted

    run_node(check)


def test_accepted_proposal_keeps_current_revision_and_rejected_proposal_keeps_episode_ready(run_node, native_class):
    def check(node):
        connected(node)
        session = node._core.session_id
        node._interface_update(proposal(node))
        node._inference_pool.finish()
        node._tick()
        assert latest(node)["interface_revision"] == 1
        assert latest(node)["pending_interface_update"]["revision"] == 2
        assert node._core.session_id == session and not node._rpc_faulted
        node._interface_update(proposal(node))
        node._inference_pool.finish(error=ValueError("conflicting pending proposal"))
        node._tick()
        assert "conflicting" in latest(node)["interface_update_error"]
        assert node._core.session_id == session and node._core.ready and not node._rpc_faulted

    run_node(check)


@pytest.mark.parametrize("retire", ["epoch", "session"])
def test_interface_acknowledgment_is_discarded_after_rollout_retirement(run_node, native_class, retire):
    def check(node):
        connected(node)
        before = deepcopy(node._metadata)
        node._interface_update(proposal(node))
        node._inference_pool.finish()
        if retire == "epoch":
            node._invalidate_request("pause")
        else:
            node._core = policy.PolicyClientCore("new-session")
        node._tick()
        assert node._metadata == before
        assert "retired" in latest(node)["interface_update_error"]
        assert node._rollout_reset_pending and not node._control_ready
        assert not node.messages["/policy/executable_action"]

    run_node(check)


def test_retired_interface_acknowledgment_requires_reset_before_resume_inference(run_node, native_class):
    def check(node):
        connected(node)
        node._observations.build = lambda *args, **kwargs: {"timestamp_ns": 1000}
        node._interface_update(proposal(node))
        node._control_status(control(paused=True))
        node._inference_pool.finish()
        node._tick()
        assert node._rollout_reset_pending and not node._control_ready
        node._control_status(control())
        node._tick()
        assert node._service_context[0] == "reset"
        node._inference_pool.finish()
        node._tick()
        assert node._metadata["pending_interface_update"] is None
        assert node._metadata["interface_revision"] == 1 and node._control_ready
        node._tick()
        assert node._inference_future is not None
        node._inference_pool.finish(response(node))
        node._tick()
        assert len(node.messages["/policy/executable_action"]) == 1
        assert node._metadata["interface_revision"] == 1

    run_node(check)


def test_native_cancelled_advanced_interface_is_refreshed_only_after_explicit_reset(run_node, native_class):
    def check(node):
        connected(node)
        node._observations.build = lambda *args, **kwargs: {"timestamp_ns": 1000}
        node._tick()
        result = response(node)
        node._client.current_metadata["interface_revision"] = 2
        node._control_status(control(paused=True))
        node._inference_pool.finish(result)
        node._tick()
        assert not node.messages["/policy/executable_action"]
        assert node._metadata["interface_revision"] == 1
        node._control_status(control())
        node._tick()
        node._inference_pool.finish()
        node._tick()
        assert node._metadata["interface_revision"] == 2 and node._control_ready
        assert node._request_metrics.inference_count == 0

    run_node(check)
