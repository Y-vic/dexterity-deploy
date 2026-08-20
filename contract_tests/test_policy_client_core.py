from copy import deepcopy

import numpy as np
import pytest

from sharpa_interface.server.action import parse_policy_action
from sharpa_interface.server.execution import build_execution_done
from sharpa_interface.server.policy import PolicyClientCore

from test_policy_input import metadata_format, sensor_frame, state_frame
from test_server_action import valid_action


class FakeServerClient:
    def __init__(self) -> None:
        self.format = metadata_format()
        self.observations: list[dict] = []
        self.closed = False

    def connect(self) -> dict:
        return {"prompt": "pick up the object", "metadata_format": self.format}

    def reset(self, session_id: str) -> dict:
        return {"metadata_format": deepcopy(self.format)}

    def infer_action(
        self,
        observation: dict,
        *,
        expected_session_id: str,
        expected_request_id: int,
    ):
        self.observations.append(observation)
        response = valid_action()
        response["session_id"] = expected_session_id
        response["request_id"] = expected_request_id
        action = parse_policy_action(
            response,
            expected_session_id=expected_session_id,
            expected_request_id=expected_request_id,
        )
        return action, 0.01

    def close(self) -> None:
        self.closed = True


def _fill_required_buffers(core: PolicyClientCore) -> None:
    core.push(
        "ego_cam",
        {"encoding": "jpeg", "data": b"jpeg", "timestamp_ns": 3, "valid": True},
    )
    core.push("state", state_frame(3))
    core.push("tau", sensor_frame(1, (22,), np.float32))
    core.push("tau", sensor_frame(2, (22,), np.float32))


def test_policy_client_core_fetches_only_after_matching_done() -> None:
    client = FakeServerClient()
    core = PolicyClientCore(client, session_id="episode")
    core.start()
    _fill_required_buffers(core)

    cycle = core.fetch(timestamp_ns=4)

    assert cycle.observation["prompt"] == "pick up the object"
    assert cycle.command["action"]["left_wrist"]["eef"].shape == (4, 9)
    assert not core.ready_to_fetch
    with pytest.raises(RuntimeError, match="phase=executing"):
        core.fetch(timestamp_ns=5)

    core.execution_done(
        build_execution_done(cycle.action, executed_in_slice=4, success=True)
    )
    assert core.ready_to_fetch


def test_policy_client_core_rejects_async() -> None:
    with pytest.raises(NotImplementedError, match="not supported"):
        PolicyClientCore(FakeServerClient(), session_id="episode", execution_mode="async")
