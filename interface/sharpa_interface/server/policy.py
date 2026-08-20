"""ROS-independent core of the single synchronous ``policy_client`` node."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .action import ParsedPolicyAction
from .buffers import PolicyInputBuffers
from .client import ServerClient
from .execution import SyncExecutionGate, SyncPhase


@dataclass(frozen=True)
class PolicyCycle:
    action: ParsedPolicyAction
    command: dict[str, Any]
    observation: dict[str, Any]
    latency_s: float


class PolicyClientCore:
    """Own metadata, buffers and the fetch/execute/done state machine."""

    def __init__(
        self,
        client: ServerClient,
        *,
        session_id: str,
        execution_mode: str = "synchronous",
        buffer_capacities: Mapping[str, int] | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must be nonempty")
        self.client = client
        self.session_id = session_id
        self.buffers = PolicyInputBuffers(buffer_capacities)
        self.gate = SyncExecutionGate(execution_mode)
        self.server_metadata: dict[str, Any] | None = None
        self.metadata_format: dict[str, Any] | None = None
        self.prompt = ""

    @property
    def ready_to_fetch(self) -> bool:
        return self.metadata_format is not None and self.gate.phase is SyncPhase.READY

    def start(self) -> dict[str, Any]:
        """Fetch server metadata, reset the session and activate its format."""

        metadata = self.client.connect()
        reset = self.client.reset(self.session_id)
        self.buffers.clear()
        self.gate.reset()
        self.server_metadata = metadata
        self.metadata_format = reset["metadata_format"]
        self.prompt = metadata["prompt"]
        return metadata

    def push(self, name: str, frame: Mapping[str, Any]) -> None:
        self.buffers.push(name, frame)

    def fetch(self, *, timestamp_ns: int, prompt: str | None = None) -> PolicyCycle:
        """Fetch one action and produce the sliced command for ``action_ik``."""

        if self.metadata_format is None:
            raise RuntimeError("policy client is not started")
        request_id, feedback = self.gate.begin_inference()
        try:
            observation = self.buffers.build_observation(
                self.metadata_format,
                session_id=self.session_id,
                request_id=request_id,
                timestamp_ns=timestamp_ns,
                prompt=self.prompt if prompt is None else prompt,
                execution_feedback=feedback,
            )
            action, latency_s = self.client.infer_action(
                observation,
                expected_session_id=self.session_id,
                expected_request_id=request_id,
            )
            command = self.gate.accept_action(action)
        except Exception:
            if self.gate.phase is SyncPhase.INFERENCE:
                self.gate.cancel_inference()
            raise

        if action.next_metadata_format is not None:
            self.metadata_format = action.next_metadata_format
        return PolicyCycle(action, command, observation, latency_s)

    def execution_done(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Accept ``action_execute`` done; success unlocks the next fetch."""

        return self.gate.complete(payload)

    def close(self) -> None:
        self.gate.close()
        self.client.close()
