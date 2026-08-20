from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any
import uuid

from .action import ParsedPolicyActionV3, parse_policy_action
from .metadata import validate_metadata_format
from .observation import ObservationBuilder, validate_format_capacity
from .transport import PolicyV3Transport


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    request_id: int
    format_id: str | None
    connected: bool
    last_action_id: str | None


class PolicySessionRuntime:
    def __init__(
        self,
        transport: PolicyV3Transport,
        observation_builder: ObservationBuilder | None = None,
        *,
        session_id: str | None = None,
        prompt: str = "",
        auto_reset: bool = True,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not isinstance(transport, PolicyV3Transport):
            raise TypeError("transport must be a PolicyV3Transport")
        if observation_builder is not None and not isinstance(
            observation_builder,
            ObservationBuilder,
        ):
            raise TypeError(
                "observation_builder must be an ObservationBuilder or None"
            )
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise ValueError("session_id must be a non-empty string or None")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if type(auto_reset) is not bool:
            raise TypeError("auto_reset must be a boolean")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self.transport = transport
        self.observation_builder = observation_builder
        self.session_id = session_id or self.new_session_id()
        self.prompt = prompt
        self.auto_reset = auto_reset
        self.clock_ns = clock_ns
        self.server_metadata: dict[str, Any] | None = None
        self.request_id = 0
        self.last_action: ParsedPolicyActionV3 | None = None
        self._started = False
        self._lifecycle_lock = asyncio.Lock()

    @staticmethod
    def new_session_id() -> str:
        return f"sharpa-{uuid.uuid4()}"

    @property
    def active_metadata_format(self) -> dict[str, Any] | None:
        return self.transport.active_metadata_format

    @property
    def started(self) -> bool:
        return self._started and self.transport.connected

    def snapshot(self) -> SessionSnapshot:
        active_format = self.active_metadata_format
        return SessionSnapshot(
            session_id=self.session_id,
            request_id=self.request_id,
            format_id=(
                active_format["format_id"] if active_format is not None else None
            ),
            connected=self.transport.connected,
            last_action_id=(
                self.last_action.action_id if self.last_action is not None else None
            ),
        )

    def _adopt_active_format(self) -> dict[str, Any]:
        active_format = self.transport.active_metadata_format
        if active_format is None:
            raise RuntimeError("active metadata format is unavailable")
        normalized = self._validate_active_format(active_format)
        self.transport.active_metadata_format = normalized
        return normalized

    def _validate_active_format(self, metadata_format: object) -> dict[str, Any]:
        if self.observation_builder is None:
            return validate_metadata_format(metadata_format)
        return validate_format_capacity(
            self.observation_builder.buffers,
            metadata_format,
            max_message_size=self.transport.effective_message_size,
        )

    def _new_session(self, requested: str | None) -> str:
        if requested is not None and requested == self.session_id:
            raise ValueError("new session_id must differ from the active session")
        return requested or self.new_session_id()

    async def start(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("policy session is already started")
            await self.transport.health()
            metadata = await self.transport.metadata()
            self.server_metadata = metadata
            if self.auto_reset:
                await self.transport.reset(self.session_id, 0)
            self._adopt_active_format()
            await self.transport.connect()
            self.request_id = 0
            self.last_action = None
            self._started = True
            return metadata

    async def reset(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise ValueError("session_id must be a non-empty string or None")
        async with self._lifecycle_lock:
            new_session_id = self._new_session(session_id)
            if self.transport.connected:
                await self.transport.disconnect_infer()
            if self.server_metadata is None:
                await self.transport.health()
                self.server_metadata = await self.transport.metadata()
            self.session_id = new_session_id
            result = await self.transport.reset(self.session_id, 0)
            self._adopt_active_format()
            if self.observation_builder is not None:
                self.observation_builder.buffers.clear()
            self.request_id = 0
            self.last_action = None
            await self.transport.connect()
            self._started = True
            return result

    async def reconnect(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise ValueError("session_id must be a non-empty string or None")
        async with self._lifecycle_lock:
            new_session_id = self._new_session(session_id)
            if self.transport.connected:
                await self.transport.disconnect_infer()
            await self.transport.health()
            self.server_metadata = await self.transport.metadata()
            self.session_id = new_session_id
            result = await self.transport.reset(self.session_id, 0)
            self._adopt_active_format()
            if self.observation_builder is not None:
                self.observation_builder.buffers.clear()
            self.request_id = 0
            self.last_action = None
            await self.transport.connect()
            self._started = True
            return result

    async def infer_once(
        self,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> ParsedPolicyActionV3:
        async with self._lifecycle_lock:
            if not self.started:
                raise RuntimeError("policy session is not started")
            if self.observation_builder is None:
                raise RuntimeError(
                    "infer_once requires a local observation builder; "
                    "use infer_observation for a state-node observation"
                )
            active_format = self.active_metadata_format
            if active_format is None:
                raise RuntimeError("active metadata format is unavailable")
            wire_request_id = self.request_id
            observation = self.observation_builder.build(
                active_format,
                session_id=self.session_id,
                request_id=wire_request_id,
                timestamp_ns=self.clock_ns(),
                prompt=self.prompt,
                execution_feedback=execution_feedback,
                max_message_size=self.transport.effective_message_size,
            )
            return await self._infer_observation_locked(
                observation,
                wire_request_id=wire_request_id,
            )

    async def infer_observation(
        self,
        observation: Mapping[str, Any],
    ) -> ParsedPolicyActionV3:
        async with self._lifecycle_lock:
            if not self.started:
                raise RuntimeError("policy session is not started")
            return await self._infer_observation_locked(
                observation,
                wire_request_id=self.request_id,
            )

    async def _infer_observation_locked(
        self,
        observation: Mapping[str, Any],
        *,
        wire_request_id: int,
    ) -> ParsedPolicyActionV3:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        active_format = self.active_metadata_format
        if active_format is None:
            raise RuntimeError("active metadata format is unavailable")
        expected = {
            "schema": "sharpa_policy_observation.v3",
            "session_id": self.session_id,
            "request_id": wire_request_id,
            "metadata_format_id": active_format["format_id"],
        }
        for field, wanted in expected.items():
            if observation.get(field) != wanted:
                raise ValueError(
                    f"observation {field} does not match active policy session"
                )
        raw_action = await self.transport.infer(observation)
        action = parse_policy_action(
            raw_action,
            expected_session_id=self.session_id,
            expected_request_id=wire_request_id,
        )
        if action.next_metadata_format is not None:
            self.transport.active_metadata_format = self._validate_active_format(
                action.next_metadata_format
            )
        self.last_action = action
        self.request_id += 1
        return action

    async def disconnect(self) -> None:
        async with self._lifecycle_lock:
            if self.transport.connected:
                await self.transport.disconnect_infer()
            self._started = False

    async def close(self) -> None:
        async with self._lifecycle_lock:
            self._started = False
            await self.transport.close()
