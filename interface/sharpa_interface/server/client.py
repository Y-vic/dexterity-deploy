"""Thread-safe lifecycle wrapper around the SharpA v3 transport."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping

from .action import ParsedPolicyAction, parse_policy_action
from .metadata import validate_reset_response, validate_server_metadata
from .serialization import MAX_MESSAGE_SIZE
from .transport import PolicyRpcResult, SharpaV3PolicyClient


@dataclass(frozen=True)
class ServerConfig:
    url: str = "ws://127.0.0.1:5500/infer"
    timeout_s: float = 60.0
    ssh_host: str = ""
    ssh_remote_host: str = ""
    ssh_remote_port: int = 0
    max_http_body_size: int = MAX_MESSAGE_SIZE
    max_message_size: int = MAX_MESSAGE_SIZE
    configured_prompt: str = ""
    expected_policy_family: str | None = None


class ServerClient:
    """Own one reconnectable transport and make shutdown race-safe."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._transport: SharpaV3PolicyClient | None = None
        self._closed = False

    def connect(self) -> dict[str, Any]:
        transport = self._require_transport()
        try:
            metadata = validate_server_metadata(transport.metadata)
            expected_family = self._config.expected_policy_family
            if expected_family is not None and metadata["policy_family"].lower() != expected_family.strip().lower():
                raise ValueError(
                    "server policy_family does not match configured provider: "
                    f"{metadata['policy_family']!r} != {expected_family!r}"
                )
            configured_prompt = self._config.configured_prompt
            server_prompt = metadata["prompt"]
            if configured_prompt and server_prompt and configured_prompt != server_prompt:
                raise ValueError("configured prompt disagrees with server task prompt")
            return metadata
        except Exception:
            with self._lock:
                if self._transport is transport:
                    self._transport = None
            transport.close()
            raise

    def metadata(self) -> dict[str, Any]:
        return self.connect()

    def reset(
        self,
        session_id: str,
        request_id: int | None = None,
    ) -> dict[str, Any]:
        self.connect()
        response = self._require_transport().reset(session_id, request_id=request_id)
        return validate_reset_response(
            response,
            expected_session_id=session_id,
            expected_request_id=request_id,
        )

    def infer(self, payload: Mapping[str, Any]) -> PolicyRpcResult:
        self.connect()
        transport = self._require_transport()
        try:
            return transport.infer(dict(payload))
        except Exception:
            with self._lock:
                if self._transport is transport:
                    self._transport = None
            transport.close()
            raise

    def infer_action(
        self,
        payload: Mapping[str, Any],
        *,
        expected_session_id: str,
        expected_request_id: int,
    ) -> tuple[ParsedPolicyAction, float]:
        result = self.infer(payload)
        action = parse_policy_action(
            result.payload,
            expected_session_id=expected_session_id,
            expected_request_id=expected_request_id,
        )
        return action, result.latency_s

    def close(self) -> None:
        with self._lock:
            self._closed = True
            transport = self._transport
            self._transport = None
        if transport is not None:
            transport.close()

    def _require_transport(self) -> SharpaV3PolicyClient:
        with self._lock:
            if self._closed:
                raise ConnectionError("policy client is shutting down")
            transport = self._transport
        if transport is not None:
            return transport

        candidate = SharpaV3PolicyClient(
            self._config.url,
            timeout_s=self._config.timeout_s,
            ssh_host=self._config.ssh_host,
            ssh_remote_host=self._config.ssh_remote_host,
            ssh_remote_port=self._config.ssh_remote_port,
            max_http_body_size=self._config.max_http_body_size,
            max_message_size=self._config.max_message_size,
        )
        with self._lock:
            if not self._closed and self._transport is None:
                self._transport = candidate
                return candidate
            transport = self._transport
            closed = self._closed
        candidate.close()
        if closed:
            raise ConnectionError("policy client is shutting down")
        if transport is None:
            raise ConnectionError("policy client connection was closed")
        return transport
