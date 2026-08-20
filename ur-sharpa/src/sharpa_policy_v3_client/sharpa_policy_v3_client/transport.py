"""aiohttp transport for the SharpA policy server v3 protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import math
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .metadata import (
    MetadataValidationError,
    validate_reset_response,
    validate_server_metadata,
)
from .serialization import (
    MAX_MESSAGE_SIZE,
    MessageTooLargeError,
    SerializationError,
    packb,
    unpackb,
)


MSGPACK_CONTENT_TYPE = "application/msgpack"
ERROR_SCHEMA = "sharpa_policy_error.v1"


class PolicyTransportError(RuntimeError):
    """Base class for v3 transport failures."""


class PolicyDependencyError(PolicyTransportError):
    """A runtime-only transport dependency is unavailable."""


class PolicyClosedError(PolicyTransportError):
    """An operation was attempted after the transport was closed."""


class PolicyStateError(PolicyTransportError):
    """An operation is invalid in the current transport state."""


class PolicyConcurrencyError(PolicyTransportError):
    """A second operation would violate the single-inflight contract."""


class PolicyProtocolError(PolicyTransportError):
    """The server returned data that violates the v3 wire contract."""


class PolicyMessageTooLargeError(PolicyProtocolError):
    """A wire message exceeds the negotiated size limit."""


class PolicyHttpError(PolicyTransportError):
    """An HTTP endpoint returned a non-success response."""

    def __init__(self, status: int, url: str, detail: str = "") -> None:
        self.status = int(status)
        self.url = url
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"HTTP {self.status} from {url}{suffix}")


class PolicyWebSocketError(PolicyTransportError):
    """The inference WebSocket failed or closed unexpectedly."""


class PolicyTimeoutError(PolicyWebSocketError):
    """Inference did not produce one response within the timeout."""


class PolicyServerError(PolicyTransportError):
    """A validated ``sharpa_policy_error.v1`` response."""

    def __init__(
        self,
        *,
        request_id: int | None,
        code: str,
        message: str,
        retryable: bool,
        payload: dict[str, Any],
    ) -> None:
        self.request_id = request_id
        self.code = code
        self.message = message
        self.retryable = retryable
        self.payload = payload
        super().__init__(f"policy server error {code}: {message}")


@dataclass(frozen=True)
class PolicyUrls:
    base_url: str
    health_url: str
    metadata_url: str
    reset_url: str
    infer_url: str


def derive_policy_urls(server_url: str) -> PolicyUrls:
    """Derive all fixed v3 endpoints from an HTTP or WebSocket URL."""

    if not isinstance(server_url, str) or not server_url.strip():
        raise ValueError("server_url must be a non-empty string")
    parsed = urlsplit(server_url.strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError("server_url scheme must be http, https, ws, or wss")
    if not parsed.hostname:
        raise ValueError("server_url must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("server_url must not contain credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("server_url has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("server_url port must be in [1, 65535]")
    if parsed.query or parsed.fragment:
        raise ValueError("server_url must not contain a query or fragment")
    if parsed.path not in {"", "/", "/healthz", "/metadata", "/reset", "/infer"}:
        raise ValueError("server_url path must be a fixed v3 endpoint or root")

    secure = parsed.scheme in {"https", "wss"}
    http_scheme = "https" if secure else "http"
    ws_scheme = "wss" if secure else "ws"
    base_url = urlunsplit((http_scheme, parsed.netloc, "", "", ""))
    infer_url = urlunsplit((ws_scheme, parsed.netloc, "/infer", "", ""))
    return PolicyUrls(
        base_url=base_url,
        health_url=f"{base_url}/healthz",
        metadata_url=f"{base_url}/metadata",
        reset_url=f"{base_url}/reset",
        infer_url=infer_url,
    )


def _load_aiohttp() -> Any:
    try:
        return importlib.import_module("aiohttp")
    except ImportError as exc:
        raise PolicyDependencyError(
            "aiohttp is required for SharpA policy v3 transport"
        ) from exc


def _positive_timeout(value: float, name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return timeout


class PolicyV3Transport:
    """One-client, one-inflight async transport for the v3 server contract."""

    def __init__(
        self,
        server_url: str,
        *,
        http_timeout_s: float = 10.0,
        connect_timeout_s: float = 10.0,
        inference_timeout_s: float = 90.0,
        max_message_size: int = MAX_MESSAGE_SIZE,
        heartbeat_s: float | None = 20.0,
        session: Any | None = None,
    ) -> None:
        if isinstance(max_message_size, bool) or not isinstance(max_message_size, int):
            raise TypeError("max_message_size must be an integer")
        if not 1 <= max_message_size <= MAX_MESSAGE_SIZE:
            raise ValueError(
                f"max_message_size must be in [1, {MAX_MESSAGE_SIZE}]"
            )
        if heartbeat_s is not None:
            heartbeat_s = _positive_timeout(heartbeat_s, "heartbeat_s")
        self.urls = derive_policy_urls(server_url)
        self.http_timeout_s = _positive_timeout(http_timeout_s, "http_timeout_s")
        self.connect_timeout_s = _positive_timeout(
            connect_timeout_s,
            "connect_timeout_s",
        )
        self.inference_timeout_s = _positive_timeout(
            inference_timeout_s,
            "inference_timeout_s",
        )
        self.max_message_size = max_message_size
        self.heartbeat_s = heartbeat_s
        self.server_metadata: dict[str, Any] | None = None
        self.active_metadata_format: dict[str, Any] | None = None
        self._effective_message_size = max_message_size
        self._session = session
        self._owns_session = session is None
        self._ws: Any | None = None
        self._aiohttp: Any | None = None
        self._inflight = False
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._ws is not None and not bool(getattr(self._ws, "closed", True))

    @property
    def inflight(self) -> bool:
        return self._inflight

    @property
    def effective_message_size(self) -> int:
        return self._effective_message_size

    def _assert_open(self) -> None:
        if self._closed:
            raise PolicyClosedError("policy transport is closed")

    def _assert_idle(self) -> None:
        if self._inflight:
            raise PolicyConcurrencyError("an inference request is already inflight")

    async def _ensure_session(self) -> Any:
        self._assert_open()
        if self._session is not None:
            if bool(getattr(self._session, "closed", False)):
                raise PolicyClosedError("aiohttp session is closed")
            return self._session
        aiohttp = self._aiohttp or _load_aiohttp()
        self._aiohttp = aiohttp
        timeout = aiohttp.ClientTimeout(
            total=self.http_timeout_s,
            connect=self.connect_timeout_s,
        )
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _read_limited(self, response: Any, limit: int) -> bytes:
        content_length = response.content_length
        if content_length is not None and content_length > limit:
            raise PolicyMessageTooLargeError(
                f"HTTP response is {content_length} bytes; limit is {limit}"
            )
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            body.extend(chunk)
            if len(body) > limit:
                raise PolicyMessageTooLargeError(
                    f"HTTP response exceeds the {limit}-byte limit"
                )
        return bytes(body)

    async def _http_msgpack(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        limit: int,
    ) -> dict[str, Any]:
        session = await self._ensure_session()
        headers = {"Accept": MSGPACK_CONTENT_TYPE}
        body: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = MSGPACK_CONTENT_TYPE
            try:
                body = packb(dict(payload), max_size=limit)
            except MessageTooLargeError as exc:
                raise PolicyMessageTooLargeError(str(exc)) from exc
            except SerializationError as exc:
                raise PolicyProtocolError(f"invalid HTTP request payload: {exc}") from exc
        try:
            async with session.request(
                method,
                url,
                data=body,
                headers=headers,
            ) as response:
                response_body = await self._read_limited(response, limit)
                if response.status != 200:
                    detail = response_body[:1024].decode("utf-8", errors="replace").strip()
                    raise PolicyHttpError(response.status, url, detail)
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type != MSGPACK_CONTENT_TYPE:
                    raise PolicyProtocolError(
                        f"{url} returned Content-Type {content_type!r}, "
                        f"expected {MSGPACK_CONTENT_TYPE}"
                    )
        except PolicyTransportError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise PolicyHttpError(0, url, str(exc)) from exc
        try:
            decoded = unpackb(response_body, max_size=limit)
        except MessageTooLargeError as exc:
            raise PolicyMessageTooLargeError(str(exc)) from exc
        except SerializationError as exc:
            raise PolicyProtocolError(f"invalid msgpack response from {url}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise PolicyProtocolError(f"{url} response must be a msgpack object")
        return decoded

    async def metadata(self) -> dict[str, Any]:
        """Fetch and validate ``GET /metadata`` before connecting inference."""

        self._assert_open()
        self._assert_idle()
        raw = await self._http_msgpack(
            "GET",
            self.urls.metadata_url,
            limit=self.max_message_size,
        )
        try:
            metadata = validate_server_metadata(raw)
        except MetadataValidationError as exc:
            raise PolicyProtocolError(f"invalid server metadata: {exc}") from exc
        effective_limit = min(
            self.max_message_size,
            metadata["max_message_size"],
        )
        if self.connected and effective_limit != self._effective_message_size:
            raise PolicyStateError(
                "cannot renegotiate max_message_size while WebSocket is connected"
            )
        self.server_metadata = metadata
        self.active_metadata_format = metadata["metadata_format"]
        self._effective_message_size = effective_limit
        return metadata

    async def health(self) -> bool:
        """Require a successful ``GET /healthz`` response."""

        self._assert_open()
        self._assert_idle()
        session = await self._ensure_session()
        try:
            async with session.get(self.urls.health_url) as response:
                body = await self._read_limited(response, 64 * 1024)
                if response.status != 200:
                    detail = body[:1024].decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                    raise PolicyHttpError(
                        response.status,
                        self.urls.health_url,
                        detail,
                    )
        except PolicyTransportError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise PolicyHttpError(0, self.urls.health_url, str(exc)) from exc
        return True

    async def get_metadata(self) -> dict[str, Any]:
        """Compatibility alias for :meth:`metadata`."""

        return await self.metadata()

    async def reset(
        self,
        session_id: str,
        request_id: int = 0,
    ) -> dict[str, Any]:
        """Reset one episode through ``POST /reset`` and adopt its format."""

        self._assert_open()
        self._assert_idle()
        if self.server_metadata is None:
            raise PolicyStateError("metadata must be fetched before reset")
        if self.connected:
            raise PolicyStateError("disconnect inference before resetting a session")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
            raise ValueError("request_id must be a non-negative integer")
        raw = await self._http_msgpack(
            "POST",
            self.urls.reset_url,
            payload={"session_id": session_id, "request_id": request_id},
            limit=self._effective_message_size,
        )
        try:
            result = validate_reset_response(
                raw,
                expected_session_id=session_id,
                expected_request_id=request_id,
            )
        except MetadataValidationError as exc:
            raise PolicyProtocolError(f"invalid reset response: {exc}") from exc
        self.active_metadata_format = result["metadata_format"]
        return result

    async def connect(self) -> None:
        """Open the persistent ``/infer`` WebSocket after metadata handshake."""

        self._assert_open()
        self._assert_idle()
        if self.server_metadata is None:
            raise PolicyStateError("metadata must be fetched before WebSocket connect")
        if self.connected:
            return
        session = await self._ensure_session()
        aiohttp = self._aiohttp or _load_aiohttp()
        self._aiohttp = aiohttp
        try:
            self._ws = await session.ws_connect(
                self.urls.infer_url,
                autoping=True,
                heartbeat=self.heartbeat_s,
                compress=0,
                max_msg_size=self._effective_message_size,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._ws = None
            raise PolicyWebSocketError(
                f"failed to connect {self.urls.infer_url}: {exc}"
            ) from exc

    async def connect_infer(self) -> None:
        """Compatibility alias for :meth:`connect`."""

        await self.connect()

    async def _drop_websocket(self) -> None:
        websocket = self._ws
        self._ws = None
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                pass

    async def disconnect_infer(self) -> None:
        """Close only the inference socket so metadata/reset can be reused."""

        self._assert_open()
        self._assert_idle()
        await self._drop_websocket()

    @staticmethod
    def _server_error(payload: dict[str, Any]) -> PolicyServerError:
        if set(payload) != {"schema", "request_id", "error"}:
            raise PolicyProtocolError("policy error response has invalid fields")
        request_id = payload["request_id"]
        if request_id is not None and (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or request_id < 0
        ):
            raise PolicyProtocolError("policy error request_id must be non-negative or null")
        error = payload["error"]
        if not isinstance(error, dict) or set(error) != {"code", "message", "retryable"}:
            raise PolicyProtocolError("policy error body has invalid fields")
        code = error["code"]
        message = error["message"]
        retryable = error["retryable"]
        if not isinstance(code, str) or not code:
            raise PolicyProtocolError("policy error code must be a non-empty string")
        if not isinstance(message, str):
            raise PolicyProtocolError("policy error message must be a string")
        if type(retryable) is not bool:
            raise PolicyProtocolError("policy error retryable must be a boolean")
        return PolicyServerError(
            request_id=request_id,
            code=code,
            message=message,
            retryable=retryable,
            payload=payload,
        )

    async def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Send one binary observation and await exactly one binary response."""

        self._assert_open()
        if self._inflight:
            raise PolicyConcurrencyError("an inference request is already inflight")
        if not self.connected:
            raise PolicyStateError("inference WebSocket is not connected")
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be an object")
        try:
            encoded = packb(
                dict(observation),
                max_size=self._effective_message_size,
            )
        except MessageTooLargeError as exc:
            raise PolicyMessageTooLargeError(str(exc)) from exc
        except SerializationError as exc:
            raise PolicyProtocolError(f"invalid inference request: {exc}") from exc

        self._inflight = True
        websocket = self._ws
        aiohttp = self._aiohttp or _load_aiohttp()
        self._aiohttp = aiohttp
        try:
            try:
                await websocket.send_bytes(encoded)
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=self.inference_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                await self._drop_websocket()
                raise PolicyTimeoutError(
                    f"inference timed out after {self.inference_timeout_s:g}s"
                ) from exc
            except asyncio.CancelledError:
                await self._drop_websocket()
                raise
            except Exception as exc:
                await self._drop_websocket()
                raise PolicyWebSocketError(f"inference WebSocket failed: {exc}") from exc

            if message.type != aiohttp.WSMsgType.BINARY:
                detail = str(message.data) if message.data is not None else message.type.name
                await self._drop_websocket()
                raise PolicyProtocolError(
                    "inference response must be one binary msgpack message; "
                    f"received {message.type.name}: {detail[:200]}"
                )
            raw = message.data
            if not isinstance(raw, bytes):
                await self._drop_websocket()
                raise PolicyProtocolError("binary WebSocket response is not bytes")
            try:
                payload = unpackb(raw, max_size=self._effective_message_size)
            except MessageTooLargeError as exc:
                await self._drop_websocket()
                raise PolicyMessageTooLargeError(str(exc)) from exc
            except SerializationError as exc:
                await self._drop_websocket()
                raise PolicyProtocolError(f"invalid inference msgpack: {exc}") from exc
            if not isinstance(payload, dict):
                await self._drop_websocket()
                raise PolicyProtocolError("inference response must be a msgpack object")
            if payload.get("schema") == ERROR_SCHEMA:
                server_error = self._server_error(payload)
                expected_request_id = observation.get("request_id")
                if (
                    server_error.request_id is not None
                    and server_error.request_id != expected_request_id
                ):
                    await self._drop_websocket()
                    raise PolicyProtocolError(
                        "policy error request_id does not match the observation"
                    )
                raise server_error
            return payload
        finally:
            self._inflight = False

    async def close(self) -> None:
        """Close the WebSocket and any internally owned aiohttp session."""

        if self._closed:
            return
        self._closed = True
        await self._drop_websocket()
        session = self._session
        self._session = None
        if self._owns_session and session is not None:
            try:
                await session.close()
            except Exception:
                pass

    async def __aenter__(self) -> "PolicyV3Transport":
        self._assert_open()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()


PolicyTransport = PolicyV3Transport
