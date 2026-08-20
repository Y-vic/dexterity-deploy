"""Authentication and same-origin checks for the Quest WebVR socket."""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit


MIN_ACCESS_TOKEN_LENGTH = 24
MAX_ACCESS_TOKEN_LENGTH = 256


class WebVRSecurityError(ValueError):
    """Raised when a WebVR authentication or request check fails."""


def generate_access_token() -> str:
    return secrets.token_urlsafe(24)


def validate_access_token(token: str) -> str:
    normalized = str(token).strip()
    if not MIN_ACCESS_TOKEN_LENGTH <= len(normalized) <= MAX_ACCESS_TOKEN_LENGTH:
        raise WebVRSecurityError(
            "Quest access token must contain between "
            f"{MIN_ACCESS_TOKEN_LENGTH} and {MAX_ACCESS_TOKEN_LENGTH} characters"
        )
    return normalized


def normalize_public_web_url(url: str) -> str:
    normalized = str(url).strip()
    try:
        parsed = urlsplit(normalized)
    except ValueError as exc:
        raise WebVRSecurityError("Quest public WebVR URL is invalid") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise WebVRSecurityError("Quest public WebVR URL must be an HTTPS URL")
    return normalized.rstrip("/") + "/"


def _header_values(headers: Any, name: str) -> list[str]:
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        return [str(value) for value in get_all(name)]
    if isinstance(headers, Mapping):
        value = headers.get(name)
        if value is None:
            value = headers.get(name.lower())
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value]
        return [str(value)]
    return []


def _single_header(headers: Any, name: str) -> str:
    values = _header_values(headers, name)
    if len(values) != 1 or not values[0].strip():
        raise WebVRSecurityError(
            f"WebSocket request requires exactly one {name} header"
        )
    return values[0].strip()


def _https_authority(value: str, *, is_origin: bool) -> tuple[str, int]:
    candidate = value if is_origin else f"//{value}"
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise WebVRSecurityError(
            "WebSocket request has an invalid HTTPS authority"
        ) from exc

    if is_origin:
        if parsed.scheme.lower() != "https":
            raise WebVRSecurityError("WebSocket Origin must use HTTPS")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise WebVRSecurityError("WebSocket Origin must not contain a path")
    elif parsed.scheme or parsed.path or parsed.query or parsed.fragment:
        raise WebVRSecurityError("WebSocket Host header is invalid")

    if (
        parsed.username is not None
        or parsed.password is not None
        or not parsed.hostname
    ):
        raise WebVRSecurityError("WebSocket request has an invalid HTTPS authority")
    return parsed.hostname.lower().rstrip("."), port or 443


def validate_secure_same_origin(headers: Any) -> None:
    """Require an HTTPS request proxied from the same public Host and Origin."""

    host = _single_header(headers, "Host")
    origin = _single_header(headers, "Origin")
    forwarded_proto = _single_header(headers, "X-Forwarded-Proto")
    if forwarded_proto.lower() != "https":
        raise WebVRSecurityError("WebSocket proxy must report HTTPS")
    if _https_authority(host, is_origin=False) != _https_authority(
        origin,
        is_origin=True,
    ):
        raise WebVRSecurityError("WebSocket Origin does not match Host")


def authenticate_first_message(message: object, expected_token: str) -> None:
    """Validate the first client message using a constant-time token comparison."""

    if not isinstance(message, (str, bytes)):
        raise WebVRSecurityError("WebSocket authentication message must be JSON")
    if len(message) > 1024:
        raise WebVRSecurityError("WebSocket authentication message is too large")
    try:
        payload = json.loads(message)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise WebVRSecurityError(
            "WebSocket authentication message must be JSON"
        ) from exc

    payload_type = payload.get("type", "") if isinstance(payload, dict) else ""
    candidate = payload.get("token", "") if isinstance(payload, dict) else ""
    type_ok = isinstance(payload_type, str) and secrets.compare_digest(
        payload_type,
        "auth",
    )
    token_ok = isinstance(candidate, str) and secrets.compare_digest(
        candidate,
        expected_token,
    )
    if not type_ok or not token_ok:
        raise WebVRSecurityError("Quest access token is invalid")
