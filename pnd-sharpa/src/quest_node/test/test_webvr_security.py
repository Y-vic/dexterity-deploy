import json

import pytest

from quest_node.webvr_security import (
    WebVRSecurityError,
    authenticate_first_message,
    generate_access_token,
    normalize_public_web_url,
    validate_access_token,
    validate_secure_same_origin,
)


def secure_headers(**overrides):
    headers = {
        "Host": "10.10.20.127",
        "Origin": "https://10.10.20.127",
        "X-Forwarded-Proto": "https",
    }
    headers.update(overrides)
    return headers


def test_generated_token_is_valid_and_authenticates():
    token = validate_access_token(generate_access_token())

    authenticate_first_message(
        json.dumps({"type": "auth", "token": token}),
        token,
    )


@pytest.mark.parametrize(
    "message",
    [
        "not-json",
        json.dumps({"type": "sample", "token": "x" * 24}),
        json.dumps({"type": "auth", "token": "wrong-token-value-000000"}),
        json.dumps(["auth", "x" * 24]),
    ],
)
def test_authentication_rejects_malformed_or_wrong_first_message(message):
    with pytest.raises(WebVRSecurityError):
        authenticate_first_message(message, "expected-token-value-0000")


def test_secure_same_origin_accepts_default_https_port():
    validate_secure_same_origin(secure_headers())
    validate_secure_same_origin(
        secure_headers(
            Host="pnd.local:443",
            Origin="https://pnd.local",
        )
    )


@pytest.mark.parametrize(
    "headers",
    [
        secure_headers(Origin="http://10.10.20.127"),
        secure_headers(Origin="https://attacker.invalid"),
        secure_headers(**{"X-Forwarded-Proto": "http"}),
        {"Host": "10.10.20.127", "X-Forwarded-Proto": "https"},
        secure_headers(Host=["10.10.20.127", "attacker.invalid"]),
    ],
)
def test_secure_same_origin_rejects_untrusted_requests(headers):
    with pytest.raises(WebVRSecurityError):
        validate_secure_same_origin(headers)


def test_explicit_access_token_must_not_be_weak():
    with pytest.raises(WebVRSecurityError, match="between"):
        validate_access_token("short")


def test_public_url_requires_https_and_normalizes_trailing_slash():
    assert (
        normalize_public_web_url("https://10.10.20.127/webvr")
        == "https://10.10.20.127/webvr/"
    )
    with pytest.raises(WebVRSecurityError, match="HTTPS URL"):
        normalize_public_web_url("http://10.10.20.127/webvr/")
