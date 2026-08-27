import pytest

from sharpa_interface.server.metadata import (
    MetadataValidationError,
    validate_server_metadata,
)

from test_policy_input import metadata_format


def server_metadata(action_schema: str = "sharpa_policy_action.v5") -> dict:
    return {
        "schema": "sharpa_policy_server.v3",
        "policy_family": "groot",
        "checkpoint_id": "checkpoint",
        "checkpoint_path": "",
        "task_id": "task",
        "run_id": "run",
        "dataset_path": "",
        "prompt": "",
        "transport": "websocket+binary_msgpack",
        "observation_schema": "sharpa_policy_observation.v3",
        "action_schema": action_schema,
        "host": "127.0.0.1",
        "port": 5500,
        "infer_path": "/infer",
        "health_path": "/healthz",
        "metadata_path": "/metadata",
        "reset_path": "/reset",
        "max_message_size": 64 * 1024 * 1024,
        "metadata_format": metadata_format(),
    }


def test_metadata_declares_v5_action_schema() -> None:
    normalized = validate_server_metadata(server_metadata())

    assert normalized["action_schema"] == "sharpa_policy_action.v5"


def test_metadata_temporarily_accepts_v4_for_sync_fallback() -> None:
    normalized = validate_server_metadata(
        server_metadata("sharpa_policy_action.v4")
    )

    assert normalized["action_schema"] == "sharpa_policy_action.v4"


def test_metadata_rejects_unknown_action_schema() -> None:
    with pytest.raises(MetadataValidationError, match="action_schema"):
        validate_server_metadata(server_metadata("sharpa_policy_action.v6"))
