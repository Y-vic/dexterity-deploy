from unittest.mock import MagicMock, patch

import pytest

from sharpa_interface.server.client import ServerClient, ServerConfig


def test_client_is_not_created_after_shutdown() -> None:
    client = ServerClient(ServerConfig())
    client.close()

    with patch(
        "sharpa_interface.server.client.SharpaV3PolicyClient"
    ) as transport_type, pytest.raises(ConnectionError, match="shutting down"):
        client.connect()

    transport_type.assert_not_called()


def test_transport_created_during_shutdown_is_closed_not_installed() -> None:
    client = ServerClient(ServerConfig())
    candidate = MagicMock()
    candidate.metadata = {}

    def construct(*args, **kwargs):
        client.close()
        return candidate

    with patch(
        "sharpa_interface.server.client.SharpaV3PolicyClient",
        side_effect=construct,
    ), pytest.raises(ConnectionError, match="shutting down"):
        client.connect()

    candidate.close.assert_called_once_with()


def test_failed_inference_detaches_and_closes_transport() -> None:
    client = ServerClient(ServerConfig())
    transport = MagicMock()
    transport.metadata = {}
    transport.infer.side_effect = ConnectionError("server stopped")

    with patch(
        "sharpa_interface.server.client.SharpaV3PolicyClient",
        return_value=transport,
    ), patch(
        "sharpa_interface.server.client.validate_server_metadata",
        return_value={
            "schema": "sharpa_policy_server.v3",
            "policy_family": "trex",
            "prompt": "",
        },
    ):
        client.connect()
        with pytest.raises(ConnectionError, match="server stopped"):
            client.infer({"schema": "sharpa_policy_observation.v3"})

    transport.close.assert_called_once_with()
