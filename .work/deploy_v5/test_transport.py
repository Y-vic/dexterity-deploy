"""End-to-end transport tests: a real socket, the real client, no model.

These start the server in --dry-run on an ephemeral port and drive it with the
workstation's own `SharpaV3PolicyClient`. That client is the thing that will talk
to this server in production, so anything it rejects is a real defect --
including the failure modes a hand-written client would not reproduce, like the
"server must not greet on /infer" rule.
"""

import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "interface"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import server as srv
from test_contract import FORMAT_ID, make_metadata, make_observation

from sharpa_interface.server.transport import SharpaV3PolicyClient


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ServerThread:
    """Run the socket accept loop in a thread for the duration of one test."""

    def __init__(self) -> None:
        self.port = free_port()
        self.server = srv.PolicyServer(
            srv.MotPolicy("", "", ""),
            metadata=None,
            format_id=FORMAT_ID,
            prompt="",
            dry_run=True,
        )
        self.server.metadata = make_metadata()
        self.server.metadata["port"] = self.port
        self.server.ready = True
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def __enter__(self) -> "ServerThread":
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.port))
        listener.listen(4)
        listener.settimeout(0.2)
        self._listener = listener

        def handle(sock: socket.socket, address) -> None:
            try:
                srv.serve_connection(self.server, sock, address)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    sock, address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                # Threaded for the same reason main() is: the client keeps its
                # /infer websocket open while calling /reset on another
                # connection, so a serial loop deadlocks. This harness caught
                # exactly that bug, so it must not paper over it.
                threading.Thread(
                    target=handle, args=(sock, address), daemon=True
                ).start()

        self._thread = threading.Thread(target=loop, name="serve", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def test_healthz_answers_before_the_model_is_ready():
    """The relay exits on a non-2xx, so /healthz must answer while loading."""
    with ServerThread() as harness:
        harness.server.ready = False
        with socket.create_connection(("127.0.0.1", harness.port), timeout=5) as sock:
            sock.sendall(b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            response = sock.recv(4096)
        assert b"200 OK" in response
        assert b"loading" in response


def test_healthz_reports_ok_when_ready():
    with ServerThread() as harness:
        with socket.create_connection(("127.0.0.1", harness.port), timeout=5) as sock:
            sock.sendall(b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            response = sock.recv(4096)
        assert b"200 OK" in response
        assert b'"ok"' in response


def test_metadata_over_the_real_client():
    """The client fetches and validates /metadata itself on construction."""
    with ServerThread() as harness:
        client = SharpaV3PolicyClient(url=f"ws://127.0.0.1:{harness.port}/infer")
        try:
            metadata = client.metadata
            assert metadata["schema"] == srv.SERVER_SCHEMA
            assert metadata["metadata_format"]["format_id"] == FORMAT_ID
        finally:
            client.close()


def test_full_reset_and_infer_over_the_real_client():
    """The whole cycle across a real websocket, driven by the real client."""
    with ServerThread() as harness:
        client = SharpaV3PolicyClient(url=f"ws://127.0.0.1:{harness.port}/infer")
        try:
            reset = client.reset("session-under-test")
            assert reset["reset"] is True
            session_id = reset["session_id"]

            result = client.infer(make_observation(session_id, request_id=1))
            action = result.payload if hasattr(result, "payload") else result
            assert action["schema"] == srv.ACTION_SCHEMA
            assert action["session_id"] == session_id
            assert action["request_id"] == 1
            assert action["execution"]["rtc"] == srv.RTC_STEPS
            assert action["action"]["left_wrist"]["eef"].shape == (srv.ACTION_HORIZON, 9)
        finally:
            client.close()


def test_several_inferences_on_one_socket():
    """The socket is reused across a rollout; state must persist per session."""
    with ServerThread() as harness:
        client = SharpaV3PolicyClient(url=f"ws://127.0.0.1:{harness.port}/infer")
        try:
            session_id = client.reset("multi")["session_id"]
            for request_id in range(4):
                result = client.infer(make_observation(session_id, request_id))
                action = result.payload if hasattr(result, "payload") else result
                assert action["request_id"] == request_id
            # The session counted every request and was never reset mid-rollout.
            # There is no server-side pose ring any more: state history rides in
            # on each observation, so nothing here accumulates across requests.
            assert harness.server._session.request_count == 4
        finally:
            client.close()


def test_an_inference_error_comes_back_as_a_typed_error():
    """A bad observation must produce a PolicyServerError at the parse layer.

    The transport itself does not raise on an error response -- it returns the
    payload, and `parse_policy_action` is what recognizes the error schema and
    raises. So the assertion belongs there: this checks the error survives the
    wire AND carries the fields the parser needs (code, message, retryable),
    which is the part a hand-built error dict gets wrong.
    """
    from sharpa_interface.server.action import PolicyServerError, parse_policy_action

    with ServerThread() as harness:
        client = SharpaV3PolicyClient(url=f"ws://127.0.0.1:{harness.port}/infer")
        try:
            session_id = client.reset("errors")["session_id"]
            broken = make_observation(session_id, 0)
            broken["state"]["current"]["left_wrist"]["eef"] = None

            result = client.infer(broken)
            payload = result.payload
            assert payload["schema"] == srv.ERROR_SCHEMA

            with pytest.raises(PolicyServerError) as caught:
                parse_policy_action(
                    payload, expected_session_id=session_id, expected_request_id=0
                )
            assert "eef" in str(caught.value).lower()
            assert caught.value.retryable is False
        finally:
            client.close()


def test_the_socket_survives_an_inference_error():
    """One bad frame must not end the rollout.

    The client reuses the socket, so a server that closed on error would turn a
    single malformed observation into a lost session.
    """
    with ServerThread() as harness:
        client = SharpaV3PolicyClient(url=f"ws://127.0.0.1:{harness.port}/infer")
        try:
            session_id = client.reset("recover")["session_id"]
            broken = make_observation(session_id, 0)
            broken["state"]["current"]["left_wrist"]["eef"] = None
            assert client.infer(broken).payload["schema"] == srv.ERROR_SCHEMA

            # The next good observation must be answered normally.
            good = client.infer(make_observation(session_id, 1)).payload
            assert good["schema"] == srv.ACTION_SCHEMA
            assert good["request_id"] == 1
        finally:
            client.close()
