import threading
import time
from types import SimpleNamespace

from bias_node.bias_node import BiasNode, BiasRun, PHASE_UR, REQUEST_HOME_UR
from ur_msgs.msg import ArmCommand


def _active_node():
    node = object.__new__(BiasNode)
    node._lock = threading.RLock()
    node._counts = {"nacks": 0}
    node._last_nack = ""
    node._run = BiasRun(
        request=REQUEST_HOME_UR,
        started_at=time.monotonic(),
        phase=PHASE_UR,
    )
    failures = []
    node._fail_locked = failures.append
    return node, failures


def _nack(source):
    return SimpleNamespace(
        source=source,
        success=False,
        device="ur",
        failure_code="arbitration_rejected",
        failure_message="command source is not active",
    )


def test_vive_nack_does_not_fail_active_bias_run():
    node, failures = _active_node()

    node._on_command_result(_nack(ArmCommand.SOURCE_VIVE))

    assert failures == []
    assert node._counts["nacks"] == 0
    assert node._run.phase == PHASE_UR


def test_bias_nack_fails_active_bias_run():
    node, failures = _active_node()

    node._on_command_result(_nack(ArmCommand.SOURCE_BIAS))

    assert failures == [
        "ur rejected a bias step: arbitration_rejected command source is not active"
    ]
    assert node._counts["nacks"] == 1
