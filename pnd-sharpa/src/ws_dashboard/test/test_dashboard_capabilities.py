import json
import threading
from types import SimpleNamespace

import pytest

from ws_dashboard.dashboard import (
    DashboardNode,
    MODEL_IMAGE_HEIGHT,
    MODEL_IMAGE_WIDTH,
    SharedFrame,
    _pred_video_capability,
    _prediction_video_path,
)


def _dashboard_node() -> DashboardNode:
    node = object.__new__(DashboardNode)
    node.lock = threading.Lock()
    node.policy_provider = ""
    node.pred_video_capability = "unknown"
    node.frames = {
        "pred_video": SharedFrame(
            MODEL_IMAGE_WIDTH,
            MODEL_IMAGE_HEIGHT,
            "Predicted Video",
            82,
        )
    }
    node._pred_video_condition = threading.Condition()
    node._pending_pred_video = None
    return node


@pytest.mark.parametrize(
    "provider",
    [
        "cgp",
        "gcc",
        "groot_n17",
        "pace",
        "trex",
        "vitacformer",
    ],
)
def test_pose_only_providers_do_not_report_prediction_video(provider: str) -> None:
    assert _pred_video_capability(provider) == "unsupported"


@pytest.mark.parametrize("provider", ["dreamzero", "dz", "sharpa62"])
def test_dreamzero_providers_report_prediction_video(provider: str) -> None:
    assert _pred_video_capability(provider) == "supported"


def test_unknown_provider_keeps_capability_unknown() -> None:
    assert _pred_video_capability("future_model") == "unknown"


@pytest.mark.parametrize(
    "payload",
    [
        {"server_video_pred_path": "/tmp/top.mp4"},
        {"debug": {"server_video_pred_path": "/tmp/debug.mp4"}},
        {"diagnostics": {"server_video_pred_path": "/tmp/diagnostics.mp4"}},
    ],
)
def test_prediction_video_path_accepts_supported_field_locations(payload: dict) -> None:
    assert _prediction_video_path(payload).endswith(".mp4")


def test_prediction_video_path_prefers_top_level_value() -> None:
    assert _prediction_video_path(
        {
            "server_video_pred_path": "/tmp/top.mp4",
            "debug": {"server_video_pred_path": "/tmp/debug.mp4"},
            "diagnostics": {"server_video_pred_path": "/tmp/diagnostics.mp4"},
        }
    ) == "/tmp/top.mp4"


def test_pose_only_provider_marks_video_panel_unsupported() -> None:
    node = _dashboard_node()

    node._set_policy_provider("PACE")

    status = node.frames["pred_video"].snapshot()[1]
    assert status["state"] == "unsupported"
    assert status["capability"] == "unsupported"
    assert status["provider"] == "pace"


def test_dreamzero_provider_waits_for_supported_video() -> None:
    node = _dashboard_node()
    node._set_policy_provider("pace")

    node._set_policy_provider("dreamzero")

    status = node.frames["pred_video"].snapshot()[1]
    assert status["state"] == "starting"
    assert status["capability"] == "supported"
    assert status["provider"] == "dreamzero"


def test_policy_prediction_queues_diagnostics_video_path() -> None:
    node = _dashboard_node()
    message = SimpleNamespace(
        header=SimpleNamespace(),
        seq=7,
        provider="dreamzero",
        payload_json=json.dumps(
            {
                "request_index": 12,
                "diagnostics": {
                    "server_video_pred_path": "/tmp/request_000012.mp4"
                },
            }
        ),
    )

    summary = node._policy_pred_summary(message)

    assert summary["provider"] == "dreamzero"
    assert node._pending_pred_video == (12, "/tmp/request_000012.mp4")
