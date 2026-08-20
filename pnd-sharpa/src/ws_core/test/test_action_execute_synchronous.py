import pytest

from ws_core.action_execute import ActionExecute


def execution() -> dict:
    return {
        "action_id": "action-1",
        "revision": 2,
        "execute_start": 4,
        "execute_length": 4,
        "action_length": 16,
        "frequency_hz": 30.0,
        "server_driven_execution": True,
    }


def test_server_execution_slice_is_the_plan_horizon() -> None:
    parsed = ActionExecute._server_execution_metadata(
        {"_ws_sharpa_v4": execution()}, horizon=4, action_hz=30.0
    )
    assert parsed == execution()


@pytest.mark.parametrize(
    ("horizon", "action_hz", "message"),
    [
        (3, 30.0, "horizon"),
        (4, 15.0, "action_hz"),
    ],
)
def test_server_execution_cannot_be_overridden_locally(
    horizon: int, action_hz: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ActionExecute._server_execution_metadata(
            {"_ws_sharpa_v4": execution()},
            horizon=horizon,
            action_hz=action_hz,
        )
