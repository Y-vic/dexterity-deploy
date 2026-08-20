from sensor_msgs.msg import Joy

from status_node.status_node import StatusNode


def _joy(axes: list[float], buttons: list[int]) -> Joy:
    msg = Joy()
    msg.axes = axes
    msg.buttons = buttons
    return msg


def _rt_pressed(msg: Joy) -> bool:
    node = object.__new__(StatusNode)
    return node._trigger_pressed(
        msg,
        axis=4,
        button=9,
        threshold=0.5,
        pressed_when="above",
    )


def test_rt_accepts_analog_axis_4():
    axes = [0.0] * 8
    axes[4] = 0.75

    assert _rt_pressed(_joy(axes, [0] * 15))


def test_rt_accepts_digital_button_9_fallback():
    buttons = [0] * 15
    buttons[9] = 1

    assert _rt_pressed(_joy([0.0] * 8, buttons))


def test_rt_neutral_is_not_pressed():
    assert not _rt_pressed(_joy([0.0] * 8, [0] * 15))
