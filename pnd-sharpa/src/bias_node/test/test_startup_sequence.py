from bias_node.bias_node import (
    BIAS_INIT_MODE,
    BIAS_MODE,
    startup_sequence_steps,
)


def test_direct_bias_startup_uses_saved_bias_only():
    bias_init = {"joint": 1.0}
    bias = {"joint": 2.0}

    steps = startup_sequence_steps("bias", bias_init, bias)

    assert steps == [("bias", BIAS_MODE, bias)]
    assert steps[0][2] is not bias


def test_default_startup_keeps_bias_init_then_bias():
    bias_init = {"joint": 1.0}
    bias = {"joint": 2.0}

    steps = startup_sequence_steps("bias_init_then_bias", bias_init, bias)

    assert steps == [
        ("startup", BIAS_INIT_MODE, bias_init),
        ("bias", BIAS_MODE, bias),
    ]
