from __future__ import annotations

import numpy as np

from ws_core.tactile_wire import (
    DEFORMATION_KEY,
    DEFORMATION_VALID_KEY,
    FINGER_ORDER,
    FORCE_KEY,
    FORCE_VALID_KEY,
    ORDER_KEY,
    build_mot_tactile_request,
)


def observation(raw_length: int = 16) -> dict:
    force = [[index, 1, 2] for index in range(10)]
    torque = [[3, 4, 5] for _ in range(10)]
    entries = [
        {
            "side": name.split("_", 1)[0],
            "finger": name.split("_", 1)[1],
            "valid": True,
            "offset": index * raw_length,
            "length": raw_length,
            "height": 4,
            "width": 4,
        }
        for index, name in enumerate(FINGER_ORDER)
    ]
    return {
        "robot_state": {
            "payload": {
                "valid": True,
                "json": {
                    "tactile": {
                        "order": list(FINGER_ORDER),
                        "force": force,
                        "torque": torque,
                        "force_valid": [True] * 9 + [False],
                    }
                },
            }
        },
        "robot_tactile": {
            "metadata": {"valid": True, "json": {"entries": entries}}
        },
    }


def test_builds_force_torque_and_deformation_in_fixed_order() -> None:
    raw = b"".join(bytes([index]) * 16 for index in range(10))
    fields, info = build_mot_tactile_request(observation(), raw)

    assert fields[ORDER_KEY] == list(FINGER_ORDER)
    assert fields[FORCE_KEY].shape == (10, 6)
    np.testing.assert_array_equal(fields[FORCE_KEY][3], [3, 1, 2, 3, 4, 5])
    assert fields[FORCE_VALID_KEY].sum() == 9
    assert not fields[FORCE_KEY][-1].any()
    assert fields[DEFORMATION_KEY].shape == (10, 64, 64)
    assert fields[DEFORMATION_KEY][4, 0, 0] == 4
    assert fields[DEFORMATION_VALID_KEY].all()
    assert info["force_valid"] == 9
    assert info["deformation_valid"] == 10


def test_bad_bulk_entry_stays_invalid_and_zero() -> None:
    wrapper = observation()
    wrapper["robot_tactile"]["metadata"]["json"]["entries"][0]["offset"] = 999
    raw = bytes(range(160))
    fields, _ = build_mot_tactile_request(wrapper, raw)

    assert not fields[DEFORMATION_VALID_KEY][0]
    assert not fields[DEFORMATION_KEY][0].any()


def test_missing_tactile_returns_no_wire_fields() -> None:
    fields, info = build_mot_tactile_request({}, b"")
    assert fields == {}
    assert info["fallback"] == "no_tactile_payload"
