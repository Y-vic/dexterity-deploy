from types import SimpleNamespace
import json

import numpy as np
import pytest

from ur_common.hil import HumanControl
from ur_common.command_guards import (
    ExecutionContractError, firmware_limit_profile, hand_bounds,
    validate_arm_step, validate_firmware_device, validate_hand_range,
)
from ur_common.protection import protection_event


def test_firmware_table_preserves_device_order_and_asymmetric_limits():
    profile = firmware_limit_profile()
    lo, hi = hand_bounds()
    for offset, side in ((0, 'left'), (22, 'right')):
        hand = profile['hands'][side]
        np.testing.assert_array_equal(lo[offset:offset+22], np.float32(hand['hard_min_position']))
        np.testing.assert_array_equal(hi[offset:offset+22], np.float32(hand['hard_max_position']))
        validate_firmware_device(side, SimpleNamespace(**hand['device']))
    # Hardware right ring MCP is below the old URDF 90 degree upper bound.
    assert hi[22+13] < np.pi/2
    assert not np.array_equal(lo[:22], lo[22:])
    for bound in (lo, hi):
        validate_hand_range(bound)
    target = np.zeros(44, np.float32)
    target[5] = -.5  # Real firmware permits a negative position at this joint.
    validate_hand_range(target)


@pytest.mark.parametrize('field', ['sn', 'pn', 'firmware_version'])
def test_changed_device_cannot_use_another_devices_limit_table(field):
    device = SimpleNamespace(**firmware_limit_profile()['hands']['left']['device'])
    setattr(device, field, 'different')
    with pytest.raises(ExecutionContractError, match='firmware_limits_mismatch') as caught:
        validate_firmware_device('left', device)
    assert protection_event(caught.value)['code'] == 'firmware_limits_mismatch'


def test_exact_upper_boundary_and_next_float_report_correct_joint_and_limits():
    lo, hi = hand_bounds()
    target = hi.copy()
    target[35] = np.nextafter(target[35], np.float32(np.inf))
    with pytest.raises(ExecutionContractError) as caught:
        validate_hand_range(target)
    event = protection_event(caught.value)
    assert event['code'] == 'hand_range_rejected'
    row = event['violations'][0]
    assert row['group'] == 'right_hand' and row['joint_number'] == 14
    assert row['upper_rad'] == hi[35] and row['lower_rad'] == lo[35]
    assert '右手关节 14' in event['details'][0]
    json.dumps(event, allow_nan=False)


@pytest.mark.parametrize('code', ['arm_startup_pose_incompatible', 'arm_command_jump', 'arm_tracking_error'])
def test_arm_protection_reports_side_joint_and_threshold(code):
    target = np.zeros(12); target[8] = .251
    with pytest.raises(ExecutionContractError) as caught:
        validate_arm_step(np.zeros(12), target, code=code)
    event = protection_event(caught.value)
    assert event['code'] == code
    assert event['violations'][0]['joint_number'] == 3
    assert event['violations'][0]['limit_rad'] == .25
    assert '右臂关节 3' in event['details'][0]


def test_missing_hil_input_is_saved_before_hil_state_is_cleared():
    now = [1.]
    hil = HumanControl(clock=lambda: now[0]); hil.begin(np.zeros(56))
    hil.arm('left', np.zeros(6), hil.token)
    hil.arm('right', np.zeros(6), hil.token)
    now[0] += .26
    hil.arm('left', np.zeros(6), hil.token)
    hil.arm('right', np.zeros(6), hil.token)
    with pytest.raises(RuntimeError) as caught:
        hil.target()
    hil.end()
    event = protection_event(caught.value)
    assert event['code'] == 'hil_input_stale'
    assert len(event['details']) == 2
    assert all('未收到输入' in line and '250 ms' in line for line in event['details'])


def test_eef_rejection_reports_side_step_and_flip_threshold():
    error = ExecutionContractError('eef_ik_rejected')
    error.context = dict(side='left', chunk_step=13, ik_rejection=dict(
        stage='WRIST_FLIP_HOLD', minimum_max_delta_rad=2.403, flip_threshold_rad=2.1))
    event = protection_event(error)
    assert event['title'] == '末端位姿逆解被拒绝'
    assert '左臂，预测动作第 14 步' in event['details'][0]
    assert '2.403000 rad' in event['details'][1]
    assert event['context'] == error.context
    json.dumps(event, allow_nan=False)




@pytest.mark.parametrize('message,code', [
    ('hand_tracking_hard_stop: right_hand joint=1', 'hand_tracking_hard_stop'),
    ('UR not in normal running mode', 'ur_hardware_protection'),
    ('unknown SDK refusal', 'hardware_rejected'),
])
def test_hardware_ack_and_runtime_messages_have_specific_protection_types(message, code):
    event = protection_event(message, source='sharpa', hardware_code='sharpa_command_failed')
    assert event['code'] == code and event['message'] == message
    assert event['source'] == 'sharpa'
