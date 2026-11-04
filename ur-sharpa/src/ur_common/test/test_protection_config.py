from pathlib import Path

import numpy as np
import pytest
import yaml

from ur_common.protection_config import (
    ENVIRONMENT_VARIABLE, default_protection_path, load_protection_config, setting,
)


def write_config(tmp_path, change):
    data = load_protection_config(default_protection_path())
    change(data)
    path = tmp_path / 'protection.yaml'
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize('section,key,value', [
    ('observation', 'observation_max_age_s', -1),
    ('status', 'camera_max_age_s', float('nan')),
    ('ur', 'max_joint_step_rad', float('inf')),
    ('hil', 'input_max_age_s', True),
    ('ik', 'jump_max_reject', 2.5),
    ('actor', 'stop_ack_grace_s', -1),
    ('hand', 'command_step_rad', [0.4]*43),
    ('hand', 'startup_step_rad', float('inf')),
    ('hand', 'tracking_error_rad', -float('inf')),
    ('tactile', 'max_age_s', 0),
])
def test_invalid_thresholds_fail_before_execution(tmp_path, section, key, value):
    path = write_config(tmp_path, lambda data: data[section].update({key: value}))
    with pytest.raises(ValueError, match=section):
        load_protection_config(path)


@pytest.mark.parametrize('mutation', [
    lambda d: d['status'].update(camera_max_age_typo=0.5),
    lambda d: d['status'].pop('camera_max_age_s'),
    lambda d: d.update(extra={}),
    lambda d: d['hand'].update(tracking_error_rad=0.5, tracking_hard_rad=0.4),
])
def test_missing_unknown_or_inconsistent_settings_rejected(tmp_path, mutation):
    with pytest.raises(ValueError):
        load_protection_config(write_config(tmp_path, mutation))


def test_duplicate_keys_and_missing_file_do_not_fall_back(tmp_path, monkeypatch):
    path = tmp_path / 'duplicate.yaml'
    path.write_text('schema: ur_sharpa.protection.v1\nschema: ur_sharpa.protection.v1\n')
    with pytest.raises(ValueError, match='duplicate'):
        load_protection_config(path)
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, str(tmp_path/'missing.yaml'))
    with pytest.raises(FileNotFoundError):
        setting('status', 'camera_max_age_s')


def test_custom_file_controls_policy_hil_and_hand_guards(tmp_path, monkeypatch):
    from ur_common.command_guards import execution_profile, validate_arm_step, ExecutionContractError, HandCommandGuard
    from ur_common.hil import HumanControl, PolicyHandback
    from ur_common.command_guards import hand_bounds
    def change(data):
        data['observation'].update(observation_max_age_s=.6, observation_max_skew_s=.3)
        data['actor'].update(arm_startup_step_rad=.1, arm_command_step_rad=.2)
        data['hil'].update(input_max_age_s=.7, return_duration_s=1.5)
        data['hand'].update(command_step_rad=.12)
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, str(write_config(tmp_path, change)))
    execution_profile.cache_clear()
    try:
        validate_arm_step(np.zeros(12), np.full(12, .15))
        with pytest.raises(ExecutionContractError, match='arm_startup_pose_incompatible'):
            validate_arm_step(np.zeros(12), np.full(12, .15), code='arm_startup_pose_incompatible')
        assert HumanControl().max_age == .7
        lo, hi = hand_bounds()
        hand = (lo+hi)/2
        assert PolicyHandback(np.r_[np.zeros(12), hand], 0).steps == 45
        guard = HandCommandGuard()
        guard.commit(hand)
        target = hand.copy(); target[0] += .15
        with pytest.raises(ExecutionContractError, match='hand_command_jump'):
            guard.check(hand, target, 0)
        assert len(execution_profile()['command_step_rad']) == 44
    finally:
        execution_profile.cache_clear()


def test_integer_spelling_preserves_float_ros_parameter_types(tmp_path):
    path = write_config(tmp_path, lambda data: data['hil'].update(return_duration_s=1))
    assert type(load_protection_config(path)['hil']['return_duration_s']) is float
