"""Validated startup-only protection settings shared by ROS and policy code."""
from copy import deepcopy
from functools import lru_cache
import math
import os
from pathlib import Path

import yaml

ENVIRONMENT_VARIABLE = 'UR_SHARPA_PROTECTION_CONFIG'
FIELDS = {
    'observation': ('observation_max_age_s', 'observation_max_skew_s', 'future_tolerance_s'),
    'status': ('camera_max_age_s', 'execution_start_timeout_s'),
    'actor': ('arm_startup_step_rad', 'arm_command_step_rad', 'arm_tracking_error_rad',
              'hand_step_rad', 'model_heartbeat_timeout_s', 'learning_command_max_age_s',
              'stop_ack_grace_s'),
    'ur': ('max_abs_joint_rad', 'max_joint_step_rad', 'joint_vel_limit'),
    'hil': ('input_max_age_s', 'return_duration_s'),
    'vive': ('stale_timeout_s', 'max_translation_delta_m', 'max_rotation_delta_rad',
             'initial_joint_tolerance_rad'),
    'ae01': ('data_timeout_s',),
    'tactile': ('max_age_s',),
    'manus': ('startup_grace_s', 'raw_timeout_s', 'incomplete_hand_timeout_s', 'action_timeout_s'),
    'camera': ('ego_first_frame_timeout_s', 'ego_frame_stall_timeout_s'),
    'recording': ('camera_max_age_s', 'heartbeat_timeout_s', 'start_timeout_s', 'saving_timeout_s'),
    'reset': ('max_abs_hand_joint_rad', 'ur_tolerance_rad', 'sharpa_tolerance_rad', 'arrival_timeout_s',
              'prepare_timeout_s', 'command_ack_timeout_s', 'state_timeout_s'),
    'ik': ('jump_vel', 'jump_rot_vel', 'jump_max_reject', 'flip_threshold_vel_multiple'),
    'hand': ('command_step_rad', 'startup_step_rad', 'tracking_error_rad',
             'tracking_hard_rad', 'max_projection_rad', 'tracking_timeout_s'),
}


def default_protection_path():
    workspace = os.environ.get('UR_SHARPA_ROOT', '').strip()
    if workspace:
        candidate = Path(workspace).expanduser() / 'config/protection.yaml'
        if candidate.is_file():
            return candidate.resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'config/protection.yaml'
        if candidate.is_file():
            return candidate
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('ur_common')) / 'config/protection.yaml'


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f'duplicate protection setting: {key}')
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@lru_cache(maxsize=8)
def _load(path):
    data = yaml.load(Path(path).read_text(), Loader=_UniqueLoader)
    if not isinstance(data, dict) or data.get('schema') != 'ur_sharpa.protection.v1':
        raise ValueError(f'{path}: invalid protection schema')
    if set(data) != {'schema', *FIELDS}:
        raise ValueError(f'{path}: protection sections must be {sorted(FIELDS)}')
    for section, fields in FIELDS.items():
        values = data[section]
        if not isinstance(values, dict) or set(values) != set(fields):
            raise ValueError(f'{path}: {section} fields must be {sorted(fields)}')
        for key, value in values.items():
            vector = section == 'hand' and key != 'tracking_timeout_s'
            items = value if isinstance(value, list) else [value]
            if isinstance(value, list) and (not vector or len(value) != 44):
                raise ValueError(f'{path}: {section}.{key} requires scalar or 44 values')
            allow_inf = section == 'hand' and key in ('tracking_error_rad', 'tracking_hard_rad')
            if any(type(v) not in (int, float) or v <= 0 or math.isnan(v)
                   or (not allow_inf and not math.isfinite(v)) for v in items):
                raise ValueError(f'{path}: {section}.{key} must be positive and finite')
            if key in ('jump_max_reject',) and type(value) is not int:
                raise ValueError(f'{path}: {section}.{key} must be an integer')
            if key not in ('jump_max_reject',):
                # Keep ROS parameter types stable when an operator writes 1 instead of 1.0.
                values[key] = [float(v) for v in value] if isinstance(value, list) else float(value)
    hand = data['hand']
    expand = lambda value: value if isinstance(value, list) else [value]*44
    if any(math.isfinite(soft) and soft >= hard for soft, hard in
           zip(expand(hand['tracking_error_rad']), expand(hand['tracking_hard_rad']))):
        raise ValueError(f'{path}: hand tracking warning must be below hard stop')
    return data


def protection_path():
    configured = os.environ.get(ENVIRONMENT_VARIABLE, '').strip()
    return _resolved_path(configured, os.environ.get('UR_SHARPA_ROOT', ''))


@lru_cache(maxsize=8)
def _resolved_path(configured, workspace):
    return Path(configured).expanduser().resolve() if configured else default_protection_path()


def load_protection_config(path=None):
    """Return an independent copy; explicit missing/bad files never fall back."""
    return deepcopy(_load(str(Path(path).expanduser().resolve() if path else protection_path())))


def setting(section, key):
    return deepcopy(_load(str(protection_path()))[section][key])
