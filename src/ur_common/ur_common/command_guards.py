"""Physical command guards shared by the Actor and SharpA outlet.

No temporal filtering or slew is applied. Range projection is bounded and done
before queue admission; command jumps and tracking failures reject execution.
"""
from ur_common.protection_config import setting
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from .sharpa_fk import LEFT_JOINT_ORDER, RIGHT_JOINT_ORDER

CONTRACT_ID = 'ur_sharpa_joint_command_v1'


@lru_cache(maxsize=1)
def execution_profile():
    path = Path(__file__).parent / 'assets/command_calibration.json'
    result = json.loads(path.read_text())
    # This asset retains calibration/contract metadata, not adjustable thresholds.
    from .protection_config import load_protection_config
    result.update(load_protection_config()['hand'])
    if result['contract_id'] != CONTRACT_ID:
        raise ValueError('unknown execution contract')
    for key in ('command_step_rad', 'startup_step_rad', 'tracking_error_rad',
                'tracking_hard_rad', 'max_projection_rad'):
        value = np.asarray(result[key],dtype=float)
        # JSON "inf" disables the corresponding software tracking check.
        # Startup, command-step and projection thresholds must stay finite.
        invalid = np.isnan(value) | np.isneginf(value)
        if key not in ('tracking_error_rad', 'tracking_hard_rad'):
            invalid |= np.isposinf(value)
        if value.shape not in ((),(44,)) or np.any(invalid) or np.any(value <= 0):
            raise ValueError(f'invalid execution profile {key}')
        result[key] = np.broadcast_to(value, (44,)).tolist()
    if not np.isfinite(result['tracking_timeout_s']) or result['tracking_timeout_s'] <= 0:
        raise ValueError('invalid tracking timeout')
    tracking = np.asarray(result['tracking_error_rad'])
    if np.any(np.isfinite(tracking) & (tracking >= result['tracking_hard_rad'])):
        raise ValueError('tracking warning must be below hard stop')
    return result


@lru_cache(maxsize=1)
def firmware_limit_profile():
    path = Path(__file__).parent / 'assets/sharpa_firmware_limits.json'
    result = json.loads(path.read_text())
    if result.get('schema') != 'ur_sharpa.hand_firmware_limits.v1' or result.get('unit') != 'rad':
        raise ValueError('invalid firmware limit profile')
    for side, names in (('left', LEFT_JOINT_ORDER), ('right', RIGHT_JOINT_ORDER)):
        hand = result['hands'][side]
        lo, hi = (np.asarray(hand[key], dtype=np.float32) for key in
                  ('hard_min_position', 'hard_max_position'))
        if (hand['joint_names'] != names or lo.shape != (22,) or hi.shape != (22,)
                or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo >= hi)):
            raise ValueError(f'invalid {side} firmware joint bounds')
    return result


def validate_firmware_device(side, device):
    expected = firmware_limit_profile()['hands'][side]['device']
    for key in ('sn', 'pn', 'firmware_version'):
        actual = str(getattr(device, key, ''))
        if actual != expected[key]:
            raise ExecutionContractError('firmware_limits_mismatch', detail=
                f'{side} {key}: actual={actual}, expected={expected[key]}; read back limits for this device')


@lru_cache(maxsize=1)
def hand_bounds():
    lower, upper = [], []
    for side in ('left', 'right'):
        hand = firmware_limit_profile()['hands'][side]
        lower.extend(hand['hard_min_position'])
        upper.extend(hand['hard_max_position'])
    # Round to the same representation used by ROS float32 commands.
    lo, hi = np.asarray(lower, np.float32), np.asarray(upper, np.float32)
    lo.flags.writeable = hi.flags.writeable = False
    return lo, hi


def vector(value, size):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError(f'expected {size} finite joint values')
    return arr


def joint_details(indices, measured, target, limit):
    rows = []
    for index in indices:
        index = int(index)
        group = 'left_hand' if index < 22 else 'right_hand'
        rows.append(dict(group=group, joint_number=index % 22 + 1, state_index=index+12,
                         measured_rad=float(measured[index]), target_rad=float(target[index]),
                         delta_rad=float(target[index])-float(measured[index]), limit_rad=float(np.broadcast_to(limit,(44,))[index])))
    return rows


class ExecutionContractError(ValueError):
    def __init__(self, code, rows=None, detail=''):
        self.code, self.violations = code, rows or []
        values = '; '.join(f"{r['group']} joint={r['joint_number']} state_index={r['state_index']} "
                          f"measured={r['measured_rad']:.6f} target={r['target_rad']:.6f} "
                          f"delta={r['delta_rad']:+.6f} limit={r['limit_rad']:.6f} rad"
                          for r in self.violations)
        super().__init__(f'{code}: {detail} {values}'.strip())


def arm_joint_details(indices, reference, target, limit):
    return [dict(group='left_arm' if i < 6 else 'right_arm',
                 joint_number=int(i % 6 + 1), state_index=int(i),
                 measured_rad=float(reference[i]), target_rad=float(target[i]),
                 delta_rad=float(target[i]-reference[i]), limit_rad=float(limit)) for i in indices]


def validate_arm_step(reference, target, limit=None, code='arm_command_jump'):
    if limit is None:
        limit = setting('actor', 'arm_startup_step_rad' if code == 'arm_startup_pose_incompatible' else 'arm_command_step_rad')
    reference, target = vector(reference, 12), vector(target, 12)
    bad = np.flatnonzero(abs(target-reference) > limit)
    if len(bad):
        raise ExecutionContractError(code, arm_joint_details(bad, reference, target, limit))


def validate_hand_range(target):
    target = vector(target, 44)
    lo, hi = hand_bounds()
    bad = np.flatnonzero((target < lo) | (target > hi))
    if len(bad):
        rows = joint_details(bad, np.clip(target, lo, hi), target, 0)
        for row, i in zip(rows, bad):
            row.update(lower_rad=float(lo[i]), upper_rad=float(hi[i]))
        raise ExecutionContractError('hand_range_rejected',
            rows, 'target outside firmware hard position bounds; driver never clips policy targets')
    return target


class HandCommandGuard:
    def __init__(self):
        self.profile = execution_profile()
        self.reset()

    def reset(self):
        self.last_target = None
        self.error_since = np.full(44, np.nan)

    def check_tracking(self, measured, now):
        measured = vector(measured, 44)
        if self.last_target is None:
            return
        delta = abs(self.last_target-measured)
        hard = np.flatnonzero(delta > self.profile['tracking_hard_rad'])
        if len(hard):
            raise ExecutionContractError('hand_tracking_hard_stop',
                joint_details(hard, measured, self.last_target, self.profile['tracking_hard_rad']))
        over = delta > self.profile['tracking_error_rad']
        self.error_since[~over] = np.nan
        begin = over & np.isnan(self.error_since)
        self.error_since[begin] = now
        expired = np.flatnonzero(over & (now-self.error_since >= self.profile['tracking_timeout_s']))
        if len(expired):
            raise ExecutionContractError('hand_tracking_timeout',
                joint_details(expired, measured, self.last_target, self.profile['tracking_error_rad']),
                f"duration >= {self.profile['tracking_timeout_s']} s")

    def check(self, measured, target, now):
        measured, target = vector(measured,44), validate_hand_range(target)
        self.check_tracking(measured, now)
        reference = measured if self.last_target is None else self.last_target
        limit = self.profile['startup_step_rad'] if self.last_target is None else self.profile['command_step_rad']
        bad = np.flatnonzero(abs(target-reference) > limit)
        if len(bad):
            raise ExecutionContractError('startup_pose_incompatible' if self.last_target is None else 'hand_command_jump',
                joint_details(bad, reference, target, limit),
                'prepare task reset pose and re-observe' if self.last_target is None else 'reference is last accepted command')
        hard = np.flatnonzero(abs(target-measured) > self.profile['tracking_hard_rad'])
        if len(hard):
            raise ExecutionContractError('hand_tracking_hard_stop',
                joint_details(hard, measured, target, self.profile['tracking_hard_rad']))

    def commit(self, target):
        self.last_target = vector(target,44).copy()
