"""Human takeover of the single Actor command outlet.

Vive supplies anchored Cartesian IK targets. AE01 supplies retargeted joints;
its deltas are added to the last commanded hand pose, without offset decay.
"""
from ur_common.protection_config import setting
import time
import uuid
import numpy as np

from .command_guards import vector, hand_bounds, validate_hand_range
from .protection import ProtectionError


class PolicyHandback:
    """Blend newly inferred policy actions over a 30 Hz physical command timeline."""
    def __init__(self, anchor, published, duration_s=None):
        duration_s = setting('hil', 'return_duration_s') if duration_s is None else duration_s
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('HIL return duration must be positive and finite')
        self.anchor = vector(anchor, 56).copy()
        validate_hand_range(self.anchor[12:])
        if np.any(abs(self.anchor[:12]) > setting('ur', 'max_abs_joint_rad')):
            raise ValueError('HIL return anchor exceeds absolute arm joint range')
        self.start = float(published)
        self.steps = max(1, int(np.ceil(duration_s * 30)))
        self.duration_s = self.steps / 30

    def progress(self, published):
        return float(np.clip((published-self.start)/self.steps, 0, 1))

    def blend(self, actions, published, frozen=0, step_scale=1.):
        result = actions.copy()
        for index in range(frozen, len(result)):
            x = self.progress(published + (index + 1)*step_scale)
            if x < 1:
                # Quintic smoothstep: zero endpoint velocity and acceleration.
                alpha = x*x*x*(10 + x*(-15 + 6*x))
                result[index] = self.anchor + alpha*(result[index]-self.anchor)
        return result


class HumanControl:
    def __init__(self, clock=time.monotonic, max_age=None):
        self.clock, self.max_age = clock, setting('hil', 'input_max_age_s') if max_age is None else max_age
        self.active = False
        self.token = ''
        self.return_pending = False
        self.samples = {}
        self.hand_origin = {}
        self.proposals = {}

    def begin(self, anchor):
        self.anchor = vector(anchor, 56).copy()
        lo, hi = hand_bounds()
        self.anchor[12:] = np.clip(self.anchor[12:], lo, hi)
        self.active = True
        self.return_pending = False
        self.token = uuid.uuid4().hex
        self.started = self.clock()
        self.samples.clear()
        self.hand_origin.clear()
        self.proposals.clear()

    def end(self):
        self.active = False
        self.return_pending = False
        self.samples.clear()

    def arm(self, side, target, token):
        if self.active and token == self.token:
            self.proposals[side + '_arm'] = vector(target, 6).tolist()
            self.samples[side + '_arm'] = (vector(target, 6).copy(), self.clock())

    def hand(self, side, target):
        if not self.active:
            return
        target = vector(target, 22).copy()
        self.hand_origin.setdefault(side, target.copy())
        offset = 12 if side == 'left' else 34
        command = self.anchor[offset:offset+22] + (target - self.hand_origin[side])
        self.proposals[side + '_hand'] = command.tolist()
        # Offsetting two valid poses can exceed a joint bound. Saturate the
        # human target here, before Actor validation, logging and SDK dispatch.
        lo, hi = hand_bounds()
        start = offset - 12
        command = np.clip(command, lo[start:start+22], hi[start:start+22])
        self.samples[side + '_hand'] = (command, self.clock())

    def target(self):
        if not self.active:
            return None
        result = self.anchor.copy()
        now = self.clock()
        for key, start, stop in (('left_arm', 0, 6), ('right_arm', 6, 12),
                                 ('left_hand', 12, 34), ('right_hand', 34, 56)):
            item = self.samples.get(key)
            if item is None or now-item[1] > self.max_age:
                if now-self.started > self.max_age:
                    raise ProtectionError('hil_input_stale', f'HIL {key} input missing or stale',
                        input_ages=self.snapshot()['input_ages'], limit_s=self.max_age)
                # Keep the entire robot at the entry anchor until all four inputs are available.
                return self.anchor.copy()
            result[start:stop] = item[0]
        return result

    def snapshot(self):
        now = self.clock()
        return dict(active=self.active, token=self.token, return_pending=self.return_pending,
                    proposals=dict(self.proposals) if self.active else {},
                    anchor=self.anchor.tolist() if self.active else None,
                    input_ages={key: now-value[1] for key, value in self.samples.items()})
