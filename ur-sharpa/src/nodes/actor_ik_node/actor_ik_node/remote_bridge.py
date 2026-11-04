"""Remote-policy recording and optional HIL, without owning inference or models."""
import json
import time
import numpy as np
from ur_common.hil import HumanControl, PolicyHandback
from ur_common.rollout_capture import RolloutCapture
from ur_common.command_guards import HandCommandGuard, validate_arm_step
from ur_common.protection_config import setting


class RemoteExecutionBridge:
    def __init__(self, node):
        from std_msgs.msg import String, Bool
        from ur_msgs.msg import ArmCommand, HandCommand
        self.node, self.String = node, String
        self.human = HumanControl()
        self.handback = None
        self.guard = HandCommandGuard()
        self.last_target = None
        self.published = 0
        self.timeline_steps = 0.
        self.was_running = False
        self.pending_rejection = None
        self.next_due = 0.
        self.last_status = 0.
        self.record = False
        self.was_active = False
        self.error = ''
        self.learning_command = None
        self.policy_seen = 0.
        self.status_pub = node.create_publisher(String, '/hil/state', 10)
        self.rollout_pub = node.create_publisher(String, '/hil/rollout', 10)
        self.commands = node.create_publisher(String, '/control/ui_command', 10)
        self.capture = RolloutCapture(self._rollout)
        node.create_subscription(Bool, '/control/policy_recording', lambda m: setattr(self, 'record', bool(m.data)), 10)
        node.create_subscription(String, '/control/record_status', self._record_status, 10)
        node.create_subscription(String, '/policy/status', self._policy_status, 10)
        if node._hil_enabled:
            node.create_subscription(String, '/learning/command', self._learning, 10)
            node.create_subscription(ArmCommand, '/hil/arm_candidate', self._arm, 10)
            node.create_subscription(HandCommand, '/hil/hand_candidate', self._hand, 10)

    def _policy_status(self, message):
        try:
            value = json.loads(message.data)
            self.policy_seen = time.monotonic()
            if (self.was_active and not self.human.active and value.get('phase') == 'fault'
                    and self.node._execution_provider in ('policy','hil') and not self.error):
                self.fail(value.get('error') or 'remote policy failed')
        except (ValueError, TypeError):
            pass

    def _record_status(self, message):
        try:
            self.capture.observe(json.loads(message.data))
        except (ValueError, TypeError):
            pass

    def _rollout(self, value):
        self.rollout_pub.publish(self.String(data=json.dumps(dict(value, execution_audit=True,
            actor_trace_path=str(getattr(self.node._trace, 'path', '') or '')))))

    def _arm(self, m):
        try:
            for side in ('left', 'right'):
                if getattr(m, side+'_enable'):
                    self.human.arm(side, getattr(m, side+'_joint'), m.action_id)
        except ValueError as exc:
            self.fail(str(exc))

    def _hand(self, m):
        try:
            if m.left_enable: self.human.hand('left', m.q_cmd[:22])
            if m.right_enable: self.human.hand('right', m.q_cmd[22:])
        except ValueError as exc:
            self.fail(str(exc))

    def _learning(self, message):
        value = {}
        try:
            value = json.loads(message.data)
            if not isinstance(value, dict):
                raise ValueError('invalid HIL command')
            if self.learning_command and value.get('id') == self.learning_command['id']:
                self.publish_status()
                return
            if abs(time.time()-float(value.get('issued_at', 0))) > setting('actor', 'learning_command_max_age_s'):
                raise ValueError('HIL command expired')
            n = self.node
            if n._execution_provider != 'hil' or not n._machine_running or n._execution_paused or n._execution_scope != 'full':
                raise ValueError('HIL requires active execution')
            action = value['action']
            if action == 'takeover':
                anchor = self.last_target if self.last_target is not None else np.r_[n._left,n._right,n._hand]
                self.human.begin(anchor)
                self.handback = None
                self.guard.reset()
                n._abort_active('human takeover', publish_latency=False)
            elif action == 'release' and self.human.active:
                anchor = self.last_target if self.last_target is not None else self.human.anchor
                self.human.end()
                self.handback = PolicyHandback(anchor, self.timeline_steps, n._hil_return_duration_s)
            elif action == 'failure':
                self.capture.finish('fail')
                self.commands.publish(self.String(data='stop_deploy'))
            else:
                raise ValueError('unsupported HIL action')
            self.learning_command = dict(id=value.get('id'), accepted=True, reason='')
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.learning_command = dict(id=value.get('id') if isinstance(value, dict) else None, accepted=False, reason=str(exc))
            self.node.get_logger().warning(str(exc))
        self.publish_status()

    def recording_required(self):
        n = self.node
        return bool(n._machine_running and not n._execution_paused and not n._execution_dry_run
                    and n._execution_provider in ('policy', 'hil') and (self.record or n._execution_provider == 'hil'))

    def publish_status(self):
        self.status_pub.publish(self.String(data=json.dumps(dict(self.human.snapshot(),
            control_timestamp_ns=getattr(self.node, "_control_timestamp_ns", 0),
            available=self.node._hil_enabled, enabled=self.node._hil_enabled and self.node._execution_provider=='hil',
            error=self.error, learning_command=self.learning_command, rollout_recording=dict(required=self.recording_required(), rollout_id=self.capture.rollout_id, requested=bool(self.capture.rollout_id), recording=self.capture.was_ready),
            transition=dict(active=self.handback is not None and self.handback.progress(self.timeline_steps)<1,
                progress=self.handback.progress(self.timeline_steps) if self.handback else 0,
                duration_s=self.handback.duration_s if self.handback else 0)))))
        self.last_status = time.monotonic()

    def rejected(self, code, reason):
        if code in ("hardware_stopped", "arbitration_rejected"):
            if self.pending_rejection is None:
                self.pending_rejection = (time.monotonic(), reason)
        else:
            self.fail(reason)

    def fail(self, reason):
        if self.error:
            return
        self.error = str(reason)
        self.human.end()
        self.handback = None
        self.capture.annotate('truncated', source='actor_error', reason=self.error)
        self.capture.tick(False)
        self.node._abort_active(self.error, publish_latency=False)
        self.commands.publish(self.String(data='stop_deploy'))
        self.publish_status()

    def tick(self):
        n = self.node
        running = n._machine_running
        remote_running = running and n._execution_provider in ('policy', 'hil')
        active = remote_running and not n._execution_paused
        if running and not self.was_running:
            self.last_target = None
            self.handback = None
            self.error = ''
            self.policy_seen = time.monotonic()
            self.guard.reset()
        if not active:
            self.human.end()
            self.handback = None
            self.pending_rejection = None
        self.was_running = running
        self.was_active = active
        wanted = active and not self.error and not n._execution_dry_run and (self.record or n._execution_provider=='hil')
        try:
            now = time.monotonic()
            if self.error:
                return False
            if self.pending_rejection is not None:
                since, reason = self.pending_rejection
                if now-since >= setting('actor', 'stop_ack_grace_s'):
                    raise RuntimeError(reason)
                return False
            if n._machine_running and now-getattr(n, '_control_seen', now)>setting('observation','observation_max_age_s'):
                raise RuntimeError('control status heartbeat expired')
            if active and now-self.policy_seen>setting('actor','model_heartbeat_timeout_s'):
                raise RuntimeError('remote policy heartbeat expired')
            if active:
                stamps = [n._hand_stamp] + ([n._ur_stamp] if n._execution_scope != 'hands_only' else [])
                if any(stamp <= 0 or not -setting('observation','future_tolerance_s') <= (time.time_ns()-stamp)/1e9 <= setting('observation','observation_max_age_s') for stamp in stamps):
                    raise RuntimeError('measured robot state is stale')
            ready = self.capture.tick(wanted)
            if time.monotonic()-self.last_status >= .1: self.publish_status()
            if self.error: return False
            if not ready: return False
            if not self.human.active: return True
            now = time.monotonic()
            if active and now >= self.next_due:
                target = self.human.target()
                validate_arm_step(np.r_[n._left,n._right], target[:12], setting('actor','arm_tracking_error_rad'),code='arm_tracking_error')
                self.guard.check(n._hand, target[12:], now)
                n._publish_human_target(target, self.human.token, self.published)
                self.next_due = now+1/30
            return False
        except Exception as exc:
            self.fail(str(exc))
            return False

    def proposed(self, action, plan, step):
        n = self.node
        phase = 'human' if self.human.active else 'handback' if self.handback and self.handback.progress(self.timeline_steps)<1 else 'auto'
        n._trace.event('action_proposed', action_id=action.action_id,
            session_id=action.session_id, request_id=int(action.request_id),
            revision=int(action.revision), step_index=int(step),
            chunk_step=int(action.execute_start)+int(step), phase=phase,
            proposal=np.r_[plan.left_joint[step], plan.right_joint[step], plan.hand_joint[step]].tolist(),
            control_observation=dict(rollout_id=self.capture.rollout_id),
            takeover_segment_id=self.human.token if phase != 'auto' else None,
            control_gate=dict(machine_running=n._machine_running, paused=n._execution_paused,
                provider=n._execution_provider, scope=n._execution_scope, dry_run=n._execution_dry_run),
            hil=self.human.snapshot())

    def effective(self, target, frequency_hz=30.):
        if self.handback and not self.human.active:
            return self.handback.blend(target[None], self.timeline_steps, step_scale=30./frequency_hz)[0]
        return target

    def committed(self, target, frequency_hz=30.):
        self.last_target = np.asarray(target).copy()
        self.guard.commit(self.last_target[12:])
        self.published += 1
        self.timeline_steps += 30./frequency_hz

    def close(self):
        self.capture.tick(False)
