"""Recording handshake: live policy commands require an acknowledged recorder."""
from ur_common.protection_config import setting
import time
import uuid


class RolloutCapture:
    def __init__(self, publish):
        self.publish = publish
        self.rollout_id = ''
        self.status = {}
        self.seen = self.started = self.sent = 0.0
        self.was_ready = False
        self.finished_id = ''
        self.category = ''
        self.termination = {}

    def annotate(self, outcome, *, action_id=None, source='actor', reason=''):
        if self.rollout_id:
            self.finished_id = self.rollout_id
            self.termination = dict(outcome=outcome, timestamp_ns=time.time_ns(),
                action_id=action_id, annotation_source=source, annotated_unix_ns=time.time_ns(), reason=reason)

    def finish(self, category, *, action_id=None, source='learning_command'):
        if category != 'fail' or not self.rollout_id or not self.was_ready:
            raise RuntimeError('请先启动 HIL 正式执行并等待录制就绪')
        self.finished_id, self.category = self.rollout_id, category
        self.annotate("failure", action_id=action_id, source=source)

    def observe(self, status, now=None):
        self.status = dict(status)
        self.seen = time.monotonic() if now is None else now

    def tick(self, wanted, now=None):
        now = time.monotonic() if now is None else now
        changed = bool(wanted) != bool(self.rollout_id)
        if changed:
            if not wanted and self.finished_id != self.rollout_id:
                self.annotate('truncated', source='actor_stop')
            if wanted:
                self.termination = {}
                self.category = ''
            self.rollout_id = uuid.uuid4().hex if wanted else ''
            self.started = now
            self.was_ready = False
        if changed or now - self.sent >= 0.2:
            self.publish(dict(active=bool(wanted), rollout_id=self.rollout_id,
                              finished_rollout_id=self.finished_id, category=self.category, termination=self.termination))
            self.sent = now
        if not wanted:
            return True
        fresh = self.seen >= self.started and now - self.seen < setting('recording', 'heartbeat_timeout_s')
        matching = self.status.get('rollout_id') == self.rollout_id
        if fresh and matching and self.status.get('rollout_error'):
            raise RuntimeError('rollout recording failed: ' + self.status['rollout_error'])
        ready = (fresh and matching and self.status.get('recording') is True
                 and not self.status.get('saving'))
        if self.was_ready and not ready:
            raise RuntimeError('rollout recorder stopped or heartbeat expired')
        # A previous episode may still be draining video queues. Never execute
        # the next rollout until its own new directory has been acknowledged.
        timeout = setting('recording', 'saving_timeout_s' if fresh and self.status.get('saving') else 'start_timeout_s')
        if not ready and now - self.started > timeout:
            raise RuntimeError('rollout recorder did not acknowledge start')
        self.was_ready |= bool(ready)
        return bool(ready)
